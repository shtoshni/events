import torch
import random
import sys
from os import path
import json
from collections import OrderedDict

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.step_result import EvalResult
from pytorch_lightning import _logger as logger
from transformers import get_linear_schedule_with_warmup

from auto_memory_model.utils import action_sequences_to_clusters, classify_errors
from red_utils.utils import load_data
from coref_utils.utils import mention_to_cluster
from coref_utils.metrics import CorefEvaluator
import pytorch_utils.utils as utils
from auto_memory_model.controller import LearnedFixedMemController, UnboundedMemController


class CorefModel(LightningModule):
    def __init__(self, data_dir=None,
                 # Model params
                 focus_group='joint',
                 seed=0, init_lr=1e-3, ft_lr=5e-5, finetune=False,
                 max_epochs=20, max_segment_len=128, num_train_docs=None,
                 mem_type=False,
                 no_singletons=False,
                 # Other params
                 slurm_id=None, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # Prepare data info
        self.train_examples, self.dev_examples, self.test_examples = load_data(data_dir, max_segment_len)
        if num_train_docs is not None:
            self.train_examples = self.train_examples[:num_train_docs]

        self.data_iter_map = {"train": self.train_examples,
                              "dev": self.dev_examples,
                              "test": self.test_examples}
        self.cluster_threshold = (2 if no_singletons else 1)
        self.focus_group = focus_group

        self.slurm_id = slurm_id

        # Initialize model and training metadata
        if mem_type == 'learned':
            self.model = LearnedFixedMemController(focus_group=focus_group, finetune=finetune, **kwargs)
        elif mem_type == 'unbounded':
            self.model = UnboundedMemController(focus_group=focus_group, finetune=finetune, **kwargs)

        # Training hyperparams
        self.finetune = finetune
        self.max_epochs = max_epochs
        self.init_lr = init_lr
        self.ft_lr = ft_lr

        utils.print_model_info(self.model)
        sys.stdout.flush()

    def configure_optimizers(self):
        other_params = []
        bert_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bert' in name:
                    if ('LayerNorm' not in name) and  ('layer_norm' not in name) and ('bias' not in name):
                        bert_params.append(param)
                else:
                    other_params.append(param)

        optimizer_mem = torch.optim.AdamW(
            other_params, lr=self.init_lr, eps=1e-6)

        if self.finetune:
            optimizer_mem.add_param_group(
                {'params': bert_params, 'lr': self.ft_lr, 'eps': 1e-6})

        scheduler = get_linear_schedule_with_warmup(
            optimizer_mem, num_warmup_steps=0,
            num_training_steps=self.max_epochs * len(self.train_examples))

        return [optimizer_mem], [scheduler]

    def forward(self, example, teacher_forcing=False):
        outputs = self.model(example, teacher_forcing=teacher_forcing)
        return outputs

    def training_step(self, example, idx):
        output = self(example)
        if output is None:
            loss = torch.tensor([0.0], requires_grad=True)
        else:
            loss, pred_action_list, pred_mentions, gt_actions, gt_mentions = output
        loss = loss['coref']
        train_log = {'loss/train_loss': loss.detach()}
        return {'loss': loss, 'log': train_log}

    def validation_step(self, example, idx, split="val"):
        # print(type(example))
        # print(example)
        output_tf = self(example, teacher_forcing=True)
        if output_tf is None:
            return {f'{split}_loss': torch.tensor([0.0])}
        # Loss is calculated in the teacher forcing setting
        loss = output_tf[0]['coref']

        # Other metrics are computed in normal setting
        output = self(example)
        action_list, pred_mentions, gt_actions, gt_mentions = output[1:]

        val_log = {f'loss/{split}_loss': loss.detach().item()}
        return {f'{split}_loss': loss.detach().item(), 'log': val_log,
                'action_list': action_list, 'pred_mentions': pred_mentions,
                'gt_actions': gt_actions, 'gt_mentions': gt_mentions,
                'example': example}

    def validation_epoch_end(self, outputs, split='val'):
        result_dict = {}
        log_file = path.join(self.logger.log_dir, split + ".log.jsonl")
        with open(log_file, 'w') as log_f:
            if self.focus_group == 'joint':
                evaluator_dict = OrderedDict(
                    [('entity', CorefEvaluator()), ('event', CorefEvaluator()), ('joint', CorefEvaluator())])
                oracle_evaluator_dict = OrderedDict(
                    [('entity', CorefEvaluator()), ('event', CorefEvaluator()), ('joint', CorefEvaluator())])
            else:
                evaluator_dict = OrderedDict([(self.focus_group, CorefEvaluator())])
                oracle_evaluator_dict = OrderedDict([(self.focus_group, CorefEvaluator())])

            total_loss = 0.0
            for output in outputs:
                if 'pred_mentions' not in output:
                    # Possible when doing just events/entities where some files don't have entity/event annotations
                    continue
                total_loss += output[f'{split}_loss']
                example, gt_actions = output['example'], output['gt_actions']
                action_list, pred_mentions = output['action_list'], output['pred_mentions']

                predicted_clusters = action_sequences_to_clusters(action_list, pred_mentions)
                oracle_clusters = action_sequences_to_clusters(gt_actions, pred_mentions)

                for focus_group in evaluator_dict:
                    filt_clusters, filt_mention_to_predicted =\
                        mention_to_cluster(predicted_clusters, threshold=self.cluster_threshold,
                                           focus_group=focus_group)
                    filt_gold_clusters, filt_gold_mention_to_predicted =\
                        mention_to_cluster(example['clusters'], threshold=self.cluster_threshold,
                                           focus_group=focus_group)

                    filt_oracle_clusters, filt_mention_to_oracle = \
                        mention_to_cluster(oracle_clusters, threshold=self.cluster_threshold,
                                           focus_group=focus_group)

                    if len(filt_gold_clusters) > 0:
                        evaluator_dict[focus_group].update(
                            filt_clusters, filt_gold_clusters,
                            filt_mention_to_predicted, filt_gold_mention_to_predicted)
                        oracle_evaluator_dict[focus_group].update(
                            filt_oracle_clusters, filt_gold_clusters,
                            filt_mention_to_oracle, filt_gold_mention_to_predicted)

                log_example = dict(example)
                log_example["gt_actions"] = gt_actions
                log_example["pred_actions"] = action_list
                log_example["predicted_clusters"] = predicted_clusters

                log_f.write(json.dumps(log_example) + "\n")

            result_dict['val_loss'] = total_loss/(len(outputs) + 1e-4)

            # Print individual metrics
            for focus_group in evaluator_dict:
                indv_metrics_list = ['MUC', 'Bcub', 'CEAFE']
                perf_str = ""
                for indv_metric, indv_evaluator in zip(indv_metrics_list, evaluator_dict[focus_group].evaluators):
                    perf_str += ", " + indv_metric + ": {:.1f}".format(indv_evaluator.get_f1() * 100)

                prec, rec, fscore = evaluator_dict[focus_group].get_prf()
                fscore = fscore * 100
                result_dict[f'{focus_group}_fscore'] = fscore
                if focus_group == self.focus_group:
                    result_dict['fscore'] = fscore
                print(focus_group.capitalize())
                print("F-score: %.1f %s" % (fscore, perf_str))
                print("Oracle F-score: %.2f\n" % (oracle_evaluator_dict[focus_group].get_prf()[2]))

            # logging.info("Action accuracy: %.3f" % (corr_actions/total_actions))
            logger.info(log_file)

            eval_result = EvalResult()
            eval_result.log_dict(result_dict)

        return eval_result

    def test_step(self, example):
        return self.validation_step(example, split='test')

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, split='test')

    def train_dataloader(self):
        random.shuffle(self.train_examples)
        for example in self.train_examples:
            yield example

    def val_dataloader(self):
        for example in self.dev_examples:
            yield example

    def test_dataloader(self):
        for example in self.test_examples:
            yield example

