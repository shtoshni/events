import sys
from os import path
import os
import time
import logging
import torch
import json
from collections import defaultdict, OrderedDict
import numpy as np
from transformers import get_linear_schedule_with_warmup

from auto_memory_model.utils import action_sequences_to_clusters, classify_errors
from red_utils.utils import load_data
from coref_utils.utils import mention_to_cluster
from coref_utils.metrics import CorefEvaluator
import pytorch_utils.utils as utils
from auto_memory_model.controller import LearnedFixedMemController, LRUController, UnboundedMemController


EPS = 1e-8
NUM_STUCK_EPOCHS = 20
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Experiment:
    def __init__(self, args, data_dir=None, conll_data_dir=None,
                 model_dir=None, best_model_dir=None,
                 # Model params
                 focus_group='joint',
                 seed=0, init_lr=5e-4, ft_lr=2e-5, finetune=False,
                 max_gradient_norm=1.0,
                 max_epochs=20, max_segment_len=128, eval=False, num_train_docs=None,
                 mem_type='unbounded', no_singletons=False,
                 # Other params
                 slurm_id=None, **kwargs):

        self.args = args
        # Set the random seed first
        self.seed = seed
        # Prepare data info
        self.train_examples, self.dev_examples, self.test_examples \
            = load_data(data_dir, max_segment_len)
        # if feedback:
        if num_train_docs is not None:
            self.train_examples = self.train_examples[:num_train_docs]

        self.data_iter_map = {"train": self.train_examples,
                              "dev": self.dev_examples,
                              "test": self.test_examples}
        self.cluster_threshold = (2 if no_singletons else 1)
        self.focus_group = focus_group

        self.slurm_id = slurm_id

        # Get model paths
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.conll_data_dir = conll_data_dir
        self.model_path = path.join(model_dir, 'model.pth')
        self.best_model_path = path.join(best_model_dir, 'model.pth')

        # Initialize model and training metadata
        self.finetune = finetune
        if mem_type == 'learned':
            self.model = LearnedFixedMemController(focus_group=focus_group, finetune=finetune, **kwargs).cuda()
        elif mem_type == 'lru':
            self.model = LRUController(focus_group=focus_group, finetune=finetune, **kwargs).cuda()
        elif mem_type == 'unbounded':
            self.model = UnboundedMemController(focus_group=focus_group, finetune=finetune, **kwargs).cuda()

        self.max_epochs = max_epochs
        self.train_info, self.optimizer, self.optim_scheduler = {}, {}, {}

        self.initialize_setup(init_lr=init_lr, ft_lr=ft_lr)
        utils.print_model_info(self.model)
        sys.stdout.flush()

        if not eval:
            self.train(max_epochs=max_epochs,
                       max_gradient_norm=max_gradient_norm)

        # Finally evaluate model
        self.final_eval()

    def initialize_setup(self, init_lr, ft_lr=5e-5):
        """Initialize model and training info."""
        other_params = []
        bert_decay_params = []
        bert_non_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bert' in name:
                    if ('LayerNorm' not in name) and ('layer_norm' not in name) and ('bias' not in name):
                        bert_decay_params.append(param)
                    else:
                        bert_non_decay_params.append(param)
                else:
                    other_params.append(param)

        # print(len(bert_non_decay_params))
        # print(len(bert_decay_params))
        total_steps = self.max_epochs * len(self.train_examples)

        self.optimizer['mem'] = torch.optim.AdamW(
            other_params, lr=init_lr, eps=1e-6)
        self.optim_scheduler['mem'] = get_linear_schedule_with_warmup(
                self.optimizer['mem'], num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps)
        if self.finetune:
            self.optimizer['doc'] = torch.optim.AdamW(
                bert_decay_params, lr=ft_lr, eps=1e-6)
            self.optimizer['doc'].add_param_group(
                {"params": bert_non_decay_params, "lr": ft_lr, "weight_decay": 0}
            )
            self.optim_scheduler['doc'] = get_linear_schedule_with_warmup(
                self.optimizer['doc'], num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps)
        self.train_info['epoch'] = 0
        self.train_info['val_perf'] = 0.0
        self.train_info['global_steps'] = 0
        self.train_info['num_stuck_epochs'] = 0

        if not path.exists(self.model_path):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        else:
            logging.info('Loading previous model: %s' % (self.model_path))
            # Load model
            self.load_model(self.model_path)

    def train(self, max_epochs, max_gradient_norm):
        """Train model"""
        model = self.model
        epochs_done = self.train_info['epoch']
        optimizer = self.optimizer
        scheduler = self.optim_scheduler

        if self.train_info['num_stuck_epochs'] >= NUM_STUCK_EPOCHS:
            return

        for epoch in range(epochs_done, max_epochs):
            print("\n\nStart Epoch %d" % (epoch + 1))
            start_time = time.time()
            # Setup training
            model.train()
            np.random.shuffle(self.train_examples)
            batch_loss = 0
            errors = OrderedDict([("WL", 0), ("FN", 0), ("WF", 0),
                                  ("WO", 0), ("FL", 0), ("C", 0)])
            for example in self.train_examples:
                self.train_info['global_steps'] += 1
                output = model(example)
                if output is None:
                    continue
                loss, pred_action_list, pred_mentions, gt_actions, gt_mentions = output
                batch_errors = classify_errors(pred_action_list, gt_actions)
                for key in errors:
                    errors[key] += batch_errors[key]

                total_loss = loss['total']
                if isinstance(total_loss, float):
                    print(f"Weird thing - {total_loss}")
                    continue
                batch_loss += total_loss.item()

                if torch.isnan(total_loss):
                    print("Loss is NaN")
                    sys.exit()
                # Backprop
                for key in optimizer:
                    optimizer[key].zero_grad()

                total_loss.backward()
                # Perform gradient clipping and update parameters
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_gradient_norm)

                for key in optimizer:
                    optimizer[key].step()
                    scheduler[key].step()
                # if self.finetune:
                #     scheduler['doc'].step()
                #     scheduler['mem'].step()

                if self.train_info['global_steps'] % 10 == 0:
                    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    print(example["doc_key"], '{:.3f}, Max memory {:.3f}'.format(total_loss.item(), max_mem))
                    torch.cuda.reset_peak_memory_stats()

            sys.stdout.flush()
            logging.info(errors)
            # Update epochs done
            self.train_info['epoch'] = epoch + 1

            # Dev performance
            fscore = self.eval_model()['fscore']

            # Assume that the model didn't improve
            self.train_info['num_stuck_epochs'] += 1

            # Update model if dev performance improves
            if fscore > self.train_info['val_perf']:
                self.train_info['num_stuck_epochs'] = 0
                self.train_info['val_perf'] = fscore
                logging.info('Saving best model')
                self.save_model(self.best_model_path, model_type='best')

            # Save last model
            self.save_model(self.model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            logging.info("Epoch: %d, F1: %.1f, Max F1: %.1f, Time: %.2f"
                         % (epoch + 1, fscore, self.train_info['val_perf'], elapsed_time))

            sys.stdout.flush()

            if self.train_info['num_stuck_epochs'] >= NUM_STUCK_EPOCHS:
                return

    def eval_model(self, split='dev'):
        """Eval model"""
        # Set the random seed to get consistent results
        model = self.model
        model.eval()

        data_iter = self.data_iter_map[split]

        pred_class_counter, gt_class_counter = defaultdict(int), defaultdict(int)

        with torch.no_grad():
            log_file = path.join(self.model_dir, split + ".log.jsonl")
            with open(log_file, 'w') as log_f:
                # Capture the auxiliary action accuracy
                corr_actions = 0.0
                total_actions = 0.0

                # Output file to write the outputs
                if self.focus_group == 'joint':
                    evaluator_dict = OrderedDict(
                        [('entity', CorefEvaluator()), ('event', CorefEvaluator()), ('joint', CorefEvaluator())])
                    oracle_evaluator_dict = OrderedDict(
                        [('entity', CorefEvaluator()), ('event', CorefEvaluator()), ('joint', CorefEvaluator())])
                else:
                    evaluator_dict = OrderedDict([(self.focus_group, CorefEvaluator())])
                    oracle_evaluator_dict = OrderedDict([(self.focus_group, CorefEvaluator())])

                coref_predictions, subtoken_maps = {}, {}
                for example in data_iter:
                    output = model(example)
                    if output is None:
                        # Possible when doing just events where some files don't have event annotations
                        continue
                    loss, action_list, pred_mentions, gt_actions, gt_mentions = output

                    for pred_action, gt_action in zip(action_list, gt_actions):
                        pred_class_counter[pred_action[1]] += 1
                        gt_class_counter[gt_action[1]] += 1

                        if tuple(pred_action) == tuple(gt_action):
                            corr_actions += 1
                    total_actions += len(action_list)

                    predicted_clusters = action_sequences_to_clusters(action_list, pred_mentions)
                    oracle_clusters = action_sequences_to_clusters(gt_actions, pred_mentions)

                    coref_predictions[example["doc_key"]] = predicted_clusters
                    subtoken_maps[example["doc_key"]] = example["subtoken_map"]

                    for focus_group in evaluator_dict:
                        filt_clusters, filt_mention_to_predicted =\
                            mention_to_cluster(predicted_clusters, threshold=self.cluster_threshold,
                                               focus_group=focus_group)
                        filt_gold_clusters, filt_gold_mention_to_predicted =\
                            mention_to_cluster(example["clusters"], threshold=self.cluster_threshold,
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

                result_dict = {}
                # Print individual metrics
                for focus_group in evaluator_dict:
                    indv_metrics_list = ['MUC', 'Bcub', 'CEAFE']
                    perf_str = ""
                    result_dict[focus_group] = {}
                    for indv_metric, indv_evaluator in zip(indv_metrics_list, evaluator_dict[focus_group].evaluators):
                        metric_num = indv_evaluator.get_f1() * 100
                        perf_str += ", " + indv_metric + ": {:.1f}".format(metric_num)
                        result_dict[focus_group][indv_metric] = metric_num

                    prec, rec, fscore = evaluator_dict[focus_group].get_prf()
                    fscore = fscore * 100
                    result_dict[focus_group]['fscore'] = fscore
                    if self.focus_group == focus_group:
                        result_dict['fscore'] = fscore
                    logging.info(focus_group.capitalize())
                    if split != 'test':
                        logging.info("F-score: %.1f %s" % (fscore, perf_str))
                        logging.info("Oracle F-score: %.2f\n" % (oracle_evaluator_dict[focus_group].get_prf()[2]))

                # logging.info("Action accuracy: %.3f" % (corr_actions/total_actions))
                logging.info(log_file)

        return result_dict

    def final_eval(self):
        """Evaluate the model on train, dev, and test"""
        # Test performance  - Load best model
        self.load_model(self.best_model_path, model_type='best')
        logging.info("Loading best model after epoch: %d" %
                     self.train_info['epoch'])

        perf_file = path.join(self.model_dir, "perf.json")
        if self.slurm_id:
            parent_dir = path.dirname(path.normpath(self.model_dir))
            perf_dir = path.join(parent_dir, "perf")
            if not path.exists(perf_dir):
                os.makedirs(perf_dir)
            perf_file = path.join(perf_dir, self.slurm_id + ".json")

        output_dict = {'model_dir': self.model_dir}
        for key, val in vars(self.args).items():
            output_dict[key] = val

        for split in ['train', 'dev', 'test']:  # , 'Test']:
            logging.info('\n')
            logging.info('%s' % split.capitalize())
            result_dict = self.eval_model(split)
            if split != 'test':
                logging.info('Calculated F1: %.3f' % result_dict['fscore'])

            output_dict[split] = result_dict

        json.dump(output_dict, open(perf_file, 'w'), indent=2)

        logging.info("Final performance summary at %s" % perf_file)
        sys.stdout.flush()

    def load_model(self, location, model_type='last'):
        checkpoint = torch.load(location)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        if model_type != 'best':
            param_groups = ['mem', 'doc'] if self.finetune else ['mem']
            for param_group in param_groups:
                self.optimizer[param_group].load_state_dict(
                    checkpoint['optimizer'][param_group])
                self.optim_scheduler[param_group].load_state_dict(
                    checkpoint['scheduler'][param_group])
        self.train_info = checkpoint['train_info']
        torch.set_rng_state(checkpoint['rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])

    def save_model(self, location, model_type='last'):
        """Save model"""
        model_state_dict = OrderedDict(self.model.state_dict())
        if not self.finetune:
            for key in self.model.state_dict():
                if 'bert.' in key:
                    del model_state_dict[key]
        save_dict = {
            'train_info': self.train_info,
            'model': model_state_dict,
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'optimizer': {},
            'scheduler': {},
        }
        if model_type != 'best':
            param_groups = ['mem', 'doc'] if self.finetune else ['mem']
            for param_group in param_groups:
                save_dict['optimizer'][param_group] = self.optimizer[param_group].state_dict()
                save_dict['scheduler'][param_group] = self.optim_scheduler[param_group].state_dict()

        torch.save(save_dict, location)
        logging.info(f"Model saved at: {location}")