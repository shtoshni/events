import sys
from os import path
import os
import time
import logging
import torch
import json
from collections import defaultdict, OrderedDict
import numpy as np
from transformers import get_linear_schedule_with_warmup, AdamW
from copy import deepcopy
from auto_memory_model.utils import action_sequences_to_clusters, classify_errors
from data_utils.utils import load_data
from coref_utils.utils import get_mention_to_cluster
from coref_utils.metrics import CorefEvaluator
import pytorch_utils.utils as utils
from data_utils.utils import get_clusters
from auto_memory_model.controller.utils import pick_controller
from kbp_2015_utils.constants import EVENT_SUBTYPES_NAME


EPS = 1e-8
NUM_STUCK_EPOCHS = 20

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


class Experiment:
    def __init__(self, args, data_dir=None, conll_data_dir=None,
                 model_dir=None, best_model_dir=None,
                 pretrained_mention_model=None,
                 # Model params
                 focus_group='joint',
                 seed=0, init_lr=5e-4, ft_lr=2e-5,
                 finetune=False, adapter_ft=False,
                 max_gradient_norm=1.0,
                 max_epochs=20, max_segment_len=512, eval=False, num_train_docs=None,
                 mem_type='unbounded', no_singletons=False,
                 # Other params
                 slurm_id=None, **kwargs):

        self.args = args
        self.model_args = vars(args)

        # Set dataset
        self.dataset = args.dataset
        # Set the random seed next
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
        self.pretrained_mention_model = pretrained_mention_model
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.conll_data_dir = conll_data_dir
        self.model_path = path.join(model_dir, 'model.pth')
        self.best_model_path = path.join(best_model_dir, 'model.pth')

        # Initialize model and training metadata
        self.finetune = finetune
        self.adapter_ft = adapter_ft

        self.model = pick_controller(mem_type=mem_type, focus_group=focus_group, finetune=finetune, **kwargs)

        self.max_epochs = max_epochs
        self.train_info, self.optimizer, self.optim_scheduler, self.optimizer_params = {}, {}, {}, {}

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
        bert_params = set(["doc_encoder.bert." + name for name, _ in self.model.doc_encoder.bert.named_parameters()])

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in bert_params:
                    continue
                else:
                    other_params.append(param)

        total_steps = self.max_epochs * len(self.train_examples)

        self.optimizer['mem'] = torch.optim.AdamW(
            other_params, lr=init_lr, eps=1e-6)
        # self.optimizer_params['mem'] = other_params  # Useful in gradient clipping
        self.optim_scheduler['mem'] = get_linear_schedule_with_warmup(
                self.optimizer['mem'], num_warmup_steps=0,
                num_training_steps=total_steps)
        if self.finetune:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters()
                            if (not any(nd in n for nd in no_decay)) and ("doc_encoder.bert" in n)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_decay) and ("doc_encoder.bert" in n)],
                 'weight_decay': 0.0}
            ]
            self.optimizer['doc'] = AdamW(optimizer_grouped_parameters, lr=ft_lr, eps=1e-6)
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
            # Try to initialize the mention model part
            if path.exists(self.pretrained_mention_model):
                logger.info("Found pretrained model!!")
                checkpoint = torch.load(self.pretrained_mention_model)
                self.model.load_state_dict(checkpoint['model'], strict=False)
        else:
            logging.info('Loading previous model: %s' % self.model_path)
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
            logger.info("\n\nStart Epoch %d" % (epoch + 1))
            start_time = time.time()
            # Setup training
            model.train()
            np.random.shuffle(self.train_examples)
            # errors = OrderedDict([("WL", 0), ("FN", 0), ("WF", 0),
            #                       ("WO", 0), ("FL", 0), ("C", 0)])
            for cur_example in self.train_examples:
                def handle_example(example):
                    self.train_info['global_steps'] += 1
                    output = model(deepcopy(example))
                    if output is None:
                        return None
                    loss = output[0]
                    # batch_errors = classify_errors(pred_action_list, gt_actions)
                    # for key in errors:
                    #     errors[key] += batch_errors[key]

                    total_loss = loss['total']
                    if isinstance(total_loss, float):
                        print(f"Weird thing - {total_loss}")
                        return None

                    elif total_loss is None:
                        return None

                    elif torch.isnan(total_loss):
                        print("Loss is NaN")
                        sys.exit()

                    # Backprop
                    for key in optimizer:
                        optimizer[key].zero_grad()

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

                    for key in optimizer:
                        # torch.nn.utils.clip_grad_norm_(
                        #     self.optimizer_params[key], max_norm=max_gradient_norm)
                        optimizer[key].step()
                        scheduler[key].step()

                    return total_loss.item()

                example_loss = handle_example(cur_example)
                if example_loss is None:
                    continue
                if self.train_info['global_steps'] % 10 == 0:
                    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    logger.info('{} {:.3f}, Max memory {:.3f}'.format(cur_example["doc_key"], example_loss, max_mem))
                    torch.cuda.reset_peak_memory_stats()

            sys.stdout.flush()
            # logging.info(errors)
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
                logger.info('Saving best model')
                self.save_model(self.best_model_path, model_type='best')

            # Save last model - but don't do it too frequently (saves time)
            if self.train_info['epoch'] % 10 == 0:
                self.save_model(self.model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            logger.info("Epoch: %d, F1: %.1f, Max F1: %.1f, Time: %.2f"
                        % (epoch + 1, fscore, self.train_info['val_perf'], elapsed_time))

            if self.train_info['num_stuck_epochs'] >= NUM_STUCK_EPOCHS:
                return

    def eval_model(self, split='dev'):
        """Eval model"""
        # Set the random seed to get consistent results
        model = self.model
        model.eval()

        data_iter = self.data_iter_map[split]

        pred_class_counter, gt_class_counter = defaultdict(int), defaultdict(int)

        tbf_path, tbf_f, mention_counter = None, None, 0
        if split == "test":
            tbf_path = path.join(self.model_dir, "hopper.tbf")
            tbf_f = open(tbf_path, "w")

        with torch.no_grad():
            log_file = path.join(self.model_dir, split + ".log.jsonl")
            with open(log_file, 'w') as log_f:
                # Capture the auxiliary action accuracy
                corr_actions = 0.0
                total_actions = 0.0

                # Output file to write the outputs
                evaluator = CorefEvaluator()
                # typeless_evaluator = CorefEvaluator()
                oracle_evaluator = CorefEvaluator()
                coref_predictions, subtoken_maps = {}, {}

                for example in data_iter:
                    loss, action_list, pred_mentions, gt_actions = model(deepcopy(example))
                    for pred_action, gt_action in zip(action_list, gt_actions):
                        pred_class_counter[pred_action[1]] += 1
                        gt_class_counter[gt_action[1]] += 1

                        if tuple(pred_action) == tuple(gt_action):
                            corr_actions += 1

                    total_actions += len(action_list)

                    predicted_clusters = action_sequences_to_clusters(action_list, pred_mentions)

                    if split == "test":
                        doc_key = example['doc_key']
                        tbf_f.write(f"#BeginOfDocument {doc_key}\n")

                        token_idx_to_orig_span_start = example["token_idx_to_orig_span_start"]
                        token_idx_to_orig_span_end = example["token_idx_to_orig_span_end"]
                        orig_doc_str = example["orig_doc"]

                        coreference_clusters = []
                        for pred_cluster in predicted_clusters:
                            coreference_cluster = {}
                            subtype_count = defaultdict(int)
                            for (span_start, span_end, event_subtype_idx) in pred_cluster:
                                if str(span_start) in token_idx_to_orig_span_start and \
                                        str(span_end) in token_idx_to_orig_span_end:
                                    orig_span_start = token_idx_to_orig_span_start[str(span_start)]
                                    orig_span_end = token_idx_to_orig_span_end[str(span_end)]

                                    span_boundary_str = f"{orig_span_start},{orig_span_end}"

                                    orig_span_str = orig_doc_str[orig_span_start: orig_span_end]
                                    event_subtype_name = EVENT_SUBTYPES_NAME[event_subtype_idx]
                                    # For now putting up any realis
                                    tbf_f.write(f"brat_conversion\t{doc_key}\tE{mention_counter}\t{span_boundary_str}\t"
                                                f"{orig_span_str}\t{event_subtype_name}\tActual\n")
                                    if span_boundary_str in coreference_cluster:
                                        # We have two spans with the same token in this cluster
                                        # Prefer the span whose event subtype is more common in the cluster
                                        earlier_subtype = coreference_cluster[span_boundary_str][1]
                                        if subtype_count[event_subtype_name] <= subtype_count[earlier_subtype]:
                                            # Current cluster has more of the
                                            continue

                                    coreference_cluster[span_boundary_str] = (f"E{mention_counter}", event_subtype_name)
                                    subtype_count[event_subtype_name] += 1
                                    mention_counter += 1
                            if len(coreference_cluster) > 1:
                                coreference_clusters.append(
                                    [mention_name for mention_name, _ in list(coreference_cluster.values())])

                        for cluster_idx, coreference_cluster in enumerate(coreference_clusters):
                            tbf_f.write(f"@Coreference\tC{cluster_idx}\t{','.join(coreference_cluster)}\n")

                        tbf_f.write(f"#EndOfDocument\n")

                    predicted_clusters, mention_to_predicted = \
                        get_mention_to_cluster(predicted_clusters, threshold=self.cluster_threshold)

                    if self.dataset == "kbp_2015":
                        gt_clusters = get_clusters(example["clusters"], key="subtype_val")
                    else:
                        gt_clusters = example["clusters"]

                    gold_clusters, mention_to_gold = \
                        get_mention_to_cluster(gt_clusters, threshold=self.cluster_threshold)

                    coref_predictions[example["doc_key"]] = predicted_clusters
                    subtoken_maps[example["doc_key"]] = example["subtoken_map"]

                    oracle_clusters = action_sequences_to_clusters(gt_actions, pred_mentions)
                    oracle_clusters, mention_to_oracle = \
                        get_mention_to_cluster(oracle_clusters, threshold=self.cluster_threshold)
                    evaluator.update(predicted_clusters, gold_clusters,
                                     mention_to_predicted, mention_to_gold)
                    oracle_evaluator.update(oracle_clusters, gold_clusters,
                                            mention_to_oracle, mention_to_gold)

                    log_example = dict(example)
                    log_example["gt_actions"] = gt_actions
                    log_example["pred_actions"] = action_list
                    log_example["predicted_clusters"] = predicted_clusters

                    log_f.write(json.dumps(log_example) + "\n")
                    # break

                logger.info(f"Ground Truth Actions: {gt_class_counter}")
                logger.info(f"Predicted Actions: {pred_class_counter}")

                # Print individual metrics
                result_dict = OrderedDict()
                indv_metrics_list = ['MUC', 'Bcub', 'CEAFE', 'BLANC']
                perf_str = ""
                for indv_metric, indv_evaluator in zip(indv_metrics_list, evaluator.evaluators):
                    perf_str += ", " + indv_metric + ": {:.1f}".format(indv_evaluator.get_f1() * 100)
                    result_dict[indv_metric] = OrderedDict()
                    result_dict[indv_metric]['recall'] = round(indv_evaluator.get_recall() * 100, 1)
                    result_dict[indv_metric]['precision'] = round(indv_evaluator.get_precision() * 100, 1)
                    result_dict[indv_metric]['fscore'] = round(indv_evaluator.get_f1() * 100, 1)

                # print(result_dict['BLANC'])

                fscore = evaluator.get_f1() * 100
                result_dict['fscore'] = round(fscore, 1)
                logger.info("F-score: %.1f %s" % (fscore, perf_str))

                logger.info("Action accuracy: %.3f, Oracle F-score: %.3f" %
                            (corr_actions / total_actions, oracle_evaluator.get_prf()[2]))
                if split == "test":
                    tbf_f.close()
                    logger.info(tbf_path)
                else:
                    logger.info(log_file)
                logger.handlers[0].flush()

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

        for split in ['dev', 'test']:  # , 'Test']:
            logging.info('\n')
            logging.info('%s' % split.capitalize())
            result_dict = self.eval_model(split)
            if split != 'test':
                logging.info('Calculated F1: %.1f' % result_dict['fscore'])

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
            'model_args': self.model_args,
        }
        if model_type != 'best':
            param_groups = ['mem', 'doc'] if self.finetune else ['mem']
            for param_group in param_groups:
                save_dict['optimizer'][param_group] = self.optimizer[param_group].state_dict()
                save_dict['scheduler'][param_group] = self.optim_scheduler[param_group].state_dict()

        torch.save(save_dict, location)
        logging.info(f"Model saved at: {location}")
