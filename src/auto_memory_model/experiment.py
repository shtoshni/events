import sys
from os import path
import os
import time
import logging
import torch
import json
from collections import defaultdict, OrderedDict
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from auto_memory_model.utils import action_sequences_to_clusters, classify_errors
from red_utils.utils import load_data
from red_utils.constants import IDX_TO_ELEM_TYPE
from coref_utils.conll import evaluate_conll
from coref_utils.utils import mention_to_cluster
from coref_utils.metrics import CorefEvaluator
import pytorch_utils.utils as utils
from auto_memory_model.controller.lfm_controller import LearnedFixedMemController
from auto_memory_model.controller.lru_controller import LRUController
from auto_memory_model.controller.um_controller import UnboundedMemController

EPS = 1e-8
NUM_STUCK_EPOCHS = 10
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Experiment:
    def __init__(self, data_dir=None, conll_data_dir=None,
                 model_dir=None, best_model_dir=None,
                 # Model params
                 batch_size=32, seed=0, init_lr=1e-3, max_gradient_norm=1.0,
                 max_epochs=20, max_segment_len=128, eval=False, num_train_docs=None,
                 mem_type=False, span_type_wt=1.0,
                 no_singletons=False,
                 # Other params
                 slurm_id=None, conll_scorer=None, **kwargs):

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
        self.span_type_wt = span_type_wt

        self.slurm_id = slurm_id
        self.conll_scorer = conll_scorer

        if not slurm_id:
            # Initialize Summary Writer
            self.writer = SummaryWriter(path.join(model_dir, "logs"),
                                        max_queue=500)
        # Get model paths
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.conll_data_dir = conll_data_dir
        self.model_path = path.join(model_dir, 'model.pth')
        self.best_model_path = path.join(best_model_dir, 'model.pth')

        # Initialize model and training metadata
        if mem_type == 'fixed_mem':
            self.model = LearnedFixedMemController(**kwargs).cuda()
        elif mem_type == 'lru':
            self.model = LRUController(**kwargs).cuda()
        elif mem_type == 'unbounded':
            self.model = UnboundedMemController(**kwargs).cuda()
        self.initialize_setup(init_lr=init_lr, max_epochs=max_epochs)
        utils.print_model_info(self.model)
        sys.stdout.flush()

        if not eval:
            self.train(max_epochs=max_epochs,
                       max_gradient_norm=max_gradient_norm)

        # Finally evaluate model
        self.final_eval()

    def initialize_setup(self, init_lr, max_epochs=10):
        """Initialize model and training info."""
        self.train_info = {}
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=init_lr, eps=1e-6, weight_decay=1e-2)

        total_training_steps = len(self.train_examples) * max_epochs
        print(total_training_steps)
        num_warmup_steps = 0  # total_training_steps // 10

        self.optim_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps)
        # torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.1, patience=3,
        #     min_lr=0.01 * init_lr, verbose=True)
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
        if not self.slurm_id:
            writer = self.writer

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
                (loss, pred_action_list, pred_mentions, gt_actions, gt_mentions,
                    span_type_corr, span_type_total) = model(example)

                batch_errors = classify_errors(pred_action_list, gt_actions)
                for key in errors:
                    errors[key] += batch_errors[key]

                loss['total'] = loss['coref'] + self.span_type_wt * loss['span_type']
                total_loss = loss['total']
                batch_loss += total_loss.item()
                if not self.slurm_id:
                    writer.add_scalar("Loss/Total", total_loss, self.train_info['global_steps'])
                    writer.add_scalar(
                        "Loss/Coref", loss['coref'].item(), self.train_info['global_steps'])
                    if 'span_type' in loss:
                        writer.add_scalar("Loss/Span Type", loss['span_type'].item(),
                                          self.train_info['global_steps'])

                if torch.isnan(total_loss):
                    print("Loss is NaN")
                    sys.exit()
                # Backprop
                optimizer.zero_grad()
                total_loss.backward()
                # Perform gradient clipping and update parameters
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_gradient_norm)

                optimizer.step()

                if self.train_info['global_steps'] % 10 == 0:
                    print(example["doc_key"], '{:.3f}'.format(total_loss.item()))

            sys.stdout.flush()
            logging.info(errors)
            # Update epochs done
            self.train_info['epoch'] = epoch + 1

            # Evaluate auto regressive performance on dev set
            val_loss, span_type_acc = self.eval_auto_reg()
            scheduler.step()

            # Dev performance
            fscore = self.eval_model()
            # Save model
            self.save_model(self.model_path)

            # Assume that the model didn't improve
            self.train_info['num_stuck_epochs'] += 1

            # Update model if dev performance improves
            if fscore > self.train_info['val_perf']:
                self.train_info['num_stuck_epochs'] = 0
                self.train_info['val_perf'] = fscore
                logging.info('Saving best model')
                self.save_model(self.best_model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            logging.info(
                "Epoch: %d, F1: %.1f, Max F1: %.1f, Time: %.2f, Loss: %.3f, Val Loss: %.3f Span Type Acc: %.3f"
                % (epoch + 1, fscore, self.train_info['val_perf'], elapsed_time,
                   batch_loss/len(self.train_examples), val_loss, span_type_acc))

            sys.stdout.flush()
            if not self.slurm_id:
                self.writer.flush()

            if self.train_info['num_stuck_epochs'] >= NUM_STUCK_EPOCHS:
                return

    def eval_auto_reg(self):
        """Evaluate teacher-forced model"""
        model = self.model
        model.eval()
        errors = OrderedDict([("WL", 0), ("FN", 0), ("WF", 0),
                              ("WO", 0), ("FL", 0), ("C", 0)])
        batch_loss = 0
        pred_class_counter, gt_class_counter = defaultdict(int), defaultdict(int)
        corr_actions, total_actions = 0, 0
        # Span type
        span_type_corr_agg = 0
        span_type_total_agg = 0

        with torch.no_grad():
            for example in self.dev_examples:
                (loss, pred_action_list, pred_mentions, gt_actions, gt_mentions,
                    span_type_corr, span_type_total) = model(example, teacher_forcing=True)

                span_type_corr_agg += span_type_corr
                span_type_total_agg += span_type_total

                batch_errors = classify_errors(pred_action_list, gt_actions)
                for key in errors:
                    errors[key] += batch_errors[key]

                for pred_action, gt_action in zip(pred_action_list, gt_actions):
                    pred_class_counter[pred_action[1]] += 1
                    gt_class_counter[gt_action[1]] += 1

                    if tuple(pred_action) == tuple(gt_action):
                        corr_actions += 1

                total_actions += len(gt_actions)
                total_loss = loss['coref']
                batch_loss += total_loss.item()

        # Span type accuracy
        span_type_acc = span_type_corr_agg/span_type_total_agg

        # logging.info("Val loss: %.3f" % batch_loss)
        logging.info("Dev: %s", str(errors))
        logging.info("(Teacher forced) Action accuracy: %.3f", corr_actions/total_actions)
        model.train()
        return batch_loss/len(self.dev_examples), span_type_acc

    def eval_model(self, split='dev'):
        """Eval model"""
        # Set the random seed to get consistent results
        model = self.model
        model.eval()

        data_iter = self.data_iter_map[split]

        pred_class_counter, gt_class_counter = defaultdict(int), defaultdict(int)
        num_gt_clusters, num_pred_clusters = 0, 0

        with torch.no_grad():
            log_file = path.join(self.model_dir, split + ".log.jsonl")
            with open(log_file, 'w') as f:
                # Capture the auxiliary action accuracy
                corr_actions = 0.0
                total_actions = 0.0

                # Output file to write the outputs
                evaluator = CorefEvaluator()
                oracle_evaluator = CorefEvaluator()
                coref_predictions, subtoken_maps = {}, {}
                for example in data_iter:
                    (loss, action_list, pred_mentions, gt_actions, gt_mentions, span_type_corr,
                        span_type_total, span_type_errors) = model(example)
                    for pred_action, gt_action in zip(action_list, gt_actions):
                        pred_class_counter[pred_action[1]] += 1
                        gt_class_counter[gt_action[1]] += 1

                        if tuple(pred_action) == tuple(gt_action):
                            corr_actions += 1
                    total_actions += len(action_list)

                    predicted_clusters = action_sequences_to_clusters(action_list, pred_mentions)
                    coref_predictions[example["doc_key"]] = predicted_clusters
                    subtoken_maps[example["doc_key"]] = example["subtoken_map"]

                    predicted_clusters, mention_to_predicted =\
                        mention_to_cluster(predicted_clusters, threshold=self.cluster_threshold)
                    gold_clusters, mention_to_gold =\
                        mention_to_cluster(example["clusters"], threshold=self.cluster_threshold)

                    # Update the number of clusters
                    num_gt_clusters += len(gold_clusters)
                    num_pred_clusters += len(predicted_clusters)

                    oracle_clusters = action_sequences_to_clusters(gt_actions, gt_mentions)
                    oracle_clusters, mention_to_oracle = \
                        mention_to_cluster(oracle_clusters,
                                           threshold=self.cluster_threshold)
                    evaluator.update(predicted_clusters, gold_clusters,
                                     mention_to_predicted, mention_to_gold)
                    oracle_evaluator.update(oracle_clusters, gold_clusters,
                                            mention_to_oracle, mention_to_gold)

                    doc = []
                    for sentence in example["sentences"]:
                        doc.extend(sentence)

                    tokenizer = self.model.doc_encoder.tokenizer
                    span_error_strings = []
                    for (span_start, span_end), gt_type, pred_type in span_type_errors:
                        span_str = tokenizer.convert_tokens_to_string(doc[span_start: span_end + 1])
                        span_error_strings.append(
                            (span_str, IDX_TO_ELEM_TYPE[gt_type], IDX_TO_ELEM_TYPE[pred_type]))

                    log_example = dict(example)
                    del log_example["subtoken_map"]
                    del log_example["sentence_map"]
                    log_example["gt_actions"] = gt_actions
                    log_example["pred_actions"] = action_list
                    log_example["predicted_clusters"] = predicted_clusters
                    log_example["span_errors"] = span_error_strings

                    f.write(json.dumps(log_example) + "\n")

                # Print individual metrics
                indv_metrics_list = ['MUC', 'Bcub', 'CEAFE']
                perf_str = ""
                for indv_metric, indv_evaluator in zip(indv_metrics_list, evaluator.evaluators):
                    perf_str += ", " + indv_metric + ": {:.1f}".format(indv_evaluator.get_f1() * 100)

                prec, rec, fscore = evaluator.get_prf()
                fscore = fscore * 100
                logging.info("F-score: %.1f %s" % (fscore, perf_str))

                if False:
                    gold_path = path.join(self.conll_data_dir, split + ".conll")
                    conll_results = evaluate_conll(
                        self.conll_scorer, gold_path, coref_predictions, subtoken_maps)
                    average_f1 = sum(results for results in conll_results.values()) / len(conll_results)
                    logging.info("(CoNLL) F-score : %.3f, MUC: %.3f, Bcub: %.3f, CEAFE: %.3f"
                                 % (average_f1, conll_results["muc"], conll_results['bcub'],
                                    conll_results['ceafe']))

                logging.info("Action accuracy: %.3f, Oracle F-score: %.3f" %
                             (corr_actions/total_actions, oracle_evaluator.get_prf()[2]))
                logging.info(log_file)

        return fscore

    def final_eval(self):
        """Evaluate the model on train, dev, and test"""
        # Test performance  - Load best model
        self.load_model(self.best_model_path)
        logging.info("Loading best model after epoch: %d" %
                     self.train_info['epoch'])

        perf_file = path.join(self.model_dir, "perf.txt")
        if self.slurm_id:
            parent_dir = path.dirname(path.normpath(self.model_dir))
            perf_dir = path.join(parent_dir, "perf")
            if not path.exists(perf_dir):
                os.makedirs(perf_dir)
            perf_file = path.join(perf_dir, self.slurm_id + ".txt")

        with open(perf_file, 'w') as f:
            for split in ['Train', 'Dev', 'Test']:
                logging.info('\n')
                logging.info('%s' % split)
                split_f1 = self.eval_model(split.lower())
                logging.info('Calculated F1: %.3f' % split_f1)

                f.write("%s\t%.4f\n" % (split, split_f1))
                if not self.slurm_id:
                    self.writer.add_scalar(
                        "F-score/{}".format(split), split_f1)
            logging.info("Final performance summary at %s" % perf_file)

        sys.stdout.flush()
        if not self.slurm_id:
            self.writer.close()

    def load_model(self, location):
        checkpoint = torch.load(location)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        print(type(checkpoint['model']))
        print(checkpoint['model'].keys())

        self.optimizer.load_state_dict(
            checkpoint['optimizer'])
        self.optim_scheduler.load_state_dict(
            checkpoint['scheduler'])
        self.train_info = checkpoint['train_info']
        torch.set_rng_state(checkpoint['rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])

    def save_model(self, location):
        """Save model"""
        model_state_dict = OrderedDict(self.model.state_dict())
        for key in self.model.state_dict():
            if 'bert.' in key:
                del model_state_dict[key]
        torch.save({
            'train_info': self.train_info,
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.optim_scheduler.state_dict(),
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state()
        }, location)
        # logging.info("Model saved at: %s" % (location))
