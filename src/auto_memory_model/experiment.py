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

from auto_memory_model.utils import action_sequences_to_clusters, classify_errors
from red_utils.utils import load_data
from coref_utils.utils import mention_to_cluster
from coref_utils.metrics import CorefEvaluator
import pytorch_utils.utils as utils
from auto_memory_model.controller.lfm_controller import LearnedFixedMemController
from auto_memory_model.controller.lru_controller import LRUController
from auto_memory_model.controller.um_controller import UnboundedMemController

EPS = 1e-8
NUM_STUCK_EPOCHS = 20
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Experiment:
    def __init__(self, data_dir=None, conll_data_dir=None,
                 model_dir=None, best_model_dir=None,
                 # Model params
                 focus_group='joint',
                 seed=0, init_lr=1e-3, max_gradient_norm=5.0,
                 max_epochs=20, max_segment_len=128, eval=False, num_train_docs=None,
                 mem_type=False,
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
        self.focus_group = focus_group

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
        if mem_type == 'learned':
            self.model = LearnedFixedMemController(focus_group=focus_group, **kwargs).cuda()
        elif mem_type == 'lru':
            self.model = LRUController(focus_group=focus_group, **kwargs).cuda()
        elif mem_type == 'unbounded':
            self.model = UnboundedMemController(focus_group=focus_group, **kwargs).cuda()
        self.initialize_setup(init_lr=init_lr)
        utils.print_model_info(self.model)
        sys.stdout.flush()

        if not eval:
            self.train(max_epochs=max_epochs,
                       max_gradient_norm=max_gradient_norm)

        # Finally evaluate model
        self.final_eval()

    def initialize_setup(self, init_lr, lr_decay=10):
        """Initialize model and training info."""
        self.train_info = {}
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=init_lr, eps=1e-6)
        self.optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3,
            min_lr=0.1 * init_lr, verbose=True)
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
                output = model(example)
                if output is None:
                    continue
                loss, pred_action_list, pred_mentions, gt_actions, gt_mentions = output
                batch_errors = classify_errors(pred_action_list, gt_actions)
                for key in errors:
                    errors[key] += batch_errors[key]

                total_loss = loss['total']
                batch_loss += total_loss.item()
                if not self.slurm_id:
                    writer.add_scalar("Loss/Total", total_loss, self.train_info['global_steps'])

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
            val_loss = self.eval_auto_reg()
            scheduler.step(val_loss)

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
            logging.info("Epoch: %d, F1: %.1f, Max F1: %.1f, Time: %.2f, Loss: %.3f, Val Loss: %.3f"
                         % (epoch + 1, fscore, self.train_info['val_perf'], elapsed_time,
                            batch_loss/len(self.train_examples), val_loss))

            sys.stdout.flush()
            if not self.slurm_id:
                self.writer.flush()

            if self.train_info['num_stuck_epochs'] >= NUM_STUCK_EPOCHS:
                return

    def eval_auto_reg(self):
        """Train model"""
        model = self.model
        model.eval()
        errors = OrderedDict([("WL", 0), ("FN", 0), ("WF", 0),
                              ("WO", 0), ("FL", 0), ("C", 0)])
        batch_loss = 0
        pred_class_counter, gt_class_counter = defaultdict(int), defaultdict(int)
        corr_actions, total_actions = 0, 0
        with torch.no_grad():
            for example in self.dev_examples:
                output = model(example, teacher_forcing=True)
                if output is None:
                    continue
                loss, pred_action_list, pred_mentions, gt_actions, gt_mentions = output
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

        # logging.info("Val loss: %.3f" % batch_loss)
        logging.info("Dev: %s", str(errors))
        logging.info("(Teacher forced) Action accuracy: %.3f", corr_actions/total_actions)
        model.train()
        return batch_loss/len(self.dev_examples)

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

                relv_fscore = 0
                # Print individual metrics
                for focus_group in evaluator_dict:
                    indv_metrics_list = ['MUC', 'Bcub', 'CEAFE']
                    perf_str = ""
                    for indv_metric, indv_evaluator in zip(indv_metrics_list, evaluator_dict[focus_group].evaluators):
                        perf_str += ", " + indv_metric + ": {:.1f}".format(indv_evaluator.get_f1() * 100)

                    prec, rec, fscore = evaluator_dict[focus_group].get_prf()
                    fscore = fscore * 100
                    if self.focus_group == focus_group:
                        relv_fscore = fscore
                    logging.info(focus_group.capitalize())
                    logging.info("F-score: %.1f %s" % (fscore, perf_str))
                    logging.info("Oracle F-score: %.2f\n" % (oracle_evaluator_dict[focus_group].get_prf()[2]))

                # logging.info("Action accuracy: %.3f" % (corr_actions/total_actions))
                logging.info(log_file)

        return relv_fscore

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
            for split in ['Train', 'Dev']:  # , 'Test']:
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
        # print(type(checkpoint['model']))
        # print(checkpoint['model'].keys())
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
