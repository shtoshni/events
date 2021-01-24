import sys
import os
import time
import logging
import torch
import json
import numpy as np

from os import path
from collections import defaultdict, OrderedDict
from transformers import get_linear_schedule_with_warmup, AdamW
from copy import deepcopy
import pytorch_utils.utils as utils
from mention_model.controller import Controller
from torch.utils.tensorboard import SummaryWriter
from data_utils.utils import load_data
from kbp_2015_utils.constants import EVENT_SUBTYPES_NAME, REALIS_VALS

EPS = 1e-8
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Experiment:
    def __init__(self, args, data_dir=None, dataset='kbp_2015',
                 model_dir=None, best_model_dir=None,
                 # Model params
                 seed=0, init_lr=1e-3, ft_lr=2e-5, finetune=False,
                 max_gradient_norm=1.0,
                 max_epochs=20, max_segment_len=512, eval=False,
                 num_train_docs=None, multitask=False,
                 # Other params
                 slurm_id=None,
                 **kwargs):

        self.args = args
        self.slurm_id = slurm_id
        # Set the random seed first
        self.seed = seed
        # Prepare data info
        self.train_examples, self.dev_examples, self.test_examples \
            = load_data(data_dir, max_segment_len, dataset=dataset)
        # self.dev_examples = self.dev_examples[:20]
        if num_train_docs is not None:
            self.train_examples = self.train_examples[:num_train_docs]
        self.data_iter_map = {"train": self.train_examples,
                              "valid": self.dev_examples,
                              # "valid": self.train_examples,
                              "test": self.test_examples}
        self.max_epochs = max_epochs

        if not slurm_id:
            # Initialize Summary Writer
            self.writer = SummaryWriter(path.join(model_dir, "logs"),
                                        max_queue=500)
        # Get model paths
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model_path = path.join(model_dir, 'model.pth')
        self.best_model_path = path.join(best_model_dir, 'model.pth')

        # Initialize model and training metadata
        self.multitask = multitask
        self.finetune = finetune
        self.model = Controller(finetune=finetune, **kwargs)
        self.model = self.model.cuda()

        self.train_info, self.optimizer, self.optim_scheduler = {}, {}, {}

        self.initialize_setup(init_lr=init_lr, ft_lr=ft_lr)
        self.model = self.model.cuda()
        # utils.print_model_info(self.model)

        if not eval:
            self.train(max_epochs=max_epochs, max_gradient_norm=max_gradient_norm)

        # Finally evaluate model
        self.final_eval()

    def initialize_setup(self, init_lr, ft_lr=2e-5):
        """Initialize model and training info."""
        other_params = []
        bert_params = set(["doc_encoder.bert." + name for name, _ in self.model.doc_encoder.bert.named_parameters()])

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in bert_params:
                    continue
                else:
                    # print("Hello", name)
                    other_params.append(param)

        self.optimizer['other'] = torch.optim.AdamW(
            other_params, lr=init_lr, eps=1e-6)

        self.optim_scheduler['other'] = get_linear_schedule_with_warmup(
                self.optimizer['other'], num_warmup_steps=0,
                num_training_steps=self.max_epochs * len(self.train_examples))

        if self.finetune:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.doc_encoder.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in self.model.doc_encoder.bert.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            self.optimizer['doc'] = AdamW(optimizer_grouped_parameters, lr=ft_lr, eps=1e-6)

            self.optim_scheduler['doc'] = get_linear_schedule_with_warmup(
                self.optimizer['doc'], num_warmup_steps=round(len(self.train_examples) * 0.1 * self.max_epochs),
                num_training_steps=self.max_epochs * len(self.train_examples))

        self.train_info['epoch'] = 0
        self.train_info['val_perf'] = 0.0
        self.train_info['threshold'] = {}
        self.train_info['global_steps'] = 0

        if not path.exists(self.model_path):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
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
        if not self.slurm_id:
            writer = self.writer

        for epoch in range(epochs_done, max_epochs):
            print("\n\nStart Epoch %d" % (epoch + 1))
            start_time = time.time()
            # with autograd.detect_anomaly():
            model.train()
            np.random.shuffle(self.train_examples)

            for idx, cur_example in enumerate(self.train_examples):
                def handle_example(train_example):
                    self.train_info['global_steps'] += 1
                    loss = model(deepcopy(train_example))
                    if self.multitask:
                        total_loss = sum([loss[loss_type] for loss_type in loss])
                    else:
                        total_loss = loss['event_subtype']
                    if not self.slurm_id:
                        writer.add_scalar(
                            "Loss/Total", total_loss,
                            self.train_info['global_steps'])

                    if torch.isnan(total_loss):
                        print("Loss is NaN")
                        sys.exit()
                    # Backprop
                    optimizer['other'].zero_grad()
                    if self.finetune:
                        optimizer['doc'].zero_grad()

                    total_loss.backward()
                    # Perform gradient clipping and update parameters
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_gradient_norm)

                    optimizer['other'].step()
                    scheduler['other'].step()

                    if self.finetune:
                        optimizer['doc'].step()
                        scheduler['doc'].step()

                    return total_loss, loss

                total_loss, loss_dict = handle_example(cur_example)

                if (idx + 1) % 10 == 0:
                    print(f"Steps {idx + 1}, Loss: {total_loss.item():.2f} "
                          f"Max memory {(torch.cuda.max_memory_allocated() / (1024 ** 3)):.3f}")
                    torch.cuda.reset_peak_memory_stats()
                    # print({loss_type: round(loss_val.item(), 4) for loss_type, loss_val in loss_dict.items()})

            # Update epochs done
            self.train_info['epoch'] = epoch + 1
            # Validation performance
            fscore, threshold = self.eval_model()

            # Update model if validation performance improves
            if fscore > self.train_info['val_perf']:
                self.train_info['val_perf'] = fscore
                self.train_info['threshold'] = threshold
                logging.info('Saving best model')
                self.save_model(self.best_model_path, best_model=True)

            # Save model
            self.save_model(self.model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            logging.info("Epoch: %d, Time: %.2f, F-score: %.1f, Max F-score: %.1f"
                         % (epoch + 1, elapsed_time, fscore, self.train_info['val_perf']))

            sys.stdout.flush()
            if not self.slurm_id:
                self.writer.flush()

    def eval_preds(self, pred_mention_probs, gold_mentions, threshold=0.5):
        pred_mentions = (pred_mention_probs >= threshold).float()
        total_corr = torch.sum(pred_mentions * gold_mentions)

        return total_corr, torch.sum(pred_mentions), torch.sum(gold_mentions)

    def eval_model(self, split='valid', threshold=None, final_eval=False):
        """Eval model"""
        # Set the random seed to get consistent results
        model = self.model
        model.eval()

        max_span_width = self.args.max_span_width
        dev_examples = self.data_iter_map[split]
        mention_counter = 0

        if split == "test":
            tbf_f = open(path.join(self.model_dir, "nugget.tbf"), "w")

        log_file = path.join(self.model_dir, split + ".log.jsonl")
        log_f = open(log_file, "w")

        with torch.no_grad():
            # total_gold = defaultdict(float)
            total_gold = 0.0
            all_golds = 0.0
            # Output file to write the outputs
            agg_results = {}
            for dev_example in dev_examples:
                preds, y, flat_cand_mask = model(deepcopy(dev_example))

                gt_mentions = set()
                for cluster in dev_example["clusters"]:
                    for mention in cluster:
                        span_start, span_end, mention_info = mention
                        gt_mentions.add((span_start, span_end))

                if threshold is not None:
                    nonzero_preds = torch.nonzero((preds >= threshold), as_tuple=False)
                    pred_mentions_idx = nonzero_preds.tolist()

                    mask_nonzero_idx = torch.squeeze(torch.nonzero(flat_cand_mask, as_tuple=False), dim=1).tolist()

                    pred_mentions = []
                    for flattened_idx in pred_mentions_idx:
                        if isinstance(flattened_idx, list):
                            flattened_idx = flattened_idx[0]
                        orig_flattened_idx = mask_nonzero_idx[flattened_idx]
                        token_idx = orig_flattened_idx // max_span_width
                        num_tokens = orig_flattened_idx % max_span_width

                        pred_mentions.append((token_idx, token_idx + num_tokens))

                    set_gt = set(gt_mentions)
                    set_pred = set(pred_mentions)
                    corr_preds = len(set_pred.intersection(set_gt))
                    prec = corr_preds / (len(set_pred) + 1e-8)
                    recall = corr_preds / (len(set_gt) + 1e-8)

                    log_example = dict(dev_example)
                    log_example["pred_mentions"] = pred_mentions
                    log_example["gt_mentions"] = list(gt_mentions)
                    log_example["f_score"] = 200 * prec * recall / (prec + recall + 1e-8)

                    log_f.write(json.dumps(log_example) + "\n")

                all_golds += len(gt_mentions)
                total_gold += torch.sum(y).item()

                if threshold:
                    corr, total_preds, total_y = self.eval_preds(
                        preds, y, threshold=threshold)
                    if threshold not in agg_results:
                        agg_results[threshold] = defaultdict(float)

                    agg_results[threshold]['corr'] += corr
                    agg_results[threshold]['total_preds'] += total_preds

                else:
                    threshold_range = np.arange(0.0, 1.00, 0.01)
                    for cur_threshold in threshold_range:
                        corr, total_preds, total_y = self.eval_preds(
                            preds, y, threshold=cur_threshold)
                        if cur_threshold not in agg_results:
                            agg_results[cur_threshold] = defaultdict(float)

                        agg_results[cur_threshold]['corr'] += corr
                        agg_results[cur_threshold]['total_preds'] += total_preds

        if threshold:
            prec = agg_results[threshold]['corr']/(agg_results[threshold]['total_preds'] + EPS)
            recall = agg_results[threshold]['corr']/all_golds
            max_fscore = 2 * prec * recall/(prec + recall + EPS)
        else:
            max_fscore, threshold = 0, 0.0
            for cur_threshold in agg_results:
                prec = agg_results[cur_threshold]['corr'] / (agg_results[cur_threshold]['total_preds'] + EPS)
                recall = agg_results[cur_threshold]['corr'] / all_golds
                fscore = 2 * prec * recall / (prec + recall + EPS)
                if fscore > max_fscore:
                    max_fscore = fscore
                    threshold = cur_threshold

        max_fscore = 100 * max_fscore  # F-score in percentage
        if isinstance(max_fscore, torch.Tensor):
            max_fscore = max_fscore.item()

        try:
            assert (all_golds == total_gold)
        except AssertionError:
            logging.info(f"Number of true mentions {all_golds} different from mentions "
                         f"filtered through {total_gold}")
        logging.info("F-score: %.1f, Threshold: %.2f" % (max_fscore, threshold))

        if final_eval and split == "test":
            tbf_f.close()
        log_f.close()
        return max_fscore, threshold

    def final_eval(self):
        """Evaluate the model on train, dev, and test"""
        # Test performance  - Load best model
        self.load_model(self.best_model_path, best_model=True)
        logging.info("Loading best model after epoch: %d" %
                     self.train_info['epoch'])
        logging.info(f"Threshold: {self.train_info['threshold']: .2f}")
        threshold = self.train_info['threshold']

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

        for split in ['Valid', 'Test']:
            logging.info('\n')
            logging.info('%s' % split)
            split_f1, _ = self.eval_model(
                split.lower(), threshold=threshold, final_eval=True)
            logging.info('Calculated F-score: %.1f' % split_f1)

            output_dict[split] = split_f1

            if not self.slurm_id:
                self.writer.add_scalar("F-score/{}".format(split), split_f1)

        json.dump(output_dict, open(perf_file, 'w'), indent=2)

        logging.info("Final performance summary at %s" % perf_file)
        sys.stdout.flush()
        if not self.slurm_id:
            self.writer.close()

    def load_model(self, location, best_model=False):
        # checkpoint = torch.load(location, map_location=torch.device('cpu'))
        # self.model.load_state_dict(checkpoint['model'], strict=False, map_location=torch.device('cuda'))
        checkpoint = torch.load(location, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.to(torch.device('cuda'))

        self.train_info = checkpoint['train_info']
        torch.set_rng_state(checkpoint['rng_state'])

        if not best_model:
            param_groups = ['other', 'doc'] if self.finetune else ['other']
            for param_group in param_groups:
                self.optimizer[param_group].load_state_dict(
                    checkpoint['optimizer'][param_group])
                self.optim_scheduler[param_group].load_state_dict(
                    checkpoint['scheduler'][param_group])

    def save_model(self, location, best_model=False):
        """Save model"""
        # self.model.to(torch.device('cpu'))

        model_state_dict = OrderedDict(self.model.state_dict())
        if not self.finetune:
            for key in self.model.state_dict():
                if 'bert.' in key:
                    del model_state_dict[key]
        save_dict = {
            'train_info': self.train_info,
            'model': model_state_dict,
            'rng_state': torch.get_rng_state(),
            'optimizer': {},
            'scheduler': {},
        }

        if not best_model:
            param_groups = ['other', 'doc'] if self.finetune else ['other']
            for param_group in param_groups:
                save_dict['optimizer'][param_group] = self.optimizer[param_group].state_dict()
                save_dict['scheduler'][param_group] = self.optim_scheduler[param_group].state_dict()

        torch.save(save_dict, location)
        logging.info(f"Model saved at: {location}")

