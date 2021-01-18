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

import pytorch_utils.utils as utils
from srl.srl_model.controller import Controller
from torch.utils.tensorboard import SummaryWriter
from srl.srl_model.data_utils import load_data
from srl.constants import LABELS

EPS = 1e-8
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Experiment:
    def __init__(self, args, data_dir=None, model_dir=None, best_model_dir=None,
                 # Model params
                 seed=0, init_lr=1e-3, ft_lr=2e-5, finetune=False, max_gradient_norm=1.0,
                 max_epochs=20, eval=False, num_train_docs=None,
                 # Other params
                 slurm_id=None, **kwargs):

        self.args = args
        self.slurm_id = slurm_id
        # Set the random seed first
        self.seed = seed
        # Prepare data info
        self.train_examples, self.dev_examples, self.test_examples = load_data(data_dir)

        if num_train_docs is not None:
            self.train_examples = self.train_examples[:num_train_docs]
            # self.dev_examples = self.train_examples
        self.data_iter_map = {"train": self.train_examples,
                              "valid": self.dev_examples,
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
        self.finetune = finetune
        self.model = Controller(finetune=finetune, **kwargs)
        self.model = self.model.cuda()

        self.train_info, self.optimizer, self.optim_scheduler = {}, {}, {}

        self.initialize_setup(init_lr=init_lr, ft_lr=ft_lr)
        self.model = self.model.cuda()
        utils.print_model_info(self.model)

        if not eval:
            self.train(max_epochs=max_epochs, max_gradient_norm=max_gradient_norm)

        # Finally evaluate model
        self.final_eval()

    def initialize_setup(self, init_lr, ft_lr=2e-5):
        """Initialize model and training info."""
        self.optimizer['other'] = torch.optim.AdamW(
            self.model.other.parameters(), lr=init_lr, eps=1e-6)

        self.optim_scheduler['other'] = get_linear_schedule_with_warmup(
                self.optimizer['other'], num_warmup_steps=0,
                num_training_steps=self.max_epochs * len(self.train_examples))

        if self.finetune:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.doc_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in self.model.doc_encoder.named_parameters() if any(nd in n for nd in no_decay)],
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
                    total_loss = model(train_example)

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
                    if self.finetune:
                        optimizer['doc'].step()
                        scheduler['doc'].step()

                    return total_loss

                total_loss = handle_example(cur_example)

                if (idx + 1) % 500 == 0:
                    print(f"Steps {idx + 1}, Loss: {total_loss.item():.2f} "
                          f"Max memory {(torch.cuda.max_memory_allocated() / (1024 ** 3)):.3f}")
                    torch.cuda.reset_peak_memory_stats()

            # Update epochs done
            self.train_info['epoch'] = epoch + 1
            # Validation performance
            fscore = self.eval_model()
            scheduler['other'].step(fscore)

            # Update model if validation performance improves
            if fscore > self.train_info['val_perf']:
                self.train_info['val_perf'] = fscore
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

    def eval_model(self, split='valid', final_eval=False):
        """Eval model"""
        # Set the random seed to get consistent results
        model = self.model
        model.eval()

        dev_examples = self.data_iter_map[split]

        log_file = path.join(self.model_dir, split + ".log.jsonl")
        log_f = open(log_file, "w")

        with torch.no_grad():
            # total_gold = defaultdict(float)
            all_preds = 0.0
            all_golds = 0.0
            all_corr = 0.0

            # Output file to write the outputs
            for dev_example in dev_examples:
                pred_arg_list = model(dev_example)
                output_dict = dict(dev_example)
                output_dict['pred_args'] = []
                for gt_list, pred_list in zip(dev_example["args"], pred_arg_list):
                    cur_arg_list = [(arg_info[0], arg_info[1]) for arg_info in gt_list]
                    all_golds += len(cur_arg_list)
                    all_preds += len(pred_list)

                    all_corr += len(set(pred_list).intersection(set(cur_arg_list)))
                    output_dict['pred_args'].append(pred_list)

                log_f.write(json.dumps(output_dict) + "\n")

            recall = all_corr/all_golds
            prec = all_corr/(all_preds + EPS)
            fscore = 2 * 100 * recall * prec / (recall + prec + EPS)

        log_f.close()
        logging.info(f"Log file: {log_file}")
        return fscore

    def final_eval(self):
        """Evaluate the model on train, dev, and test"""
        # Test performance  - Load best model
        self.load_model(self.best_model_path, best_model=True)
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

        for split in ['Valid', 'Test']:
            logging.info('\n')
            logging.info('%s' % split)
            split_f1 = self.eval_model(split.lower(), final_eval=True)
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

