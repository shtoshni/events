import argparse
import os
from os import path
import logging

import subprocess

from mention_model.experiment import Experiment
from mention_model.utils import get_mention_model_name

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '-base_data_dir', default='/home/shtoshni/Research/events/proc_data/',
        help='Root directory of data', type=str)

    parser.add_argument('-base_model_dir',
                        default='/home/shtoshni/Research/events/models',
                        help='Root folder storing model runs', type=str)
    parser.add_argument('-dataset', default='kbp_2015', choices=['kbp_2015'], type=str)
    parser.add_argument('-doc_proc', default='srl', choices=['cleaned', 'orig', 'srl'], type=str)

    # Doc encoder
    parser.add_argument('-model_size', default='base', type=str, help='BERT model type')
    parser.add_argument('-pretrained_model', default="spanbert",
                        type=str, help='SpanBERT model location')
    parser.add_argument('-max_segment_len', default=512, type=int,
                        help='Max segment length of BERT segments.')
    parser.add_argument('-add_speaker_tags', default=False, action='store_true',
                        help='Whether to add speaker tags to document or not.')
    parser.add_argument('-use_local_attention', default=False, action="store_true",
                        help='Local Attention on top of BERT embeddings.')
    parser.add_argument('-use_srl', default=False, action="store_true",
                        help='Use SRL guided attention.')

    parser.add_argument('-ment_emb', default='attn', choices=['attn', 'max', 'endpoint'],
                        type=str, help='If true use an RNN on top of mention embeddings.')
    parser.add_argument('-max_span_width',
                        help='Max span width', default=4, type=int)
    parser.add_argument('-mlp_depth', default=1, type=int,
                        help='Number of hidden layers in other MLPs')
    parser.add_argument('-mlp_size', default=1000, type=int,
                        help='MLP size used in the model')

    # Training params
    parser.add_argument('-num_train_docs', default=None, type=int,
                        help='Number of training docs.')
    parser.add_argument('-dropout_rate', default=0.5, type=float,
                        help='Dropout rate')
    parser.add_argument('-max_training_segments', default=3, type=int,
                        help='Max. number of BERT segments in a document.')
    parser.add_argument('-max_epochs',
                        help='Maximum number of epochs', default=20, type=int)
    parser.add_argument('-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-no_multitask', dest='multitask', default=True, action="store_false",
                        help='Mutlitask learning with event type loss and event realis loss')
    parser.add_argument('-srl_loss_wt', help="SRL loss wt",
                        default=1.0, type=float)
    parser.add_argument('-label_smoothing_wt', help="Label Smoothing",
                        default=0.0, type=float)
    parser.add_argument('-init_lr', help="Initial learning rate",
                        default=5e-4, type=float)
    parser.add_argument('-ft_lr', help="Initial learning rate",
                        default=5e-5, type=float)
    parser.add_argument('-no_finetune', default=True, dest="finetune",
                        action="store_false", help="Fine-tuning document encoder")
    parser.add_argument('-eval', help="Evaluate model",
                        default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID",
                        default=None, type=str)

    args = parser.parse_args()

    model_name = get_mention_model_name(args)
    print(model_name)

    model_dir = path.join(args.base_model_dir, model_name)
    args.model_dir = model_dir
    best_model_dir = path.join(model_dir, 'best_models')
    args.best_model_dir = best_model_dir
    if not path.exists(model_dir):
        os.makedirs(model_dir)
    if not path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    suffix = ''
    if args.add_speaker_tags:
        suffix = '_speaker'
    args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{args.doc_proc}{suffix}')
    print(args.data_dir)

    # Log directory for Tensorflow Summary
    log_dir = path.join(model_dir, "logs")
    if not path.exists(log_dir):
        os.makedirs(log_dir)

    Experiment(args, **vars(args))


if __name__ == "__main__":
    main()