import argparse
import os
from os import path
import hashlib
import logging
import subprocess
from collections import OrderedDict

from auto_memory_model.experiment import Experiment
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
    parser.add_argument(
        '-dataset', default='kbp_2015', choices=['kbp_2015'], type=str)
    parser.add_argument('-doc_proc', default='cleaned', choices=['cleaned', 'orig'], type=str)

    parser.add_argument('-model_size', default='base', type=str,
                        help='BERT model type')
    parser.add_argument('-pretrained_model', default='spanbert', type=str,
                        help='Pretrained BERT model')
    parser.add_argument('-max_segment_len', default=512, type=int,
                        help='Max segment length of BERT segments.')
    parser.add_argument('-add_speaker_tags', default=False, action='store_true',
                        help='Whether to add speaker tags to document or not.')
    parser.add_argument('-use_local_attention', default=False, action="store_true",
                        help='Local Attention on top of BERT embeddings.')

    # Mention variables
    parser.add_argument('-max_span_width', default=4, type=int,
                        help='Max span width.')
    parser.add_argument('-top_span_ratio', default=0.1, type=float,
                        help='Fraction of top spans.')
    parser.add_argument('-ment_emb', default='attn', choices=['attn', 'endpoint'],
                        type=str, help='If true use an RNN on top of mention embeddings.')

    # Clustering variables
    parser.add_argument('-mem_type', default='unbounded', choices=['unbounded', 'unbounded_rnn'],
                        help="Memory type.")
    parser.add_argument('-mem_size', default=None, type=int,
                        help='Memory size used in the model')
    parser.add_argument('-rnn_size', default=50, type=int,
                        help='RNN size used in the unbounded rnn model')
    parser.add_argument('-mlp_size', default=1000, type=int,
                        help='MLP size used in the model')
    parser.add_argument('-no_use_doc_type', default=True, dest="use_doc_type", action="store_false",
                        help="If true, document type is used during clustering.")
    parser.add_argument('-no_use_ment_type', default=True, dest="use_ment_type", action="store_false",
                        help="If true, mentions are only merged with clusters of the same mention type.")
    parser.add_argument('-no_use_mem_context', default=True, dest="use_mem_context", action="store_false",
                        help="If true, use memory context.")
    parser.add_argument('-entity_rep', default='wt_avg', type=str,
                        choices=['learned_avg', 'wt_avg'], help='Entity representation.')
    parser.add_argument('-emb_size', default=20, type=int,
                        help='Embedding size of features.')

    # Training params
    parser.add_argument('-new_ent_wt', help='Weight of new entity term in coref loss',
                        default=1.0, type=float)
    parser.add_argument('-over_loss_wt', help='Weight of overwrite loss',
                        default=1.0, type=float)
    parser.add_argument('-sample_invalid', help='Sample invalids during training',
                        default=0.2, type=float)
    parser.add_argument('-num_train_docs', default=None, type=int,
                        help='Number of training docs.')
    parser.add_argument('-max_training_segments', default=3, type=int,
                        help='Max. number of BERT segments in a document.')
    parser.add_argument('-dropout_rate', default=0.5, type=float,
                        help='Dropout rate')
    parser.add_argument('-label_smoothing_wt', help='Weight of label smoothing',
                        default=0.0, type=float)

    parser.add_argument('-max_epochs',
                        help='Maximum number of epochs', default=25, type=int)
    parser.add_argument('-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-init_lr', help="Initial learning rate",
                        default=5e-4, type=float)
    parser.add_argument('-ft_lr', help="Fine-tuning learning rate",
                        default=5e-5, type=float)
    parser.add_argument('-no_finetune', default=True, dest="finetune",
                        action="store_false", help="Fine-tuning document encoder")
    parser.add_argument('-no_singletons', help="No singletons.",
                        default=False, action="store_true")
    parser.add_argument('-eval', help="Evaluate model",
                        default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID",
                        default=None, type=str)

    args = parser.parse_args()

    # Get model directory name
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model_size', 'pretrained_model', 'max_segment_len',
                'max_span_width', 'ment_emb', 'top_span_ratio',
                'mem_type', 'mem_size', 'mlp_size', # Memory params
                'use_ment_type', 'use_doc_type',  # Clustering params
                'max_epochs', 'dropout_rate', 'seed', 'init_lr', 'finetune', 'ft_lr', 'label_smoothing_wt',
                'num_train_docs', 'sample_invalid', 'max_training_segments', 'doc_proc',
                "new_ent_wt", # Training params
                ]
    if args.mem_type == 'unbounded_rnn':
        imp_opts.append('rnn_size')

    for key, val in vars(args).items():
        if key in imp_opts:
            opt_dict[key] = val

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = f"coref_{args.dataset}_" + str(hash_idx)

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

    # Get mention model name
    args.pretrained_mention_model = path.join(
        path.join(args.base_model_dir, get_mention_model_name(args)), "best_models/model.pth")
    print(args.pretrained_mention_model)

    # Log directory for Tensorflow Summary
    log_dir = path.join(model_dir, "logs")
    if not path.exists(log_dir):
        os.makedirs(log_dir)

    config_file = path.join(model_dir, 'config')
    with open(config_file, 'w') as f:
        for key, val in opt_dict.items():
            logging.info('%s: %s' % (key, val))
            f.write('%s: %s\n' % (key, val))

    Experiment(args, **vars(args))


if __name__ == "__main__":
    main()
