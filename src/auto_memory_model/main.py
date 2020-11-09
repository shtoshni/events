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
        '-dataset', default='red', choices=['red'], type=str)

    parser.add_argument('-model_size', default='base', type=str,
                        help='BERT model type')
    parser.add_argument('-doc_enc', default='independent', type=str,
                        choices=['independent', 'overlap'], help='BERT model type')
    parser.add_argument('-proc_strategy', default='duplicate', type=str,
                        choices=['default', 'duplicate'],
                        help='Document processing strategy. In duplicate we add [DUPLICATE] tags to document.')
    parser.add_argument('-pretrained_bert_dir', default="/home/shtoshni/Research/litbank_coref/resources", type=str,
                        help='SpanBERT model location')
    parser.add_argument('-max_segment_len', default=512, type=int,
                        help='Max segment length of BERT segments.')

    # Mention variables
    parser.add_argument('-max_span_width', default=10, type=int,
                        help='Max span width.')
    parser.add_argument('-ment_emb', default='attn', choices=['attn', 'endpoint'],
                        type=str, help='If true use an RNN on top of mention embeddings.')
    parser.add_argument('-focus_group', default='joint', choices=['joint', 'entity', 'event'], type=str,
                        help='Mentions in focus. If both, the cluster all mentions, otherwise cluster particular type'
                             ' of mentions.')
    parser.add_argument('-ment_ordering', default='ment_type', type=str,
                        choices=['ment_type', 'document'],
                        help='Order in which detected mentions are clustered. If ment_type, entity mentions are'
                        'clustered before event mentions, otherwise mentions are ordered by their location in doc.')

    # Clustering variables
    parser.add_argument('-mem_type', default='unbounded',
                        choices=['learned', 'lru', 'unbounded'],
                        help="Memory type.")
    parser.add_argument('-num_cells', default=20, type=int,
                        help="Number of memory cells.")
    parser.add_argument('-mem_size', default=None, type=int,
                        help='Memory size used in the model')
    parser.add_argument('-mlp_size', default=1000, type=int,
                        help='MLP size used in the model')
    parser.add_argument('-use_srl', default=None, choices=['joint', 'event'], type=str,
                        help="If true, coreference for event would also attend to entities, and vice-versa.")
    parser.add_argument('-use_ment_type', default=False, action="store_true",
                        help="If true, mentions are only merged with clusters of the same mention type.")
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
                        default=1.0, type=float)
    parser.add_argument('-num_train_docs', default=None, type=int,
                        help='Number of training docs.')
    parser.add_argument('-max_training_segments', default=None, type=int,
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
    parser.add_argument('-finetune', help="Fine-tuning document encoder",
                        default=False, action="store_true")
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
    imp_opts = ['model_size', 'max_segment_len', 'doc_enc', 'proc_strategy',  # Document + Encoder params
                'ment_emb', 'ment_ordering', 'focus_group',  # Mention params
                'mem_type', 'num_cells', 'mem_size', 'mlp_size',  # Memory params
                'use_srl',  'use_ment_type',  # Clustering params
                'max_epochs', 'dropout_rate', 'seed', 'init_lr', 'finetune', 'ft_lr', 'label_smoothing_wt',
                'dataset', 'num_train_docs', 'sample_invalid', 'max_training_segments',
                'over_loss_wt',  "new_ent_wt",  # Training params
                ]
    for key, val in vars(args).items():
        if key in imp_opts:
            opt_dict[key] = val

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = "events_" + str(hash_idx)

    model_dir = path.join(args.base_model_dir, model_name)
    args.model_dir = model_dir
    best_model_dir = path.join(model_dir, 'best_models')
    args.best_model_dir = best_model_dir
    if not path.exists(model_dir):
        os.makedirs(model_dir)
    if not path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    # doc_enc = args.doc_enc + ('_truecase' if args.all_truecase else '')
    doc_enc = args.doc_enc + (f'_{args.proc_strategy}' if (args.proc_strategy != 'default') else '')
    args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{doc_enc}')
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
