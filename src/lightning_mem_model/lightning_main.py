import argparse
from os import path
import hashlib
import logging
from collections import OrderedDict
from pytorch_lightning import Trainer

from lightning_mem_model.lightning_experiment import experiment

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
    # parser.add_argument('-all_truecase', default=True,
    #                     action="store_true", help='All documents processed through truecase.')
    parser.add_argument('-pretrained_bert_dir', default="/home/shtoshni/Research/litbank_coref/resources", type=str,
                        help='SpanBERT model location')
    parser.add_argument('-max_segment_len', default=512, type=int,
                        help='Max segment length of BERT segments.')

    # Mention variables
    parser.add_argument('-max_span_width', default=20, type=int,
                        help='Max span width.')
    parser.add_argument('-ment_emb', default='endpoint', choices=['attn', 'endpoint'],
                        type=str, help='Mention embedding type, attention adds an additional term corr. to whole span.')
    parser.add_argument('-focus_group', default='joint', choices=['joint', 'entity', 'event'], type=str,
                        help='Mentions in focus. If both, the cluster all mentions, otherwise cluster particular type'
                             ' of mentions.')
    parser.add_argument('-include_singletons', default=False,
                        action="store_true", help='Include singletons in experiment or not.')

    # Memory variables
    parser.add_argument('-mem_type', default='unbounded',
                        choices=['learned', 'lru', 'unbounded'],
                        help="Memory type.")
    parser.add_argument('-num_cells', default=20, type=int,
                        help="Number of memory cells.")
    parser.add_argument('-mem_size', default=None, type=int,
                        help='Memory size used in the model')
    parser.add_argument('-mlp_size', default=1024, type=int,
                        help='MLP size used in the model')
    parser.add_argument('-coref_mlp_depth', default=1, type=int,
                        help='Number of hidden layers in Coref MLP')
    parser.add_argument('-mlp_depth', default=1, type=int,
                        help='Number of hidden layers in other MLPs')
    parser.add_argument('-use_srl', default=False, action="store_true",
                        help="If true, coreference for event would also attend to entities, and vice-versa.")
    parser.add_argument('-emb_size', default=20, type=int,
                        help='Embedding size of features.')

    # Training params
    parser.add_argument('-new_ent_wt', help='Weight of new entity term in coref loss',
                        default=1.0, type=float)
    parser.add_argument('-over_loss_wt', help='Weight of overwrite loss',
                        default=1.0, type=float)
    parser.add_argument('-sample_singletons', help='Sample singletons during training',
                        default=1.0, type=float)
    parser.add_argument('-num_train_docs', default=None, type=int,
                        help='Number of training docs.')
    parser.add_argument('-dropout_rate', default=0.5, type=float,
                        help='Dropout rate')
    parser.add_argument('-max_epochs',
                        help='Maximum number of epochs', default=30, type=int)
    parser.add_argument('-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-init_lr', help="Initial learning rate",
                        default=5e-4, type=float)
    parser.add_argument('-ft_lr', help="Fine-tuning learning rate",
                        default=5e-5, type=float)
    parser.add_argument('-finetune', help="Fine-tuning document encoder",
                        default=False, action="store_true")
    parser.add_argument('-no_singletons', help="No singletons.",
                        default=True, action="store_true")
    parser.add_argument('-eval', help="Evaluate model",
                        default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID",
                        default=None, type=str)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Get model directory name
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model_size', 'max_segment_len', 'ment_emb', "doc_enc",  # Encoder params
                'mem_type', 'num_cells', 'mem_size', 'mlp_size', 'mlp_depth',
                'use_srl', 'include_singletons',  # SRL vector
                'coref_mlp_depth', 'emb_size',  # Memory params
                'max_epochs', 'dropout_rate', 'seed', 'init_lr', 'finetune', 'ft_lr',  # Training params
                'dataset', 'num_train_docs', 'over_loss_wt', "new_ent_wt", 'sample_singletons',  # Training params
                'focus_group',  # Mentions of particular focus
                ]
    for key, val in vars(args).items():
        if key in imp_opts:
            opt_dict[key] = val

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = "events_" + str(hash_idx)

    args.save_dir = args.weights_save_path if args.weights_save_path is not None else args.base_model_dir
    args.model_name = model_name

    # doc_enc = args.doc_enc + ('_truecase' if args.all_truecase else '')
    doc_enc = args.doc_enc + ('_singleton' if args.include_singletons else '')
    args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{doc_enc}')
    print(args.data_dir)

    experiment(args)


if __name__ == "__main__":
    main()