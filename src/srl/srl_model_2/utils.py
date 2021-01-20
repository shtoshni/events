
def get_srl_model_name(args):
    model_name_suffix = ""
    model_name_suffix += f'{args.dataset}' + "_"
    model_name_suffix += 'mlp_' + f'{args.mlp_size}' + '_'
    # model_name_suffix += 'drop_' + f'{args.dropout_rate}' + '_'
    model_name_suffix += 'model_' + f'{args.model_size}' + '_'
    model_name_suffix += 'emb_' + f'{args.ment_emb}' + '_'
    model_name_suffix += 'type_' + ('spanbert' if args.pretrained_bert_dir else 'bert') + '_'

    # if not args.multitask:
    #     model_name_suffix += 'no_multitask_'

    if args.dropout_rate != 0.5:
        model_name_suffix += f'drop_{args.dropout_rate}'

    if args.finetune:
        model_name_suffix += 'ft_'  # + f'{args.ft_lr}'

    if args.num_train_docs is not None:
        model_name_suffix += f'docs_{args.num_train_docs}_'  # + f'{args.ft_lr}'

    if model_name_suffix[-1] == '_':
        model_name_suffix = model_name_suffix[:-1]

    # model_name_suffix += f"_seed_{args.seed}"
    model_name = "srl_" + model_name_suffix
    return model_name
