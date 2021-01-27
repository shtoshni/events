
def get_mention_model_name(args):
    model_name_suffix = ""
    model_name_suffix += f'{args.dataset}' + "_"
    model_name_suffix += 'srl_'  # f'{args.doc_proc}' + "_"
    model_name_suffix += 'mlp_' + f'{args.mlp_size}' + '_'
    # model_name_suffix += 'drop_' + f'{args.dropout_rate}' + '_'
    model_name_suffix += 'model_' + f'{args.model_size}' + '_'
    model_name_suffix += 'emb_' + f'{args.ment_emb}' + '_'
    model_name_suffix += 'type_' + args.pretrained_model + '_'
    # model_name_suffix += 'segments_' + f'{args.max_training_segments}' + '_'
    model_name_suffix += 'segments_3_'  # + f'{args.max_training_segments}' + '_'
    model_name_suffix += 'width_' + f'{args.max_span_width}' + '_'

    # if not args.multitask:
    #     model_name_suffix += 'no_multitask_'

    if args.dropout_rate != 0.5:
        model_name_suffix += f'drop_{args.dropout_rate}'

    # if args.use_local_attention:
    #     model_name_suffix += 'local_'

    # if args.use_srl:
    #     model_name_suffix += f'srl_{args.srl_loss_wt}_'

    if not args.add_speaker_tags:
        model_name_suffix += 'no_speaker_'

    if args.finetune:
        model_name_suffix += 'ft_'  # + f'{args.ft_lr}'

    if model_name_suffix[-1] == '_':
        model_name_suffix = model_name_suffix[:-1]

    # if args.seed != 0:
    #     model_name_suffix += f"_seed_{args.seed}"
    model_name = "ment_" + model_name_suffix
    return model_name
