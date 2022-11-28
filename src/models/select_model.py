
def select_model(args, train_loader, device):

    # set up model
    if args.model_type == 'densenet':
        from models.densenet_pool import make_model
    # elif args.model_type == 'resnet':
    #     from models.resnet import make_model
    elif args.model_type == 'mlp':
        from models.mlp import make_model
    elif args.model_type == 'seq2seq':
        from models.seq2seq import make_model
    else:
        raise NotImplementedError

    temp1, temp2, _ = next(iter(train_loader))
    n_in = temp1.shape[-1]
    n_out = temp2.shape[-1]

    if args.model_type == 'mlp':
        net = make_model(in_dim=temp1.shape[1] * temp1.shape[-1],
                         out_dim=n_out,
                         drop_rate=args.drop_ratio,
                         n_blocks=args.mlp_n_blocks,
                         hidden_channels=args.mlp_n_hidden).to(device)
    elif args.model_type == 'densenet':
        pool = 3 if temp1.shape[1] > 12 else 2 # for one setup with 27 timepoints in input
        net = make_model(in_dim=n_in,
                         out_dim=n_out,
                         drop_rate=args.drop_ratio,
                         bottleneck=args.dense_bottleneck,
                         reduction=args.dense_reduce,
                         depth=args.dense_n_layers,
                         growth_rate=args.dense_growth_rate,
                         pool=pool).to(device)
    elif args.model_type == 'seq2seq':
        net = make_model(in_dim=n_in,
                         out_dim=n_out,
                         drop_ratio=args.drop_ratio,
                         n_enc_layers=args.seq_n_enc_layers,
                         n_dec_layers=args.seq_n_dec_layers,
                         enc_bidirectional=args.seq_bidirectional).to(device)
    else:
        raise NotImplementedError

    return net