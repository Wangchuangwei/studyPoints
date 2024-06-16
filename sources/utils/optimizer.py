def build_optimizer(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    use_bertadam = False
    #visual_backbone
    optimizer_grouped_parameters += [
        {
            "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) 
                                                                    and 'visual_backbone' in n)],
            "weight_decay": args.weight_decay,
            "lr": args.swin_learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) 
                                                                    and 'visual_backbone' in n)],
            "weight_decay": 0.0,
            "lr": args.swin_learning_rate,
        },
    ]
    #encoder 
    optimizer_grouped_parameters += [
        {
            "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) 
                                                                    and 'encoder' in n)],
            "weight_decay": args.weight_decay,
            "lr": args.bert_learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) 
                                                                    and 'encoder' in n)],
            "weight_decay": 0.0,
            "lr": args.bert_learning_rate,
        },
    ]
    #other
    optimizer_grouped_parameters += [
        {
            "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) 
                                                                    and 'encoder' not in n and 'visual_backbone' not in n)],
            "weight_decay": args.weight_decay,
            "lr": args.other_learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)) ]
        }
    ]
    return optimizer_grouped_parameters
                                              