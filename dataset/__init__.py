
def make_dataset(args):
    if args.dataset == 'l3das23':
        from dataset.custom_dataset import load_dataset
        return load_dataset(args)
    elif args.dataset == 'mcse':
        from dataset.mcse_dataset import make_mcse_dataset
        return make_mcse_dataset(args)