import argparse
import os
import copy
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Data dir")
    parser.add_argument("--root", help="Root dir. If not set, root=data_dir")
    parser.add_argument("--subset", help="Subset", default='test')
    args = parser.parse_args()
    
    os.chdir(args.data_dir)
    if not args.root:
        args.root = args.data_dir
    datasets = os.listdir()

    for dataset in datasets:
        if not os.path.isfile(os.path.join(args.dataset, args.subset)):
            continue
        with open(os.path.join(dataset, f'{args.subset}.csv'), 'w') as csv, open(
            os.path.join(dataset, f'{args.subset}.wrd')
        ) as sentences, open(
            os.path.join(dataset, f'{args.subset}.tsv')
        ) as manifest:
            next(manifest)
            data_root = os.path.join(args.root, dataset, 'preprocessed', 'test')
            print(data_root)
            print('path,sentence', file=csv)
            for lm, ls in zip(manifest, sentences):
                sentence = ls.strip()
                if "".join(list(sentence)) == "":
                    continue
                path = os.path.join(data_root, lm.split()[0].strip())
                print(f'{path},{sentence}', file=csv)

if __name__ == "__main__":
    main()