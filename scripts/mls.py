""" MLS dataset handler
"""
import os
import csv
import os.path as osp
from shutil import Error
import shutil
from string import ascii_lowercase
from collections import Counter

from tqdm import tqdm

from common.manifest import write_manifest
from common.corpus import Corpus

DEFAULT_SET_NAME = 'all'
MLS_DATASET_URL = "https://dl.fbaipublicfiles.com/mls/mls_portuguese.tar.gz"
SET_NAMES = [DEFAULT_SET_NAME]


class Mls(Corpus):

    DATASET_URLS = {DEFAULT_SET_NAME: [MLS_DATASET_URL]}

    def __init__(self, args):
        super().__init__(args)

    def pre_process_audios(self, keep_extracted=True):
        # Move subfolder's content to root tedx_dataset/extracted/all/
        if 'metainfo.txt' not in os.listdir(osp.join(self.extracted_dir, DEFAULT_SET_NAME)):
            tree = [*self.extracted_dir.split(os.sep)]
            while 'metainfo.txt' not in os.listdir(osp.join(*tree)):
                tree.append(os.listdir(osp.join(*tree))[0])
            extracted = os.listdir(osp.join(*tree))
            for f in extracted:
                print('Moving',
                    osp.join(*tree, f), '->',
                    osp.join(self.extracted_dir, DEFAULT_SET_NAME, f)
                )
                shutil.move(
                    osp.join(*tree, f),
                    osp.join(self.extracted_dir, DEFAULT_SET_NAME, f)
                )
        for folder in os.listdir(osp.join(self.extracted_dir)):
            if osp.isdir(
                osp.join(self.extracted_dir, DEFAULT_SET_NAME, folder)
            ) and (len(
                os.listdir(osp.join(self.extracted_dir, DEFAULT_SET_NAME, folder))
            ) == 0 or '.tar' in folder):
                shutil.rmtree(osp.join(self.extracted_dir, DEFAULT_SET_NAME, folder))
        
        super().pre_process_audios(keep_extracted=keep_extracted)

    def generate_dict(self):
        with open(
            osp.join(self.dataset_dir, 'preprocessed', DEFAULT_SET_NAME + '.ltr'), "r"
        ) as ltr:
            counter = Counter(letter for line in ltr for letter in line
                              if letter not in [' ', '"', '', '\n'])
        with open(
            osp.join(self.dataset_dir, 'preprocessed', 'dict.ltr.txt'), "w"
        ) as dict_f:
            for c in counter.most_common():
                print(f"{c[0]} {c[1]}", file=dict_f)

    def generate_manifest(self):
        print('Generating manifests')
        preproc_dir = osp.join(self.dataset_dir, 'preprocessed')
        if not osp.isdir(preproc_dir):
            raise Error(f'Preprocessed data dir {preproc_dir} not found. Did you download and preprocessed the dataset?')

        for set_name in SET_NAMES:
            manifest_path = osp.join(preproc_dir, set_name + '.tsv')
            if osp.isfile(manifest_path):
                print(f"Manifest {manifest_path} already exists")
            else:
                print("Writing manifest {}...".format(manifest_path))
                write_manifest(osp.join(preproc_dir, set_name), 
                            manifest_path, 
                            max_frames=self.max_frames, 
                            min_frames=self.min_frames)

        for set_name in ['train', 'dev', 'test']:
            manifest_path = osp.join(preproc_dir, set_name + '.tsv')
            if osp.isfile(manifest_path):
                print(f"Manifest {manifest_path} already exists")
            else:
                print("Writing manifest {}...".format(manifest_path))
                write_manifest(osp.join(preproc_dir, DEFAULT_SET_NAME, set_name), 
                               manifest_path, 
                               max_frames=self.max_frames, 
                               min_frames=self.min_frames)
        if osp.isfile(osp.join(preproc_dir, 'dev.tsv')):
            shutil.copy(osp.join(preproc_dir, 'dev.tsv'), osp.join(preproc_dir, 'valid.tsv'))

    def generate_labels(self):
        print('Generating labels')
        preproc_dir = osp.join(self.dataset_dir, 'preprocessed')

        sentences_by_fname = {}
        for subdir in ['train', 'dev', 'test']:
            with open(osp.join(preproc_dir, DEFAULT_SET_NAME, subdir, 'transcripts.txt')) as transcriptions:
                for line in transcriptions:
                    sentences_by_fname[line.split('\t')[0]] = line.split('\t')[1]

        for set_name in [*SET_NAMES, 'train', 'dev', 'test']:
            print(f'Generating labels for set {set_name}')
            manifest_path = osp.join(preproc_dir, set_name + '.tsv')
            with open(manifest_path) as manifest_file, open(
                osp.join(preproc_dir, set_name + ".ltr"), "w"
            ) as ltr_out, open(
                osp.join(preproc_dir, set_name + ".wrd"), "w"
            ) as wrd_out:
                root = next(manifest_file).strip()

                for i, line in enumerate(manifest_file):
                    fname = line.strip().split()[0].split('/')[-1].split('.wav')[0]
                    sentence = sentences_by_fname[fname]
                    sentence = self._pre_process_transcript(sentence)
                    print(sentence, file=wrd_out)
                    print(
                        " ".join(list(sentence.replace(" ", "|"))) + " |",
                        file=ltr_out,
                    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generates MLS files for wav2vec2 fine tuning (manifest, dict.txt, lrt and wrd files")
    Corpus.add_args(parser)
    parser.add_argument("--download-url", help="Download url.", default=MLS_DATASET_URL)
    args = parser.parse_args()

    args.download_urls = {DEFAULT_SET_NAME: [args.download_url]}
    args.name = 'mls'

    mls = Mls(args)
    if not args.skip_download:
        if not args.download_url:
            raise ValueError('Please provide an url for download')
        mls.download()
    if not args.skip_preprocessing:
        mls.pre_process_audios()
    mls.generate_manifest()
    mls.generate_labels()
    mls.generate_dict()