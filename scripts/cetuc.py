""" CETUC dataset handler
"""
import os
import os.path as osp
import csv
import re
import random
from shutil import Error
import shutil
from string import ascii_lowercase
from collections import Counter

from tqdm import tqdm

from common.manifest import write_manifest
from common.corpus import Corpus

DEFAULT_SET_NAME = 'all'
CETUC_DATASET_URL = "http://www02.smt.ufrj.br/~igor.quintanilha/alcaim.tar.gz"
SET_NAMES = [DEFAULT_SET_NAME]


class Cetuc(Corpus):

    PATTERN_FILE = re.compile('[F|M][0-9]*-[0-9]*\.(wav|txt)')
    DATASET_URLS = {DEFAULT_SET_NAME: [CETUC_DATASET_URL]}
    TOTAL_SENTENCES = list(range(1000))

    def __init__(self, args):
        super().__init__(args)

    def _create_sets(self, test_speakers_count=20, test_sentences_count=200):
        speakers = list(filter(lambda d: osp.isdir(osp.join(self.extracted_dir, DEFAULT_SET_NAME, d)),
                               os.listdir(osp.join(self.extracted_dir, DEFAULT_SET_NAME))))
                               
                               
        random.seed(1)
        test_speakers = set(random.choices(speakers, k=test_speakers_count))
        train_speakers = set(speakers)-test_speakers
      
        random.seed(1)
        test_sentences = set(random.choices(Cetuc.TOTAL_SENTENCES, k=test_sentences_count))
        train_sentences = set(Cetuc.TOTAL_SENTENCES)-test_sentences
        
        data_sets = {
            'all': {'speakers': set(speakers), 'sentences': set(Cetuc.TOTAL_SENTENCES)},
            'train': {'speakers': train_speakers, 'sentences': train_sentences},
            'test': {'speakers': test_speakers, 'sentences': test_sentences}
        }

        print('Creating sets and copying files')
        preproc_dir = osp.join(self.dataset_dir, 'preprocessed')

        for data_set in data_sets:
            print(f'Processing {data_set}')
            set_dir = osp.join(preproc_dir, data_set)
            if osp.isdir(set_dir):
                print(f"{set_dir} exists. Skip copying files...")
            else:
                os.makedirs(set_dir)
                for speaker_folder in sorted(list(data_sets[data_set]['speakers'])):
                    print(f'Copying files of folder {speaker_folder}')
                    speaker_folder_path = osp.join(preproc_dir, DEFAULT_SET_NAME, speaker_folder)
                    new_speaker_folder_path = osp.join(set_dir, speaker_folder)
                    os.makedirs(new_speaker_folder_path, exist_ok=True)
                    
                    for file in tqdm(os.listdir(speaker_folder_path)):
                        if not Cetuc.PATTERN_FILE.match(file):
                            continue
                        sentence_id = file.split('-')[1].split('.')[0]
                        if sentence_id.isdigit() and int(sentence_id) in data_sets[data_set]['sentences']:
                            new_file_path = osp.join(new_speaker_folder_path, file)
                            shutil.copy(osp.join(speaker_folder_path, file),
                                        new_file_path)
                            assert osp.isfile(new_file_path)
        return data_sets

    def pre_process_audios(self, keep_extracted=True):
        # Move subfolder's content to root tedx_dataset/extracted/all/
        if 'README.md' not in os.listdir(osp.join(self.extracted_dir, DEFAULT_SET_NAME)):
            tree = [*self.extracted_dir.split(os.sep)]
            while 'README.md' not in os.listdir(osp.join(*tree)):
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
        for folder in os.listdir(osp.join(self.extracted_dir, DEFAULT_SET_NAME)):
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
        
        data_sets = self._create_sets()

        for data_set in data_sets:
            print(f'Processing {data_set}')
            set_dir = osp.join(preproc_dir, data_set)
            
            manifest_path = osp.join(preproc_dir, f'{data_set}.tsv')
            if osp.isfile(manifest_path):
                print(f"Manifest {manifest_path} already exists")
            else:
                print("Writing manifest {}...".format(manifest_path))
                write_manifest(set_dir, manifest_path, max_frames=self.max_frames, min_frames=self.min_frames)

    def generate_labels(self):
        print('Generating labels')
        preproc_dir = osp.join(self.dataset_dir, 'preprocessed')

        for set_name in [*SET_NAMES, 'train', 'test']:
            print(f'Generating labels for set {set_name}')
            manifest_path = osp.join(preproc_dir, set_name + '.tsv')
            transcriptions = {}
            with open(manifest_path) as manifest_file, open(
                osp.join(preproc_dir, set_name + ".ltr"), "w"
            ) as ltr_out, open(
                osp.join(preproc_dir, set_name + ".wrd"), "w"
            ) as wrd_out:
                root = next(manifest_file).strip()
                for dir in tqdm(os.listdir(root)):
                    if not osp.isdir(osp.join(root, dir)): 
                        continue
                    for file in os.listdir(osp.join(root, dir)):
                        if file.endswith('.txt'):
                            transcriptions[file.split('.txt')[0]] = open(
                                osp.join(root, dir, file)
                            ).read().rstrip()

                for line in manifest_file:
                    fn = line.strip().split()[0].split(os.sep)[-1].split('.wav')[0]
                    sentence = transcriptions[fn]
                    sentence = self._pre_process_transcript(sentence)
                    print(sentence, file=wrd_out)
                    print(
                        " ".join(list(sentence.replace(" ", "|"))) + " |",
                        file=ltr_out,
                    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generates CETUC files for wav2vec2 fine tuning (manifest, dict.txt, lrt and wrd files")
    Corpus.add_args(parser)
    parser.add_argument("--download-url", help="Download url.", default=CETUC_DATASET_URL)
    args = parser.parse_args()

    args.download_urls = {DEFAULT_SET_NAME: [args.download_url]}
    args.name = 'cetuc'

    cetuc = Cetuc(args)
    if not args.skip_download:
        if not args.download_url:
            raise ValueError('Please provide an url for download')
        cetuc.download()
    if not args.skip_preprocessing:
        cetuc.pre_process_audios()
    cetuc.generate_manifest()
    cetuc.generate_labels()
    cetuc.generate_dict()