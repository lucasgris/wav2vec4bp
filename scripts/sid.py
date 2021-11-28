""" SID dataset handler
"""
import os
import csv
import os.path as osp
from shutil import Error
import shutil
import random
from string import ascii_lowercase
from collections import Counter

from tqdm import tqdm

from common.manifest import write_manifest
from common.corpus import Corpus

DEFAULT_SET_NAME = 'all'
SID_DATASET_URL = "http://www02.smt.ufrj.br/~igor.quintanilha/sid.tar.gz"
SET_NAMES = [DEFAULT_SET_NAME]

class Sid(Corpus):

    DATASET_URLS = {DEFAULT_SET_NAME: [SID_DATASET_URL]}

    def __init__(self, args):
        super().__init__(args)

    def _create_sets(self, test_speakers_perc=0.1):
        speakers = list(filter(lambda d: osp.isdir(osp.join(self.extracted_dir, d)),
                               os.listdir(self.extracted_dir)))
        print(speakers)
        male_speakers = list(filter(lambda s: s.startswith('M'), speakers))
        female_speakers = list(filter(lambda s: s.startswith('F'), speakers))

        test_speakers_count = int(len(speakers)*test_speakers_perc)
        test_speakers_count += test_speakers_count % 2

        test_speakers = set()
        for gender_speakers in [male_speakers, female_speakers]:
            random.seed(1)
            test_speakers |= set(random.choices(gender_speakers, k=test_speakers_count//2))

        train_speakers = set(speakers) - test_speakers
        
        data_sets = {
            'all': set(speakers),
            'test': test_speakers,
            'train': train_speakers
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
                for i, speaker_folder in enumerate(data_sets[data_set]):
                    speaker_folder_path = osp.join(preproc_dir, DEFAULT_SET_NAME, speaker_folder)
                    assert osp.isdir(speaker_folder_path), f"Directory {speaker_folder_path} does not exists"
                    print(f"[{i} of {len(data_sets[data_set])}] Copying", 
                            osp.abspath(speaker_folder_path), "->", 
                            osp.join(osp.abspath(set_dir), osp.basename(speaker_folder_path)))
                    shutil.copytree(osp.abspath(speaker_folder_path), osp.join(osp.abspath(set_dir), 
                                    osp.basename(speaker_folder_path)))
                    assert osp.isdir(osp.join(osp.abspath(set_dir), osp.basename(speaker_folder_path)))
        return data_sets

    def pre_process_audios(self, keep_extracted=True):
        # Move subfolder's content to root tedx_dataset/extracted/all/
        if 'F0001' not in os.listdir(osp.join(self.extracted_dir, DEFAULT_SET_NAME)):
            tree = [*self.extracted_dir.split(os.sep)]
            while 'F0001' not in os.listdir(osp.join(*tree)):
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
        
        self.extracted_dir = osp.join(self.extracted_dir, DEFAULT_SET_NAME)

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
            with open(manifest_path) as manifest_file, open(
                osp.join(preproc_dir, set_name + ".ltr"), "w"
            ) as ltr_out, open(
                osp.join(preproc_dir, set_name + ".wrd"), "w"
            ) as wrd_out:
                root = next(manifest_file).strip()
                for i, line in enumerate(manifest_file):
                    fp = line.strip().split()[0]
                    spk = fp.split(os.sep)[0]
                    try:
                        sentence_id = int(fp.split(os.sep)[1].split(spk)[1].split('.wav')[0])
                        found = False
                        with open(osp.join(root, fp.split(os.sep)[0], "prompts.txt")) as prompts:
                            for line in prompts:
                                splits = line.strip().split('=')
                                if sentence_id == int(splits[0]):
                                    sentence = splits[1]
                                    sentence = self._pre_process_transcript(sentence)
                                    print(sentence, file=wrd_out)
                                    print(
                                        " ".join(list(sentence.replace(" ", "|"))) + " |",
                                        file=ltr_out,
                                    )
                                    found = True
                                    break
                            if not found:
                                print(" ", file=wrd_out)
                                print(" |", file=ltr_out)
                                print(f"[WARNING] Missing transcription for {fp}")
                    except:
                        print(f'An error occured parsing {fp}')
                        print(" ", file=wrd_out)
                        print(" |", file=ltr_out)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generates Sid files for wav2vec2 fine tuning (manifest, dict.txt, lrt and wrd files")
    Corpus.add_args(parser)
    parser.add_argument("--download-url", help="Download url.", default=SID_DATASET_URL)
    args = parser.parse_args()

    args.download_urls = {DEFAULT_SET_NAME: [args.download_url]}
    args.name = 'sid'

    sid = Sid(args)
    if not args.skip_download:
        if not args.download_url:
            raise ValueError('Please provide an url for download')
        sid.download()
    if not args.skip_preprocessing:
        sid.pre_process_audios()
    sid.generate_manifest()
    sid.generate_labels()
    sid.generate_dict()