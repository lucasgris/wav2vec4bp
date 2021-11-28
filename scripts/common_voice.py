""" Common Voice dataset handler
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
SET_NAMES = [DEFAULT_SET_NAME]
COMMONVOICE_DATASET_URL = None

class CommonVoice(Corpus):

    DATASET_URLS = {DEFAULT_SET_NAME: [COMMONVOICE_DATASET_URL]}
    TRAIN_TSV = 'validated.tsv'
    VALID_TSV = 'dev.tsv'
    TEST_TSV = 'test.tsv'

    def __init__(self, args):
        super().__init__(args)
    
    def _create_sets(self):
        all_speakers = []
        all_files = []
        all_sentences = []
        for fname in os.listdir(osp.join(self.dataset_dir, 'preprocessed', DEFAULT_SET_NAME)):
            if not fname.endswith('tsv'):
                continue
            with open(osp.join(self.dataset_dir, 'preprocessed', DEFAULT_SET_NAME, fname)) as tsv:
                tsv_reader = csv.reader(tsv, delimiter="\t")            
                next(tsv_reader) # skip header
                for line in tsv_reader:
                    all_speakers.append(line[0])
                    all_files.append(line[1].replace('.mp3', '.wav'))
                    all_sentences.append(line[2])
                    
        test_speakers = []
        test_files = []
        test_sentences = []
        with open(osp.join(self.dataset_dir, 'preprocessed', DEFAULT_SET_NAME, 'test.tsv')) as test_tsv:
            tsv_reader = csv.reader(test_tsv, delimiter="\t")            
            next(tsv_reader) # skip header
            for line in tsv_reader:
                test_speakers.append(line[0])
                test_files.append(line[1].replace('.mp3', '.wav'))
                test_sentences.append(line[2])

        valid_speakers = []        
        valid_files = []
        valid_sentences = []
        with open(osp.join(self.dataset_dir, 'preprocessed', DEFAULT_SET_NAME, 'dev.tsv')) as valid_tsv:
            tsv_reader = csv.reader(valid_tsv, delimiter="\t")            
            next(tsv_reader) # skip header
            for line in tsv_reader:
                speaker_id = line[0]
                sentence = line[2]
                if speaker_id not in test_speakers and sentence not in test_sentences:
                    valid_speakers.append(speaker_id)
                    valid_files.append(line[1].replace('.mp3', '.wav'))
                    valid_sentences.append(sentence)

        train_speakers = []
        train_files = []
        train_sentences = []
        with open(osp.join(self.dataset_dir, 'preprocessed', DEFAULT_SET_NAME, 'validated.tsv')) as train_tsv:
            tsv_reader = csv.reader(train_tsv, delimiter="\t")            
            next(tsv_reader) # skip header
            for line in tsv_reader:
                speaker_id = line[0]
                sentence = line[2]
                if (speaker_id not in test_speakers and speaker_id not in valid_speakers and
                    sentence not in valid_sentences and sentence not in test_sentences):
                    train_speakers.append(speaker_id)
                    train_files.append(line[1].replace('.mp3', '.wav'))
                    train_sentences.append(sentence)

        data_sets = {
            'all': dict(zip(all_files, all_sentences)),
            'train': dict(zip(train_files, train_sentences)),
            'valid': dict(zip(valid_files, valid_sentences)),
            'test': dict(zip(test_files, test_sentences))
        }
        
        print('Creating sets and copying files')
        preproc_dir = osp.join(self.dataset_dir, 'preprocessed')
        
        for data_set in data_sets:
            set_dir = osp.join(preproc_dir, data_set)
            if os.path.isdir(set_dir):
                print(f"{set_dir} exists. Skiping...")
            else:
                os.makedirs(set_dir)
                for i, file_name in enumerate(data_sets[data_set]):
                    file_path = osp.join(preproc_dir, DEFAULT_SET_NAME, 'clips', file_name)
                    assert os.path.isfile(file_path)
                    print(f"[{i} of {len(data_sets[data_set])}] Copying", os.path.abspath(file_path), "->", osp.join(os.path.abspath(set_dir), file_name))
                    shutil.copy(os.path.abspath(file_path), osp.join(os.path.abspath(set_dir), file_name))
                    assert os.path.isfile(osp.join(os.path.abspath(set_dir), file_name))

        return data_sets

    def pre_process_audios(self, keep_extracted=True):
        # Move subfolder's content to root commonvoice_dataset/extracted/all/
        if 'clips' not in os.listdir(osp.join(self.extracted_dir, DEFAULT_SET_NAME)):
            tree = [*self.extracted_dir.split(os.sep)]
            while 'clips' not in os.listdir(osp.join(*tree)):
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
                os.remove(osp.join(self.extracted_dir, DEFAULT_SET_NAME, folder))
        
        super().pre_process_audios(keep_extracted=keep_extracted)

    def generate_dict(self):
        with open(
            osp.join(self.dataset_dir, 'preprocessed', 'train.ltr'), "r"
        ) as ltr:
            counter = Counter(letter for line in ltr for letter in line
                              if letter not in [' ', '"', '', '\n'])
        with open(
            osp.join(self.dataset_dir, 'preprocessed', 'dict.ltr.txt'), "w"
        ) as dict_f:
            for c in counter.most_common():
                print(f"{c[0]} {c[1]}", file=dict_f)
            for c in self.vocab:
                if c not in counter:
                    if c != ' ':
                        print(f"{c} 0", file=dict_f)

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
                write_manifest(osp.join(preproc_dir, set_name, 'clips'), 
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
        data_sets = self._create_sets()

        for data_set in data_sets:
            print(f'Generating labels for set {data_set}')
            manifest_path = osp.join(preproc_dir, f'{data_set}.tsv')
            with open(manifest_path) as manifest_file, open(
                osp.join(preproc_dir, data_set + ".ltr"), "w"
            ) as ltr_out, open(
                osp.join(preproc_dir, data_set + ".wrd"), "w"
            ) as wrd_out:
                root = next(manifest_file).strip()

                for i, line in enumerate(manifest_file):
                    fp = line.strip().split()[0]
                    transc = self._pre_process_transcript(data_sets[data_set][os.path.basename(fp)])
                    print(transc, file=wrd_out)
                    print(
                        " ".join(list(transc.replace(" ", "|"))) + " |",
                        file=ltr_out,
                    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generates Common Voice files for wav2vec2 fine tuning (manifest, dict.txt, lrt and wrd files")
    Corpus.add_args(parser)
    parser.add_argument("--download-url", help="Download url. Get it at https://commonvoice.mozilla.org.")
    args = parser.parse_args()

    args.download_urls = {DEFAULT_SET_NAME: [args.download_url]}
    args.name = 'commonvoice'

    common_voice = CommonVoice(args)
    if not args.skip_download:
        if not args.download_url:
            raise ValueError('Please provide an url for download')
        common_voice.download()
    if not args.skip_preprocessing:
        common_voice.pre_process_audios()
    common_voice.generate_manifest()
    common_voice.generate_labels()
    common_voice.generate_dict()