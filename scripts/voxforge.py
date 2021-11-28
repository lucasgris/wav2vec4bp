""" VOXFORGE dataset handler
"""
import os
import csv
import random
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
VOXFORGE_DATASET_URL = "http://www02.smt.ufrj.br/~igor.quintanilha/voxforge-ptbr.tar.gz"


class VoxForge(Corpus):

    DATASET_URLS = {DEFAULT_SET_NAME: [VOXFORGE_DATASET_URL]}

    def __init__(self, args):
        super().__init__(args)

    def _create_sets(self, test_speakers_perc=0.1):
        male_speaker_folders = {}
        female_speaker_folders = {}

        train_folders = set()
        all_folders = set()

        def add_folder_to_skp(d, spk, folder):
            if spk not in d:
                d[spk] = [folder]
            else:
                d[spk].append(folder) 

        for folder in os.listdir(osp.join(self.extracted_dir, DEFAULT_SET_NAME)):
            print(folder)
            readme_fp = osp.join(self.extracted_dir, DEFAULT_SET_NAME, folder, 'README')
            if not osp.isfile(readme_fp):
                readme_fp = osp.join(self.extracted_dir, DEFAULT_SET_NAME, folder, 'etc', 'README')
            if not osp.isfile(readme_fp):
                print(f'Could not read file of {folder} properly')
                continue

            with open(readme_fp) as readme:
                readme_lines = readme.readlines()
                gender = readme_lines[4].split('Gender:')[-1].strip().lower()
                username = readme_lines[0].split('User Name:')[-1].strip().lower()

                if gender == 'masculino' and username != 'anonymous':
                    add_folder_to_skp(male_speaker_folders, username, folder)
                elif gender == 'feminino' and username != 'anonymous':
                    add_folder_to_skp(female_speaker_folders, username, folder)
                elif gender == 'desconhecido':
                    print(f'Unidentified gender of {folder}')
                    train_folders.add(folder)
                elif username == 'anonymous':
                    print(f'[Anonymous user] Adding {folder} to the train set')
                    train_folders.add(folder)
                else:
                    print(f'Could not read gender of file {readme_fp} properly')
                    train_folders.add(folder)
                all_folders.add(folder)

        min_of_each = min(len(male_speaker_folders), len(female_speaker_folders))
        test_speakers_count = int((min_of_each*2)*test_speakers_perc)
        test_speakers_count += test_speakers_count % 2
        test_folders = set()
        
        for gender_speaker_folders in [male_speaker_folders, female_speaker_folders]:
            random.seed(1)
            for _ in range(test_speakers_count//2):
                random_skp = random.choice(list(gender_speaker_folders.keys()))
                random_skp_folders = gender_speaker_folders[random_skp]
                test_folders |= set(random_skp_folders)

        train_folders |= set(all_folders-test_folders)
        
        data_sets = {
            'all': all_folders,
            'test': test_folders,
            'train': train_folders
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
        if 'aglehg-20130430-mfm' not in os.listdir(osp.join(self.extracted_dir, DEFAULT_SET_NAME)):
            tree = [*self.extracted_dir.split(os.sep)]
            while 'aglehg-20130430-mfm' not in os.listdir(osp.join(*tree)):
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

                for line in tqdm(manifest_file):
                    fp = line.strip().split()[0]
                    if osp.isdir(osp.join(root, fp.split(os.sep)[0], "etc")):
                        prompts_file = osp.join(root, fp.split(os.sep)[0], "etc", "prompts-original")
                    else:
                        prompts_file = osp.join(root, fp.split(os.sep)[0], "prompts-original")
                    with open(prompts_file) as prompts:
                        found = False
                        for line in prompts:
                            splits = line.strip().split()
                            fp_id = fp.split(os.sep)[-1].split(".")[0]
                            prompt_id = splits[0]
                            if fp_id == prompt_id:   
                                transc = " ".join(splits[1:])
                                transc = self._pre_process_transcript(transc)
                                print(transc, file=wrd_out)
                                print(
                                    " ".join(list(transc.replace(" ", "|"))) + " |",
                                    file=ltr_out,
                                )
                                found = True
                        if not found:
                            print("", file=wrd_out)
                            print(" |", file=ltr_out)
                            print(f"[WARNING] Missing transcription for {fp}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generates Vox Forge files for wav2vec2 fine tuning (manifest, dict.txt, lrt and wrd files")
    Corpus.add_args(parser)
    parser.add_argument("--download-url", help="Download url.", default=VOXFORGE_DATASET_URL)
    args = parser.parse_args()

    args.download_urls = {DEFAULT_SET_NAME: [args.download_url]}
    args.name = 'voxforge'

    vox_forge = VoxForge(args)
    if not args.skip_download:
        if not args.download_url:
            raise ValueError('Please provide an url for download')
        vox_forge.download()
    if not args.skip_preprocessing:
        vox_forge.pre_process_audios()
    vox_forge.generate_manifest()
    vox_forge.generate_labels()
    vox_forge.generate_dict()