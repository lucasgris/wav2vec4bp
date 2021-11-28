import glob as glob
import concurrent.futures
import shutil
import tarfile
import os

from tqdm import tqdm
import wget
import librosa
import soundfile

from common.manifest import write_manifest


DEFAULT_MIN_DURATION = 0.5
DEFAULT_MAX_DURATION = 50
DEFAULT_FRAME_RATE = 16e3
DEFAULT_NUM_JOBS = 16
DEFAULT_VOCAB = "abcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû-"
DEFAULT_DATA_DIR = 'data'


class Corpus:

    def __init__(self, args):

        self._num_jobs = args.jobs
        self.fs = int(args.frame_rate)
        self.min_frames = args.min_duration*self.fs
        self.max_frames = args.max_duration*self.fs

        self.download_urls = args.download_urls
        self.data_dir = args.data_dir

        self.name = args.name
        self.vocab = args.vocab + ' |'

        self.dataset_dir = os.path.join(self.data_dir, '{}_dataset'.format(self.name))
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        self.extracted_dir = os.path.join(self.dataset_dir, "extracted")
    
    def _pre_process_transcript(self, transcript):
        transcript = transcript.lower()
        if self.vocab is not None:
            for c in set(transcript):
                if c == ' ':
                    continue
                if c not in self.vocab:
                    transcript = transcript.replace(c, "")
        return transcript

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser."""
        parser.add_argument("--data-dir", help="data to store the dataset", default=DEFAULT_DATA_DIR)
        parser.add_argument('--min-duration', type=float, help='min audio duration in seconds', default=DEFAULT_MIN_DURATION)
        parser.add_argument('--max-duration', type=float, help='min audio duration in seconds', default=DEFAULT_MAX_DURATION)
        parser.add_argument('--frame-rate', type=int, help='target audio frame rate', default=DEFAULT_FRAME_RATE)
        parser.add_argument("--vocab", help="Vocabulary. Ex: \"abcdefghijklmnopqrstuvwxyz\"", default=DEFAULT_VOCAB)
        parser.add_argument('--jobs', type=int, help='number of parallel processes', default=DEFAULT_NUM_JOBS)
        parser.add_argument('--skip-download', action='store_true', help='skip download files')
        parser.add_argument('--skip-preprocessing', action='store_true', help='skip preprocessing files')
    
    def download(self, files_to_download=None, keep_download=False):

        downloads_dir = os.path.join(self.dataset_dir, "downloads")
        if not os.path.exists(downloads_dir):
            os.makedirs(downloads_dir)   

        for set_type, urls in self.download_urls.items():

            for url in urls:
                if url is not None:
                    if files_to_download:
                        for f in files_to_download:
                            if url.find(f) != -1:
                                break
                        else:
                            print("Skipping url: {}".format(url))
                            continue

                    fname = wget.detect_filename(url)
                    name = os.path.splitext(fname)[0]
                    target_fname = os.path.join(downloads_dir, set_type, fname)
                    if not os.path.exists(os.path.join(downloads_dir, set_type)):
                        os.makedirs(os.path.join(downloads_dir, set_type))
                    curr_extracted_dir = os.path.join(self.extracted_dir, set_type, fname)
                    if not os.path.exists(os.path.join(self.extracted_dir, set_type)):
                        os.makedirs(os.path.join(self.extracted_dir, set_type))

                    print('\nDownloading {}...'.format(fname))
                    if not os.path.exists(target_fname):
                        wget.download(url, target_fname)
                    else:
                        print("\nFile {} already downloaded".format(fname))

                    print("\nUnpacking {}...".format(fname))
                    if not os.path.exists(curr_extracted_dir):
                        tar = tarfile.open(target_fname)
                        tar.extractall(curr_extracted_dir)
                        tar.close()
                    else:
                        print("\nFile {} already unpacked".format(fname))

                    if not keep_download:
                        print("Removing downloaded file {}...".format(target_fname))
                        os.remove(target_fname)

                    assert os.path.exists(curr_extracted_dir), "Archive {} was not properly uncompressed.".format(fname)

                else:
                    print('No URL found. Skipping download.')
    

    def _preprocess_file(self, root, file):
        inpath = os.path.join(root, file)
        outpath = inpath.replace('extracted', 'preprocessed')
        splitpath = outpath.split(os.sep)
        os.makedirs(os.path.join(*splitpath[:-1]), exist_ok=True)
        
        audio_ext = None
        for ext in [".flac", ".mp3", ".wav"]:
            if file.endswith(ext):
                audio_ext = ext
                break
        if audio_ext is not None:
            audio, orig_sr = librosa.load(inpath, sr=None, mono=True)
            audio = librosa.resample(audio, orig_sr, self.fs)
            soundfile.write(outpath.replace(audio_ext, '.wav'), audio, self.fs)
        else:
            shutil.copyfile(inpath, outpath)

    def pre_process_audios(self, keep_extracted=False):
        assert os.path.exists(self.extracted_dir), 'No folder found in {}'.format(
            self.extracted_dir)
        for root, subfolders, files in os.walk(self.extracted_dir):
            print(f'Pre processing {root}')
            if self._num_jobs > 1:
                with concurrent.futures.ProcessPoolExecutor(self._num_jobs) as \
                        executor:
                    futures = [executor.submit(self._preprocess_file, root, file)
                                for file in files]
                    for f in tqdm(concurrent.futures.as_completed(futures)):
                        pass
            else:
                for file in tqdm(files):
                    self._preprocess_file(root, file)

        if not keep_extracted:
            print("Removing extracted files {}...".format(self.extracted_dir))
            shutil.rmtree(self.extracted_dir)

    def generate_manifest(self):
        for set_dir in self.dataset_dir:
            manifest_path = os.path.join(self.dataset_dir, set_dir + '.tsv')
            print("Writing manifest {}...".format(manifest_path))
            write_manifest(self.dataset_dir,
                           manifest_path, 
                           max_frames=self.max_frames, 
                           min_frames=self.min_frames)

    def generate_labels(self):
        raise NotImplementedError

    def generate_dict(self):
        raise NotImplementedError

