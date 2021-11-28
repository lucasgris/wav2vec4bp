import argparse
import glob
import os
import random

from tqdm import tqdm
import soundfile


def write_manifest(root, dest, ext='wav', path_must_contain=None, max_frames=None, min_frames=None):
    dir_path = os.path.realpath(root)
    search_path = os.path.join(dir_path, '**/*.' + ext)

    with open(dest, 'w') as tsv:
        print(dir_path, file=tsv)

        for fname in tqdm(glob.iglob(search_path, recursive=True)):
            file_path = os.path.realpath(fname)

            if path_must_contain and path_must_contain not in file_path:
                continue

            frames = soundfile.info(fname).frames
            if max_frames and frames > max_frames:
                continue
            if min_frames and frames < min_frames:
                continue
            print('{}\t{}'.format(os.path.relpath(file_path, dir_path), frames), file=tsv)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('root', metavar='DIR', help='root directory containing wav files to index')
    parser.add_argument('--dest', type=str, help='output', required=True)
    parser.add_argument('--ext', default='wav', type=str, metavar='EXT', help='extension to look for')
    parser.add_argument('--path-must-contain', default=None, type=str, metavar='FRAG',
                        help='if set, path must contain this substring for a file to be included in the manifest')

    args = parser.parse_args()
    write_manifest(args.root, args.ext, args.dest, args.path_must_contain)
