import os
import sys
import argparse
from collections import Counter
from random import shuffle
from shlex import quote


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', 
                        help='Manifest path. '
                        'Ex: /path/to/dataset/manifest.tsv. The dataset directory ' 
                        'must contain .wrd and .ltr files.')
    parser.add_argument('--frame-limit', help='Max frame allowed', required=True, type=int)
    parser.add_argument('--min-frames', help='Min frame allowed', type=int)
    parser.add_argument('--output-dir', help='output directory. Ex: /path/to/dataset. The path must exists ',
                        required=True)
    parser.add_argument('--output-name', help='output name. Ex: train', required=True)
    args = parser.parse_args()
    manifest_path = args.manifest
    words_path = manifest_path.split('.tsv')[0] + '.wrd'
    letters_path = manifest_path.split('.tsv')[0] + '.ltr'
    
    
    out_manifest = open(os.path.join(args.output_dir, args.output_name + '.tsv'), 'w')
    out_words = open(os.path.join(args.output_dir, args.output_name + '.wrd'), 'w')
    out_letters = open(os.path.join(args.output_dir, args.output_name + '.ltr'), 'w')
    
    with open(manifest_path) as train_manifest, open(words_path) as words, open(letters_path) as letters:            
        out_manifest.write(next(train_manifest))
        for manifest_line, word_line, letter_line in zip(train_manifest, words, letters):
            frames = int(manifest_line.split()[-1])
            if frames > args.frame_limit:
                print('Skip', manifest_line, '->', frames, '>', args.frame_limit)
                continue
            if args.min_frames and frames < args.min_frames:
                print('Skip', manifest_line, '->', frames, '<', args.min_frames)
                continue
            out_manifest.write(manifest_line)
            out_words.write(word_line)
            out_letters.write(letter_line)

