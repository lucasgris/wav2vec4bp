import os
import argparse
from collections import Counter
from random import shuffle
from shlex import quote

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='N', nargs='+',
                        help='a list of files to process. '
                        'Ex: /path/to/dataset1/train /path/to/dataset2/train. The dataset directory ' 
                        'must contains train.tsv, train.wrd and train.ltr files.')
    parser.add_argument('--output-dir', help='output directory. Ex: /path/to/dataset. The path must exists ',
                        required=True)
    parser.add_argument('--output-name', help='output name', default='train')
    parser.add_argument('--skip-empty', help='skips empty transcriptions', action='store_true')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, args.output_name), exist_ok=True)

    manifest = []
    words = []
    letters = []
    
    counters = []
    counter = Counter() 

    total_frames = 0

    for i, path in enumerate(args.files):
        with open(path + '.tsv') as train_manifest, open(path + '.wrd') as train_words, open(path + '.ltr') as train_letters:
            root = next(train_manifest).rstrip()
            dest = os.path.join(args.output_dir, args.output_name, str(i))
            cmd = f'ln -s {quote(root)} {quote(dest)}'
            print(cmd)
            os.system(cmd)
            for line_manifest, line_word, line_ltr in zip(train_manifest, train_words, train_letters):
                audio_path = os.path.join(args.output_dir, args.output_name, str(i), line_manifest.split()[0])
                if args.skip_empty and (" ".join(list(line_word)) == " "):
                    print("Skipping", audio_path, "(empty transcription)")
                    continue

                print(path, audio_path)
                assert os.path.isfile(audio_path)
                frames = int(line_manifest.split()[-1].rstrip())
                total_frames += frames
                
                audio_manifest_path = ''.join(audio_path.split(args.output_name+os.sep)[1:])
                manifest.append(f'{audio_manifest_path}\t{frames}\n')
                words.append(line_word)
                letters.append(line_ltr)
                counters.append(Counter(letter for letter in line_ltr
                                        if letter not in [' ', '"', '', '\n']))

        print(len(manifest), len(words), len(letters))
        assert len(manifest) == len(words) == len(letters), f"Error parsing dataset {path} files"
    
    data = list(zip(manifest, words, letters))
    shuffle(data)
    manifest, words, letters = zip(*data)
    for c in counters:
        counter += c
    
    with open(os.path.join(args.output_dir, f'{args.output_name}.tsv'), 'w') as manifest_output:
        manifest_output.write(os.path.join(os.path.abspath(args.output_dir), args.output_name) + '\n')
        for line in manifest:
            manifest_output.write(line)
    
    with open(os.path.join(args.output_dir, f'{args.output_name}.wrd'), 'w') as wrd_output:
        for line in words:
            wrd_output.write(line)
    
    with open(os.path.join(args.output_dir, f'{args.output_name}.ltr'), 'w') as ltr_output:
        for line in letters:
            ltr_output.write(line)
    
    with open(os.path.join(args.output_dir, f'dict.ltr.txt'), 'w') as dict_output: 
        for c in counter.most_common():
            print(f"{c[0]} {c[1]}", file=dict_output)

    print("Total frames:", total_frames)