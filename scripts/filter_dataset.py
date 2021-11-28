import os
import argparse
from tqdm import tqdm
import jiwer


def compute(ref, other, lower=False, metric='cer'):
    if lower:
        ref = ref.lower().strip()
    if lower:
        other = other.lower().strip()
    if metric == 'cer':
        return jiwer.wer([char for char in ref], [char for char in other])
    return jiwer.wer(ref, other)


def check_equal(ref, other, lower=False):
    if lower:
        ref = ref.lower().strip()
    if lower:
        other = other.lower().strip()
    return ref == other


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subset', help= 'Subset path to filter. Ex: /path/to/dataset/train. The dataset directory ' 
                        'must contains train.tsv, train.wrd and train.ltr files.')
    parser.add_argument('others', metavar='N', nargs='+',
                        help='a list of subsets to use for the filter process. '
                        'Ex: /path/to/dataset1/test /path/to/dataset2/test. The dataset directory ' 
                        'must contains test.tsv, test.wrd and test.ltr files.')
    parser.add_argument('--output-dir', help='output directory. Ex: /path/to/dataset. The path must exists ',
                        required=True)
    parser.add_argument('--output-name', help='output name', default='train')
    parser.add_argument('--threshold', '-t', help='CER/WER threshold', type=float, default=0.1)
    parser.add_argument('--lower', '-l', help='make all lowercase to compute the CER', action='store_true')
    parser.add_argument('--remove-extra-spaces', '-rs', help='Remove extra spaces', action='store_true')
    parser.add_argument('--use-wer', '-w', help='faster method', action='store_true')
    parser.add_argument('--check-equal', '-ce', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    def count_lines(file):
        with open(file) as f:
            return len(f.readlines())

    manifest = []
    words = []
    letters = []

    skipped, total = 0,  0
    
    with open(args.subset + '.tsv') as train_manifest, open(args.subset + '.wrd') as train_words, open(args.subset + '.ltr') as train_letters:        
        root = next(train_manifest).rstrip()
        total_lines = count_lines(args.subset + '.wrd')
        
        for line_manifest, line_word, line_ltr in tqdm(zip(train_manifest.readlines(), train_words.readlines(), train_letters.readlines()), total=total_lines):
            
            found_similar_sentence = False

            line_word = line_word.strip()
            if args.remove_extra_spaces:
                line_word = " ".join(line_word.split())
            if line_word == '' or line_word == ' ':
                continue
            total += 1
        
            for other_subset in args.others:
                with open(other_subset + '.wrd') as other_words:
                    for other_line_word in other_words:
                        other_line_word = other_line_word.strip()
                        if args.remove_extra_spaces:
                            other_line_word = " ".join(other_line_word.split())
                        if other_line_word == '' or other_line_word == ' ':
                            continue
                        
                        if args.verbose:
                            print(
                                f"[E={round(compute(line_word, other_line_word, lower=args.lower, metric='wer' if args.use_wer else 'cer'), 4)}]"
                                f'|{line_word[:80]}|',
                                f'|{other_line_word[:80]}|',
                                f'({other_subset})'
                            )

                        if args.check_equal and check_equal(line_word, other_line_word, lower=args.lower):
                            print(
                                f'|{line_word}|',
                                f'({line_manifest.split()[0]})',
                                'equal to',
                                f'|{other_line_word}|',
                                f'({other_subset})'
                            )
                            skipped += 1
                            found_similar_sentence = True
                        elif compute(line_word, other_line_word, lower=args.lower, metric='wer' if args.use_wer else 'cer') < args.threshold:
                            print(
                                f'|{line_word}|'    ,
                                f'({line_manifest.split()[0]})',
                                'similar to',
                                f'|{other_line_word}|',
                                f'({other_subset})',
                                f"-> SKIP [E={compute(line_word, other_line_word, lower=args.lower, metric='wer' if args.use_wer else 'cer')}] < {args.threshold}"
                            )
                            skipped += 1
                            found_similar_sentence = True

                        if found_similar_sentence:
                            break
                
                if found_similar_sentence:
                    break

            if not found_similar_sentence:
                manifest.append(line_manifest)
                words.append(line_word+'\n')
                letters.append(line_ltr)

    print(os.path.join(args.output_dir, f'{args.output_name}'))
    with open(os.path.join(args.output_dir, f'{args.output_name}.tsv'), 'w') as manifest_output:
        manifest_output.write(root + '\n')
        for line in manifest:
            manifest_output.write(line)
    
    with open(os.path.join(args.output_dir, f'{args.output_name}.wrd'), 'w') as wrd_output:
        for line in words:
            wrd_output.write(line)
    
    with open(os.path.join(args.output_dir, f'{args.output_name}.ltr'), 'w') as ltr_output:
        for line in letters:
            ltr_output.write(line)

    print("Skipped:", skipped, 'of total of', total)