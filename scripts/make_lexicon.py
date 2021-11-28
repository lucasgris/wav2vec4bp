import argparse
import os
import copy
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input text file")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--output-dict")
    parser.add_argument("--dict", help="Ignore words having chars not present in the provided dict.")
    parser.add_argument("--lower", action="store_true")
    parser.add_argument("--min-count", help="Min occurence to count a word in the lexicon", 
                        type=int, default=1)
    args = parser.parse_args()
    
    words_counts = dict()
    raw_words = set()

    valid_chars = None
    if args.dict is not None:
        valid_chars = []
        with open(args.dict) as vocab:
            for line in vocab:
                letter = line.split()[0]
                valid_chars.append(letter)

    with open(args.file, "r") as text_file:
        text = text_file.read()
        
        for word in text.split():
            word = word.strip()
            if word not in words_counts:
                words_counts[word] = 1
            else:
                words_counts[word] += 1

            raw_words.add(word)
            if valid_chars is not None:
                for c in word:
                    if c not in valid_chars:
                        if word in raw_words:
                            del words_counts[word]
                            raw_words.remove(word)
    
    if args.min_count > 1:
        words = set()
        skipped = 0
        print('Removing words with count less than', args.min_count)
        for word in raw_words:
            if words_counts[word] < args.min_count:
                skipped += 1
            else:
                words.add(word)
            print(f'Expected={len(raw_words)} Counted={len(words)} Skip={skipped}')
    else:
        words = raw_words

    with open(args.output_file, "w") as out:    
        words = sorted(list(words))
        print(f"Found {len(words)} words")
        for word in words:
            word_lex = " ".join(word) + " |"
            if args.lower:
                word_lex = word_lex.lower()
                word = word.lower()
            # print(word_lex)
            out.write(f"{word}\t{word_lex}\n")
    with open(args.output_file.split('.')[0] + '.count.tsv', 'w') as count_tsv: 
        for word in words_counts:
            count_tsv.write(f'{word}\t{words_counts[word]}\n')
    
    if args.output_dict is not None:
        with open(args.output_dict, 'w') as out:
            for word in dict(sorted(words_counts.items(), 
                                    key=lambda words_counts: words_counts[1], 
                                    reverse=True)):
                out.write(f'{word} {words_counts[word]}\n')

if __name__ == "__main__":
    main()

