import argparse
from collections import OrderedDict
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input text file")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--min-count", help="Min occurence to count", 
                        type=int, default=1)
    parser.add_argument("--max-size", help="Prune variety of words", type=int)
    args = parser.parse_args()
    
    words_counts = dict()
    unique = set()

    num_lines = sum(1 for _ in open(args.file))

    print('Counting words')
    with open(args.file, "r") as text_file:
        for line in tqdm(text_file, total=num_lines):
        
            for word in line.split():
                word = word.strip()
                if word not in words_counts:
                    words_counts[word] = 1
                else:
                    words_counts[word] += 1
                unique.add(word)

    print('Unique words:', len(unique))
    if args.max_size:
        print('Prunning words')
        words_counts = dict(
            sorted(words_counts.items(), key=lambda x:x[1], reverse=True)[:args.max_size]
        )

    prunned = 0
    print('Writting file')
    with open(args.file) as text_file, open(args.output_file, 'w') as out:
        for line in tqdm(text_file, total=num_lines):
            ok = True
            line = line.rstrip()
            for word in line.split():
                if word not in words_counts or words_counts[word] < args.min_count:
                    ok = False
                    break
            if not ok:
                prunned += 1
                continue
            print(line, file=out)

    print(f'Prunned %{round((prunned/num_lines)*100, 2)}')

if __name__ == "__main__":
    main()
