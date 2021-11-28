mkdir -p data/bp_dataset

if test -d "data/commonvoice_dataset/preprocessed"; then
    python scripts/common_voice.py --skip-download --skip-preprocessing
elif test -d "data/commonvoice_dataset/downloads"; then
    python scripts/common_voice.py --skip-download
else
    python scripts/common_voice.py --download-url $1 # INSERT COMMON VOICE LINK HERE #
fi

for data_set in cetuc lapsbm mls sid tedx voxforge; do
    if test -d "data/${data_set}_dataset/preprocessed"; then
        python scripts/common_voice.py --skip-download --skip-preprocessing
    elif test -d "data/${data_set}_dataset/downloads"; then
        python scripts/common_voice.py --skip-download
    else
        python scripts/${data_set}.py
    fi
done

for data_set in cetuc_dataset commonvoice_dataset mls_dataset tedx_dataset voxforge_dataset sid_dataset lapsbm_dataset; do
    python scripts/filter_dataset.py data/$data_set/preprocessed/train \
        data/$data_set/preprocessed/test \
        --output-dir data/$data_set/preprocessed \
        --output-name train_filtered -rs -l --use-wer
done

python scripts/join_datasets.py \
    data/cetuc_dataset/preprocessed/train_filtered \
    data/commonvoice_dataset/preprocessed/train_filtered \
    data/mls_dataset/preprocessed/train_filtered \
    data/tedx_dataset/preprocessed/train_filtered \
    data/voxforge_dataset/preprocessed/train_filtered \
    data/sid_dataset/preprocessed/train_filtered \
    data/lapsbm_dataset/preprocessed/train_filtered \
    --output-dir data/bp_dataset \
    --output-name train_raw \
    --skip-empty
    
python scripts/filter_dataset.py data/bp_dataset/train_raw \
    data/cetuc_dataset/preprocessed/test \
    data/commonvoice_dataset/preprocessed/test \
    data/mls_dataset/preprocessed/test \
    data/tedx_dataset/preprocessed/test \
    data/voxforge_dataset/preprocessed/test \
    data/sid_dataset/preprocessed/test \
    data/lapsbm_dataset/preprocessed/test \
    --output-dir data/bp_dataset \
    --output-name train_filtered -rs -l --use-wer

for ext in tsv ltr wrd; do
    cp data/commonvoice_dataset/preprocessed/test.$ext  data/bp_dataset/test_raw.$ext
    cp data/commonvoice_dataset/preprocessed/valid.$ext data/bp_dataset/valid_raw.$ext
    for data_set in cetuc_dataset commonvoice_dataset mls_dataset tedx_dataset voxforge_dataset sid_dataset lapsbm_dataset; do
        cp data/$data_set/preprocessed/test.$ext        data/bp_dataset/test_${data_set}.$ext
        # Necessary to train the models, even if validation is not used in these cases
        if ! test -f "data/$data_set/preprocessed/valid.$ext"; then
            cp data/$data_set/preprocessed/test.$ext    data/$data_set/preprocessed/valid.$ext
        fi
    done
done

for data_set in cetuc_dataset commonvoice_dataset mls_dataset tedx_dataset voxforge_dataset sid_dataset lapsbm_dataset; do
    cp data/$data_set/preprocessed/dict.ltr.txt data/$data_set/dict.ltr.txt 
    python scripts/limit_dataset_frames.py data/$data_set/preprocessed/train_filtered.tsv \
        --frame-limit 480000 \
        --output-dir data/$data_set \
        --output-name train_480
    for ext in tsv ltr wrd; do
        cp data/$data_set/train_480.$ext data/$data_set/train.$ext 
    done
    python scripts/limit_dataset_frames.py data/$data_set/preprocessed/valid.tsv \
        --frame-limit 480000 \
        --output-dir data/$data_set \
        --output-name valid_480
    for ext in tsv ltr wrd; do
        cp data/$data_set/valid_480.$ext data/$data_set/valid.$ext 
    done
    # If you want to limit test subsets
    # python scripts/limit_dataset_frames.py data/$data_set/preprocessed/test.tsv \
    #     --frame-limit 480000 \
    #     --output-dir data/$data_set \
    #     --output-name test_480
    # for ext in tsv ltr wrd; do
    #     cp data/$data_set/test_480.$ext data/$data_set/test.$ext 
    # done
done

python scripts/limit_dataset_frames.py data/bp_dataset/train_filtered.tsv \
    --frame-limit 480000 \
    --output-dir data/bp_dataset \
    --output-name train_480    
for ext in tsv ltr wrd; do
    cp data/bp_dataset/train_480.$ext data/bp_dataset/train.$ext 
done

python scripts/limit_dataset_frames.py data/bp_dataset/valid_raw.tsv \
    --frame-limit 480000 \
    --output-dir data/bp_dataset \
    --output-name valid_480
for ext in tsv ltr wrd; do
    cp data/bp_dataset/valid_480.$ext data/bp_dataset/valid.$ext 
done

# If you want to limit test subsets 
# python scripts/limit_dataset_frames.py data/bp_dataset/test_raw.tsv \
#     --frame-limit 480000 \
#     --output-dir data/bp_dataset \
#     --output-name test_480
# for ext in tsv ltr wrd; do
#     cp data/bp_dataset/test_480.$ext data/bp_dataset/test.$ext 
# done
