root=$1
dataset=$2
name=$3

TEXT=${root}/data/text/raw/$text_file
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/train.tokens \
    --validpref $TEXT/valid.tokens \
    --testpref $TEXT/test.tokens \
    --destdir data/text/$dataset \
    --workers 20

fairseq-train --task language_modeling \
  data/text/$dataset \
  --save-dir checkpoints/transformer_lms/$name \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 1024 --update-freq 32 \
  --fp16 \
  --max-update 50000
