root=$1
kenlm_models_dir=${root}/kenlm_models
transformer_lm_models_dir=${root}/checkpoints/transformer_lms

DEFAULT='--task audio_finetuning --output-csv infer_summary.csv --output-preds infer_preds.csv --nbest 1 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 5000000 --post-process letter --sample-rate 16000 --num-workers 0'
LEXICON=./lexicon-wiki.lst

for kenlm in pt-BR.word.3-gram.binary pt-BR.word.5-gram.binary; do
    if ! test -f $kenlm_models_dir; then
        echo wget http://www02.smt.ufrj.br/~igor.quintanilha/$kenlm kenlm_models/
    fi
done


function run_infer_viterbi() {
    dataset=$1
    exp=$2
    subset=$3
    ckp_name=checkpoint_best.pt
    if ! test -f $root/checkpoints/stt/$exp/$ckp_name; then
        ckp_name=checkpoint_last.pt
    fi
    python3 scripts/infer.py $dataset $DEFAULT \
            --beam 100 \
            --w2l-decoder viterbi \
            --path $root/checkpoints/stt/$exp/$ckp_name \
            --gen-subset $subset \
            --lm-weight 1 
}


function run_infer_kenlm() {
    dataset=$1
    exp=$2
    subset=$3
    lm=$4
    ckp_name=checkpoint_best.pt
    if ! test -f $root/checkpoints/stt/$exp/$ckp_name; then
        ckp_name=checkpoint_last.pt
    fi
    python3 scripts/infer.py $dataset $DEFAULT \
            --beam 100 \
            --w2l-decoder kenlm \
            --lm-model $lm \
            --lexicon $LEXICON \
            --path $root/checkpoints/stt/$exp/$ckp_name \
            --gen-subset $subset \
            --lm-weight 1 
}


function run_infer_fairseqlm() {
    dataset=$1
    exp=$2
    subset=$3
    lm=$4
    ckp_name=checkpoint_best.pt
    if ! test -f $root/checkpoints/stt/$exp/$ckp_name; then
        ckp_name=checkpoint_last.pt
    fi
    python3 scripts/infer.py $dataset $DEFAULT \
            --beam 100 \
            --w2l-decoder fairseqlm \
            --lm-model $lm \
            --lexicon $LEXICON \
            --path $root/checkpoints/stt/$exp/$ckp_name \
            --gen-subset $subset    \
            --lm-weight 1 
}


rm errors.txt

test_subsets="test_voxforge_dataset test_cetuc_dataset test_commonvoice_dataset test_lapsbm_dataset test_mls_dataset test_sid_dataset test_tedx_dataset "
#                           DATASET                                     EXP                 SUBSET          LM
for subset in $test_subsets; do
    run_infer_viterbi       data/bp_dataset                             bp_400h             $subset           
done
for subset in $test_subsets; do
    for kenlm in pt-BR.word.3-gram.binary pt-BR.word.5-gram.binary; do
        run_infer_kenlm     data/bropen_dataset                         bp_400h             $subset         kenlm_models/$kenlm            
    done;
done
for subset in $test_subsets; do
   run_infer_fairseqlm      data/bropen_dataset                         bp_400h             $subset         $transformer_lm_models_dir/wiki-pt-br/checkpoint_best.pt        
   run_infer_fairseqlm      data/bropen_dataset                         bropen_100k         $subset         $transformer_lm_models_dir/wiki-pt-br/checkpoint_best.pt        
   run_infer_fairseqlm      data/bropen_dataset                         bropen_100k         $subset         $transformer_lm_models_dir/wiki-pt-br/checkpoint_best.pt          
done

run_infer_viterbi           data/cetuc_dataset/preprocessed             cetuc_100h          test
run_infer_viterbi           data/commonvoice_dataset/preprocessed       commonvoice_10h     test
run_infer_viterbi           data/commonvoice_dataset/preprocessed       commonvoice_100h    test
run_infer_viterbi           data/lapsbm_dataset/preprocessed            lapsbm_1h           test
run_infer_viterbi           data/mls_dataset/preprocessed               mls_100h            test
run_infer_viterbi           data/sid_dataset/preprocessed               sid_10h             test
run_infer_viterbi           data/tedx_dataset/preprocessed              tedx_100h           test
run_infer_viterbi           data/voxforge_dataset/preprocessed          voxforge_1h         test
for kenlm in pt-BR.word.3-gram.binary pt-BR.word.5-gram.binary; do
    run_infer_kenlm         data/cetuc_dataset/preprocessed             cetuc_100h          test            $kenlm_models_dir/$kenlm
    run_infer_kenlm         data/commonvoice_dataset/preprocessed       commonvoice_10h     test            $kenlm_models_dir/$kenlm
    run_infer_kenlm         data/commonvoice_dataset/preprocessed       commonvoice_100h    test            $kenlm_models_dir/$kenlm
    run_infer_kenlm         data/lapsbm_dataset/preprocessed            lapsbm_1h           test            $kenlm_models_dir/$kenlm
    run_infer_kenlm         data/mls_dataset/preprocessed               mls_100h            test            $kenlm_models_dir/$kenlm
    run_infer_kenlm         data/sid_dataset/preprocessed               sid_10h             test            $kenlm_models_dir/$kenlm
    run_infer_kenlm         data/tedx_dataset/preprocessed              tedx_100h           test            $kenlm_models_dir/$kenlm
    run_infer_kenlm         data/voxforge_dataset/preprocessed          voxforge_1h         test            $kenlm_models_dir/$kenlm
done;
run_infer_fairseqlm         data/bropen_dataset                         bp_400h             test            $transformer_lm_models_dir/wiki-pt-br/checkpoint_best.pt    
run_infer_fairseqlm         data/lapsbm_dataset                         lapsbm_1h           test            $transformer_lm_models_dir/wiki-pt-br/checkpoint_best.pt
run_infer_fairseqlm         data/commonvoice_dataset                    commonvoice_10h     test            $transformer_lm_models_dir/wiki-pt-br/checkpoint_best.pt
run_infer_fairseqlm         data/commonvoice_dataset                    commonvoice_100h    test            $transformer_lm_models_dir/wiki-pt-br/checkpoint_best.pt
run_infer_fairseqlm         data/mls_dataset                            mls_100h            test            $transformer_lm_models_dir/wiki-pt-br/checkpoint_best.pt
run_infer_fairseqlm         data/sid_dataset                            sid_10h             test            $transformer_lm_models_dir/wiki-pt-br/checkpoint_best.pt
run_infer_fairseqlm         data/tedx_dataset                           tedx_100h           test            $transformer_lm_models_dir/wiki-pt-br/checkpoint_best.pt
run_infer_fairseqlm         data/voxforge_dataset                       voxforge_1h         test            $transformer_lm_models_dir/wiki-pt-br/checkpoint_best.pt
run_infer_fairseqlm         data/cetuc_dataset                          cetuc_100h          test            $transformer_lm_models_dir/wiki-pt-br/checkpoint_best.pt
