root=$1

if ! test -f "$root/xlsr_53_56k.pt"; then
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt $root/xlsr_53_56k.pt
fi
if ! test -f "$root/wav2vec2_base_10k.pt"; then
    wget https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_base_10k.pt $root/wav2vec2_base_10k.pt
fi
if ! test -f "$root/wav2vec2_base_100k.pt"; then
    wget https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_base_100k.pt $root/wav2vec2_base_100k.pt
fi

fairseq-hydra-train \
   task.data=$root/data/bp_dataset \
   checkpoint.save_dir=$root/checkpoints/stt/bp_400h \
   model.w2v_path=$root/xlsr_53_56k.pt \
   common.tensorboard_logdir=$root/logs/stt/bp_400h \
   --config-dir $root/configs \
   --config-name finetune_400h 

fairseq-hydra-train \
    dataset.disable_validation=true \
    task.data=$root/data/cetuc_dataset \
    checkpoint.save_dir=$root/checkpoints/stt/cetuc_100h \
    common.tensorboard_logdir=$root/logs/stt/cetuc_100h \
    model.w2v_path=$root/xlsr_53_56k.pt \
    --config-dir $root/configs \
    --config-name finetune_100h 

fairseq-hydra-train \
    dataset.disable_validation=true \
    task.data=$root/data/commonvoice_dataset \
    checkpoint.save_dir=$root/checkpoints/stt/commonvoice_10h \
    common.tensorboard_logdir=$root/logs/stt/commonvoice_10h \
    model.w2v_path=$root/xlsr_53_56k.pt \
    --config-dir $root/configs \
    --config-name finetune_10h 

fairseq-hydra-train \
    dataset.disable_validation=true \
    task.data=$root/data/commonvoice_dataset \
    checkpoint.save_dir=$root/checkpoints/stt/commonvoice_100h \
    common.tensorboard_logdir=$root/logs/stt/commonvoice_100h \
    model.w2v_path=$root/xlsr_53_56k.pt \
    --config-dir $root/configs \
    --config-name finetune_100h 

fairseq-hydra-train \
    dataset.disable_validation=true \
    task.data=$root/data/lapsbm_dataset \
    checkpoint.save_dir=$root/checkpoints/stt/lapsbm_1h \
    common.tensorboard_logdir=$root/logs/stt/lapsbm_1h \
    model.w2v_path=$root/xlsr_53_56k.pt \
    --config-dir $root/configs \
    --config-name finetune_1h 

fairseq-hydra-train \
    dataset.disable_validation=true \
    task.data=$root/data/mls_dataset \
    checkpoint.save_dir=$root/checkpoints/stt/mls_100h \
    common.tensorboard_logdir=$root/logs/stt/mls_100h \
    model.w2v_path=$root/xlsr_53_56k.pt \
    --config-dir $root/configs \
    --config-name finetune_100h 

fairseq-hydra-train \
    dataset.disable_validation=true \
    task.data=$root/data/sid_dataset \
    checkpoint.save_dir=$root/checkpoints/stt/sid_10h \
    common.tensorboard_logdir=$root/logs/stt/sid_10h \
    model.w2v_path=$root/xlsr_53_56k.pt \
    --config-dir $root/configs \
    --config-name finetune_10h 

fairseq-hydra-train \
    dataset.disable_validation=true \
    task.data=$root/data/tedx_dataset \
    checkpoint.save_dir=$root/checkpoints/stt/tedx_100h \
    common.tensorboard_logdir=$root/logs/stt/tedx_100h \
    model.w2v_path=$root/xlsr_53_56k.pt \
    --config-dir $root/configs \
    --config-name finetune_100h 

fairseq-hydra-train \
    dataset.disable_validation=true \
    task.data=$root/data/voxforge_dataset \
    checkpoint.save_dir=$root/checkpoints/stt/voxforge_1h \
    common.tensorboard_logdir=$root/logs/stt/voxforge_1h \
    model.w2v_path=$root/xlsr_53_56k.pt \
    --config-dir $root/configs \
    --config-name finetune_1h 
