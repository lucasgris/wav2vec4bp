# Wav2vec 2.0 for Brazilian Portuguese :brazil:

> This repository aims at the development of audio technologies using Wav2vec 2.0, such as Automatic Speech Recognition (ASR), for the Brazilian Portuguese language. 

## Description 

This repository contains code and fine-tuned Wav2vec checkpoints for Brazilian Portuguese, including some useful scripts to download and preprocess transcribed data. 

Wav2vec 2.0 learns speech representations on unlabeled data as described in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477). For more information about Wav2vec, please access the [official repository](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec).

## Tasks

- [ ] Add [CORAA](https://github.com/nilc-nlp/CORAA) to the BP Dataset (BP Dataset Version 2);
- [ ] Release BP Dataset V2 fine tuned models;
- [ ] Finetune using the XLR-S 300M, XLR-S 1B and XLR-S 2B models.

## Checkpoints

### ASR checkpoints

We provide several Wav2vec fine-tuned models for ASR. For a more detailed description of how we finetuned these models, please check the paper [Brazilian Portuguese Speech Recognition Using Wav2vec 2.0](https://arxiv.org/abs/2107.11414).

Our last model is the bp\_400. It was finetuned using the 400h filtered version of the BP Dataset (see [Brazilian Portuguese (BP) Dataset Version 1](#Brazilian-Portuguese-(BP)-Dataset-Version-1) below). The results against each gathered dataset are shown below.

#### Checkpoints of BP Dataset V1

| Model name        | Pretrained model | Fairseq model | Dict     | Hugging Face link |
|-------------------|------------------|---------------|----------|-------------------|
| bp\_400           | XLSR-53          | [fairseq](https://drive.google.com/file/d/1AUqILVOLxcHzk7mu7YqtjBYkppni9Lgc/view?usp=sharing)   | [dict](https://drive.google.com/file/d/1J7hkjJjSoHNXUPO7A3O5XGCUnBTj7h9p/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp400-xlsr)  |
| bp\_400_xls-r-300M| XLS-R-300M       | [fairseq](https://drive.google.com/file/d/1o88iS12Lv214xWJ-l9PU6mk42R7Y0jWn/view?usp=sharing)   | [dict](https://drive.google.com/file/d/1J7hkjJjSoHNXUPO7A3O5XGCUnBTj7h9p/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp_400h_xlsr2_300M)  |

#### Checkpoints of non-filtered BP Dataset (early version of the BP dataset)

| Model name        | Pretrained model | Fairseq model | Dict     | Hugging Face link |
|-------------------|------------------|---------------|----------|-------------------|
| bp\_500           | XLSR-53          | [fairseq](https://drive.google.com/file/d/1J8aR1ltDLQFe-dVrGuyxoRm2uyJjCWgf/view?usp=sharing)   | [dict](https://drive.google.com/file/d/13fLaBgtImYxjnHKpSzyKycHnmSLlPZ2u/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp500-xlsr)  |
| bp\_500_10k       | VoxPopuli 10k BASE | [fairseq](https://drive.google.com/file/d/19kkENi8uvczmw9OLSdqnjvKqBE53cl_W/view?usp=sharing)   | [dict](https://drive.google.com/file/d/1V2Ke4bACFppWpcEwRdfAZdo4Nnpwjac9/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp500-base10k_voxpopuli)  |
| bp\_500_100k      | VoxPopuli 100k BASE | [fairseq](https://drive.google.com/file/d/10iESR5AQxuxF5F7w3wLbpc_9YMsYbY9H/view?usp=sharing)   | [dict](https://drive.google.com/file/d/1aWUCAK6f2fpQZCTTkAihsXRUvkgwyqCJ/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp500-base100k_voxpopuli)  |

#### Checkpoints of each gathered dataset

| Model name            | Pretrained model | Fairseq model | Dict     | Hugging Face link |
|-----------------------|------------------|---------------|----------|-------------------|
| bp\_cetuc\_100        | XLSR-53          | [fairseq](https://drive.google.com/file/d/13gR-LdJ5lP9lRst587apZ47hD_xPM2Xg/view?usp=sharing)   | [dict](https://drive.google.com/file/d/1Wv0UgkUF1s_sDJ0qjjptUorZKdWgnSfW/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp-cetuc100-xlsr)  |
| bp\_commonvoice\_100  | XLSR-53          | [fairseq](https://drive.google.com/file/d/1rX6lGz6rcVtlRjygzRNGuy2Q2jkVdMXS/view?usp=sharing)   | [dict](https://drive.google.com/file/d/1HemD4eOivY1SHvQ1D7YPlOciiwcInoU6/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp-commonvoice100-xlsr)  |
| bp\_commonvoice\_10   | XLSR-53          | [fairseq](https://drive.google.com/file/d/1N2cCvYeEm5xqRj1oylqBX1T7tYNMzk7R/view?usp=sharing)   | [dict](https://drive.google.com/file/d/11DnZ5EBxs79Z-_0XeKyCSHAGVt8gq4MG/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp-commonvoice10-xlsr)  |
| bp\_lapsbm\_1         | XLSR-53          | [fairseq](https://drive.google.com/file/d/1tjKhS-mdDoW8q_lsxWQiIY0xDbRXk9JX/view?usp=sharing)   | [dict](https://drive.google.com/file/d/1BRBqOiJIlXrvAb5_p5Pz_HQdHfX0TCyV/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp-lapsbm1-xlsr)  |
| bp\_mls\_100          | XLSR-53          | [fairseq](https://drive.google.com/file/d/1HY5LLere94SFhX9-YF0dZeuL-XGfothS/view?usp=sharing)   | [dict](https://drive.google.com/file/d/1dd1Jl6OEnsxjwOhbnM6oXWeD7PevWEoT/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp-mls100-xlsr)  |
| bp\_sid\_10           | XLSR-53          | [fairseq](https://drive.google.com/file/d/1gFCZSZI4iXYVUBIn058ITVU4V1yHyfIE/view?usp=sharing)   | [dict](https://drive.google.com/file/d/1qHgW9WXE-hTGiVQeLVKv0kKs-63j2Pui/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp-sid10-xlsr)  |
| bp\_tedx\_100         | XLSR-53          | [fairseq](https://drive.google.com/file/d/1m3nZTj87NzK-XXBrjJQiOsgwYQeVi_O0/view?usp=sharing)   | [dict](https://drive.google.com/file/d/1UvNwWIsd2Z11LPylh6C-IKSwNf73z7OF/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp-tedx100-xlsr)  |
| bp\_voxforge\_1       | XLSR-53          | [fairseq](https://drive.google.com/file/d/16T-NuY00xcudixfPv8k5A7fBaAQhIQ3D/view?usp=sharing)   | [dict](https://drive.google.com/file/d/1ZIPguXhdlGPqiV3NAF4Bc7dkh5NH3U38/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp-voxforge1-xlsr)  |

#### Other checkpoints 

We provide other Wav2vec checkpoints. These models were trained using all the available data at the time, including its dev and test subsets. Only Common Voice dev/test was selected to validate and test the model, respectively.

| Datasets used for training | Fairseq model | Dict | Hugging Face link |
|----------------------------|---------------|------|-------------------|
| CETUC + CV 6.1 (only train) + LaPS BM + MLS + VoxForge | [fairseq](https://drive.google.com/file/d/1wyRzlagentrI5PQvMaIjjm-DhLcIRgAx/view?usp=sharing) | [dict](https://drive.google.com/file/d/1BrFpmgZDc5xuQ7xvNgm4eiepi-7UJLyw/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/wav2vec2-large-xlsr-open-brazilian-portuguese) |
| CETUC + CV 6.1 (all validated) + LaPS BM + MLS + VoxForge |  | | [hugging face](https://huggingface.co/lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2) |


#### ASR Results

##### Summary (WER)

| Model                          | CETUC | CV    | LaPS  | MLS   | SID   | TEDx  | VF    | AVG   |
|--------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|
| __bp\_400__                    | 0.052 | 0.140 | 0.074 | 0.117 | 0.121 | 0.245 | 0.118 | 0.124 |
| __bp\_400_xls-r-300M__         | 0.048 | 0.123 | 0.068 | 0.111 | 0.084 | 0.207 | 0.095 | 0.105 |
| bp\_500                        | 0.052 | 0.137 | 0.032 | 0.118 | 0.095 | 0.236 | 0.082*| 0.112 |
| bp\_500-base10k_voxpopuli      | 0.120 | 0.249 | 0.039 | 0.227 | 0.169 | 0.349 | 0.116*| 0.181 |
| bp\_500-base100k_voxpopuli     | 0.074 | 0.174 | 0.032 | 0.182 | 0.181 | 0.349 | 0.111*| 0.157 |
| bp\_cetuc\_100**               | 0.446 | 0.856 | 0.089 | 0.967 | 1.172 | 0.929 | 0.902 | 0.765 | 
| bp\_commonvoice\_100           | 0.088 | 0.126 | 0.121 | 0.173 | 0.177 | 0.424 | 0.145 | 0.179 |
| bp\_commonvoice\_10            | 0.133 | 0.189 | 0.165 | 0.189 | 0.247 | 0.474 | 0.251 | 0.235 | 
| bp\_lapsbm\_1                  | 0.111 | 0.418 | 0.145 | 0.299 | 0.562 | 0.580 | 0.469 | 0.369 |
| bp\_mls\_100                   | 0.192 | 0.260 | 0.162 | 0.163 | 0.268 | 0.492 | 0.268 | 0.257 |
| bp\_sid\_10                    | 0.186 | 0.327 | 0.207 | 0.505 | 0.124 | 0.835 | 0.472 | 0.379 | 
| bp\_tedx\_100                  | 0.138 | 0.369 | 0.169 | 0.165 | 0.794 | 0.222 | 0.395 | 0.321 | 
| bp\_voxforge\_1                | 0.468 | 0.608 | 0.503 | 0.505 | 0.717 | 0.731 | 0.561 | 0.584 |

\* We found a problem with the dataset used in these experiments regarding the VoxForge subset. In this test set, some speakers were also present in the training set (which explains the lower WER). The final version of the dataset does not have such contamination.

\** We do not perform validation in the subset experiments. CETUC has a poor variety of transcriptions. It might be overfitted.

##### Transcription examples

| Text                                                                                                       | Transcription                                                                                                |
|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
|alguém sabe a que horas começa o jantar | alguém sabe a que horas **começo** jantar |
|lila covas ainda não sabe o que vai fazer no fundo|**lilacovas** ainda não sabe o que vai fazer no fundo|
|que tal um pouco desse bom spaghetti|**quetá** um pouco **deste** bom **ispaguete**|
|hong kong em cantonês significa porto perfumado|**rongkong** **en** **cantones** significa porto perfumado|
|vamos hackear esse problema|vamos **rackar** esse problema|
|apenas a poucos metros há uma estação de ônibus|apenas **ha** poucos metros **á** uma estação de ônibus|
|relâmpago e trovão sempre andam juntos|**relampagotrevão** sempre andam juntos|

## Datasets

Datasets provided:

- [CETUC](http://www02.smt.ufrj.br/~igor.quintanilha/alcaim.tar.gz): contains approximately 145 hours of Brazilian Portuguese speech distributed among 50 male and 50 female speakers, each pronouncing approximately 1,000 phonetically balanced sentences selected from the [CETEN-Folha](https://www.linguateca.pt/cetenfolha/) corpus.
- [Common Voice 7.0](https://commonvoice.mozilla.org/pt):  is a project proposed by Mozilla Foundation with the goal to create a wide-open dataset in different languages. In this project, volunteers donate and validate speech using the [oficial site](https://commonvoice.mozilla.org/pt). 
- [Lapsbm](https://github.com/falabrasil/gitlab-resources): "Falabrasil - UFPA" is a dataset used by the Fala Brasil group to benchmark ASR systems in Brazilian Portuguese. Contains 35 speakers (10 females), each one pronouncing 20 unique sentences, totaling 700 utterances in Brazilian Portuguese. The audios were recorded in 22.05 kHz without environment control.
- [Multilingual Librispeech (MLS)](https://arxiv.org/abs/2012.03411): a massive dataset available in many languages. The MLS is based on audiobook recordings in the public domain like [LibriVox](https://librivox.org/). The dataset contains a total of 6k hours of transcribed data in many languages. The set in Portuguese [used in this work](http://www.openslr.org/94/) (mostly Brazilian variant) has approximately 284 hours of speech, obtained from 55 audiobooks read by 62 speakers.
- [Multilingual TEDx](http://www.openslr.org/100): a collection of audio recordings from TEDx talks in 8 source languages. The Portuguese set (mostly Brazilian Portuguese variant) contains 164 hours of transcribed speech. 
- [Sidney](https://igormq.github.io/datasets/) (SID): contains 5,777 utterances recorded by 72 speakers (20 women) from 17 to 59 years old with fields such as place of birth, age, gender, education, and occupation;
- [VoxForge](http://www.voxforge.org/): is a project with the goal to build open datasets for acoustic models. The corpus contains approximately 100 speakers and 4,130 utterances of Brazilian Portuguese, with sample rates varying from 16kHz to 44.1kHz.

These datasets were combined to build a larger Brazilian Portuguese dataset (BP Dataset). All data was used for training except Common Voice dev/test sets, which were used for validation/test respectively. We also made test sets for all the gathered datasets.

| Dataset                        |  Train | Valid |  Test |
|--------------------------------|-------:|------:|------:|
| CETUC                          |  93.9h |    -- |  5.4h |
| Common Voice                   |  37.6h |  8.9h |  9.5h |
| LaPS BM                        |   0.8h |    -- |  0.1h |
| MLS                            | 161.0h |    -- |  3.7h |
| Multilingual TEDx (Portuguese) | 144.2h |    -- |  1.8h |
| SID                            |   5.0h |    -- |  1.0h |
| VoxForge                       |   2.8h |    -- |  0.1h |
| Total                          | 437.2h |  8.9h | 21.6h |

You can download the datasets individually using the scripts at scripts/ directory. The scripts will create the respective dev and test sets automatically.

```
python scripts/mls.py
```

If you want to join several datasets, execute the script join_datasets at scripts/: 

```
python scripts/join_datasets.py /path/to/dataset1/train /path/to/dataset2/train ... --output-dir data/my_dataset --output-name train
```

After joining datasets, you might have some degree of transcription contamination. To remove all transcriptions present in a specific subset (for example, test subset), you can use the filter_dataset script:

```
python scripts/filter_datasets.py /path/to/my_dataset/train /path/to/dataset1/test /path/to/dataset2/test -output-dir data/my_dataset --output-name my_filtered_train
```

Alternativelly, download the raw datasets using the links below:

- https://igormq.github.io/datasets/
- https://commonvoice.mozilla.org/
- http://www.openslr.org/94/
- http://www.openslr.org/100

### Brazilian Portuguese (BP) Dataset Version 1

The BP Dataset is an assembled dataset composed of many others in Brazilian Portuguese. We used the original test sets of each gathered dataset to make individual test sets. For the datasets without test sets, we created them by selecting 5% of unique male and female speakers. Additionally, we performed some filtering removing all transcriptions of the test sets from the final training set. We also ignored audio more than 30 seconds long from the dataset. 

If you run the provided scripts, you might generate a slightly different version of the BP dataset. If you want to use the same files used to train, validate and test our models, you can download the metadata [here](https://drive.google.com/drive/folders/1N_LHtbupQgpIoEobss4ZNw8azHcvn7Zq?usp=sharing).

#### Other versions

Our first attempt to build a larger dataset for BP produced a 500 hours dataset. However, we found some problems with the VoxForge subset. We also found some transcriptions of the test sets present in the training set. We made available the models trained with this version of the dataset (bp_500).

### Language models

Language models can improve the ASR output. To use with fairseq, you will need to install [flashlight python bindings](https://github.com/facebookresearch/flashlight/tree/master/bindings/python). You will also need a lexicon containing the possible words.

#### Ken LM models

You can download some Ken LM models [here](https://github.com/igormq/speech2text#language-models). It is compatible with the flashlight decoder.

#### Transformer LM (fairseq) models


| Model name                       | Fairseq model | Dict     |
|----------------------------------|---------------|----------|
| BP Transformer LM                | [fairseq model](https://drive.google.com/file/d/1zdq02Cp57NMoL9QgRL8MT-A6wi5KA5cJ/view?usp=sharing) | [dict](https://drive.google.com/file/d/1-95QYcfp6jXw1mEz6zTaqRZps-0OhgPk/view?usp=sharing) |
| Wikipedia Transformer LM         | [fairseq model](https://drive.google.com/file/d/1gm9Jova7ACDN6prn9JKbYvUEsbrpTWfL/view?usp=sharing) | [dict](https://drive.google.com/file/d/1A2eNXyfV6SPRk8R9ZguwfjYHwju2-x1W/view?usp=sharing) |
| Wikipedia Prunned Transformer LM | [fairseq model](https://drive.google.com/file/d/124jmnQbYI4JcQs4MKPe2EVCz-kRySUkq/view?usp=sharing) | [dict](https://drive.google.com/file/d/17n0tNftu9NgGRRvyWLInzkGZqaTBRU7f/view?usp=sharing) |

#### Lexicon

- [Wikipedia](https://drive.google.com/file/d/13xUf4OLW2l6FDe9oqN-4HW3P_XfkCEWk/view?usp=sharing)
- [BP Dataset](https://drive.google.com/file/d/1_luSxSClE9L5wvCGM_wLJtZJpD_9dWr8/view?usp=sharing)

#### 🤗 Hugging Face Transformers + Wav2Vec2_PyCTCDecode 

If you want to use [Wav2Vec2_PyCTCDecode](https://github.com/patrickvonplaten/Wav2Vec2_PyCTCDecode) with Transformers to decode the Hugging Face models, the Ken LM models provided above might not work. In this case, you should train your own following the instructions [here](https://github.com/patrickvonplaten/Wav2Vec2_PyCTCDecode), or use one of the two models trained with BP Dataset and Wikipedia below:

- [BP KenLM 4-gram](https://drive.google.com/file/d/1dLFldy7eguPtyJj5OAlI4Emnx0BpFywg/view?usp=sharing)
- [Wikipedia KenLM 4-gram](https://drive.google.com/file/d/1GJIKseP5ZkTbllQVgOL98R4yYAcIySFP/view?usp=sharing)

## ASR finetune

1. To finetune the model, first install [fairseq](https://github.com/pytorch/fairseq) and its dependencies.

```
cd fairseq
pip install -e .
```

2. Download a pre-trained model (See [pretrained models](#Pretrained-models))

2. Create or use a configuration file (see configs/ directory). 

3. Finetune the model executing fairseq-hydra-train

```
root=/path/to/wav2vec4bp
fairseq-hydra-train \
   task.data=$root/data/my_dataset \
   checkpoint.save_dir=$root/checkpoints/stt/my_model_name \
   model.w2v_path=$root/xlsr_53_56k.pt \
   common.tensorboard_logdir=$root/logs/stt/my_model_name \
   --config-dir $root/configs \
   --config-name my_configuration_file_name
```

### Pretrained models

To fine-tune Wav2vec, you will need to download a pre-trained model first.

- [XLSR-53 (large) (recommended)](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt)
- [VoxPopuli 10k (base)](https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_base_10k.pt)
- [VoxPopuli 10k (large)](https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_large_10k.pt)
- [VoxPopuli 100k (base)](https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_base_100k.pt)
- [VoxPopuli 100k (large)](https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_large_100k.pt)
- [XLR-S 300M](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt)
- [XLR-S 1B](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_960m_1000k.pt)
- [XLR-S 2B](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_2B_1000k.pt)

## 🤗 ASR finetune with HuggingFace

To easily finetune the model using hugging face, you can use the repository [Wav2vec-wrapper](https://github.com/Edresson/Wav2Vec-Wrapper).

## Language model training

To train a language model, one can use a Transformer LM or KenLM.

### Ken LM

First, install [KenLM](https://github.com/kpu/kenlm). 

```
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build
cd build
cmake ..
make -j 4
```

Then create a text file and run the following command:

```
./kenlm/build/bin/lmplz -o 5 <text.txt > path_to_lm.arpa
```

### Transformer LM

To train a [Transformer LM](https://github.com/pytorch/fairseq/tree/main/examples/language_model), first prepare and preprocess train, valid and test text files:

```
TEXT=path/to/dataset
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/train.tokens \
    --validpref $TEXT/valid.tokens \
    --testpref $TEXT/test.tokens \
    --destdir data/text/$dataset \
    --workers 20
```

Then train the model:

```
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
```

## Docker

We recommend using a docker container, such as [flml/flashlight](https://hub.docker.com/r/flml/flashlight/tags), to easily finetune and test your models.

