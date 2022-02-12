# Brazilian Portuguese Speech Recognition Using Wav2vec 2.0 :brazil:

Paper: https://arxiv.org/abs/2107.11414

## Description 

This repository contains code and fine-tuned Wav2vec ASR checkpoints for Brazilian Portuguese, including some useful scripts to download and preprocess transcribed data. 

Wav2vec 2.0 learns speech representations on unlabeled data as described in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477). For more information about Wav2vec, please access the [official repository](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec).

## Checkpoints

### ASR checkpoints

We provide several Wav2vec fine-tuned models for ASR. For a more detailed description of how we finetuned these models, please check the paper [Brazilian Portuguese Speech Recognition Using Wav2vec 2.0](https://arxiv.org/abs/2107.11414).

Our last model is the bp\_400. It was finetuned using the 400h filtered version of the BP Dataset (see [Brazilian Portuguese (BP) Dataset Version 1](#Brazilian-Portuguese-(BP)-Dataset-Version-1) below). The results against each gathered dataset are shown below.

#### Checkpoints of BP Dataset V1

| Model name        | Pretrained model | Fairseq model | Dict     | Hugging Face link |
|-------------------|------------------|---------------|----------|-------------------|
| bp\_400           | XLSR-53          | [fairseq](https://drive.google.com/file/d/1AUqILVOLxcHzk7mu7YqtjBYkppni9Lgc/view?usp=sharing)   | [dict](https://drive.google.com/file/d/1J7hkjJjSoHNXUPO7A3O5XGCUnBTj7h9p/view?usp=sharing) | [hugging face](https://huggingface.co/lgris/bp400-xlsr)  |

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

#### ASR Results

##### Summary (WER)

| Model                          | CETUC | CV    | LaPS  | MLS   | SID   | TEDx  | VF    | AVG   |
|--------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|
| __bp\_400__                    | 0.052 | 0.140 | 0.074 | 0.117 | 0.121 | 0.245 | 0.118 | 0.124 |
| bp\_cetuc\_100**               | 0.446 | 0.856 | 0.089 | 0.967 | 1.172 | 0.929 | 0.902 | 0.765 | 
| bp\_commonvoice\_100           | 0.088 | 0.126 | 0.121 | 0.173 | 0.177 | 0.424 | 0.145 | 0.179 |
| bp\_commonvoice\_10            | 0.133 | 0.189 | 0.165 | 0.189 | 0.247 | 0.474 | 0.251 | 0.235 | 
| bp\_lapsbm\_1                  | 0.111 | 0.418 | 0.145 | 0.299 | 0.562 | 0.580 | 0.469 | 0.369 |
| bp\_mls\_100                   | 0.192 | 0.260 | 0.162 | 0.163 | 0.268 | 0.492 | 0.268 | 0.257 |
| bp\_sid\_10                    | 0.186 | 0.327 | 0.207 | 0.505 | 0.124 | 0.835 | 0.472 | 0.379 | 
| bp\_tedx\_100                  | 0.138 | 0.369 | 0.169 | 0.165 | 0.794 | 0.222 | 0.395 | 0.321 | 
| bp\_voxforge\_1                | 0.468 | 0.608 | 0.503 | 0.505 | 0.717 | 0.731 | 0.561 | 0.584 |

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
