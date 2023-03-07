# 🇺🇦 Ukrainian ELECTRA model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4267880.svg)](https://doi.org/10.5281/zenodo.4267880)

In this repository we monitor all experiments for our trained ELECTRA model
for Ukrainian.

Made with 🤗, 🥨 and ❤️ from Munich.

# Changelog

* 06.03.2023: Model is now located under `lang-uk` organization on the Hugging Face
              [model hub page](https://huggingface.co/lang-uk). New evaluation results using Flair library are added.
* 11.11.2020: Add DOI/Zenodo information, including citation section.
* 10.11.2020: Initial version and public release of Ukrainian ELECTRA model.

# Training

The source data for the Ukrainian ELECTRA model consists of two corpora:

* Recent Wikipedia dump
* Deduplicated Ukrainian part from the [OSCAR](https://oscar-corpus.com/) corpus

The resulting corpus has a size of 31GB (uncompressed). For the Wikipedia dump we use newlines as
document boundaries (ELECTRA pre-training does support it).

We then apply two preprocessing steps:

Sentence splitting. We use [tokenize-uk](https://github.com/lang-uk/tokenize-uk) in order to perform
sentence splitting.

Filtering. We discard all sentences that are shorter than 5 tokens.

The final training corpus has a size of 30GB and consits of exactly 2,402,761,324 tokens.

The Ukrainian ELECTRA model was trained for 1M steps in total using a batch
size of 128. We pretty much following the ELECTRA training procedure as used for
[BERTurk](https://github.com/stefan-it/turkish-bert/tree/master/electra).

# Experiments

We use latest Flair version (release 0.12) for performing experiments on NER and PoS Tagging
downstream tasks. Older experiments can be found under [this](https://github.com/stefan-it/ukrainian-electra/tree/1.0.0)
tag.

The script `flair-fine-tuner.py` is used to perform a basic hyper-parameter search. The scripts
expects a json-based configuration file. Examples can be found in the `./configs/ner` and `./configs/pos`
folders of this repository.

## PoS Tagging

### Ukrainian-IU

Description:

> UD Ukrainian comprises 122K tokens in 7000 sentences of fiction, news, opinion articles, Wikipedia,
> legal documents, letters, posts, and comments — from the last 15 years, as well as from the first half
> of the 20th century.

Details:

We use the [`UD_UKRAINIAN`](https://github.com/flairNLP/flair/pull/3069) dataset and perform basic
hyper-parameter search.

Results (Development set, best hyper-param config):

| Model                          | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| ------------------------------ | ----- | ----- | ----- | ----- | ----- | -------------- |
| bert-base-multilingual-cased   | 98.03 | 98.11 | 98.18 | 98.02 | 97.95 | 98.06 ± 0.09
| xlm-roberta-base               | 98.57 | 98.47 | 98.49 | 98.40 | 98.43 | 98.47 ± 0.06
| facebook/xlm-v-base            | 98.50 | 98.48 | 98.54 | 98.56 | 98.60 | 98.54 ± 0.05
| Ukrainian ELECTRA (1M)         | 98.57 | 98.64 | 98.60 | 98.56 | 98.62 | **98.60** ± 0.03

Results (Test set, best hyper-param config on Development set):

| Model                          | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| ------------------------------ | ----- | ----- | ----- | ----- | ----- | -------------- |
| bert-base-multilingual-cased   | 97.90 | 97.89 | 97.98 | 97.84 | 97.94 | 97.91 ± 0.05
| xlm-roberta-base               | 98.33 | 98.51 | 98.43 | 98.41 | 98.43 | 98.42 ± 0.06
| facebook/xlm-v-base            | 98.39 | 98.37 | 98.47 | 98.15 | 98.44 | 98.36 ± 0.13
| Ukrainian ELECTRA (1M)         | 98.63 | 98.55 | 98.53 | 98.50 | 98.59 | **98.56** ± 0.05

## NER

We use the train split (`train.iob`) from [this `lang-uk` repository](https://github.com/lang-uk/flair-ner/tree/main/fixed-split)
and create 5 random splits train and development splits. We perform hyper-parameter search
on these 5 splits and select the best configuration (based on F1-Score on development set).
In the final step we use the best hyper-parameter configuration, train 5 models with
development data and evaluate them on the test split (`test.iob`) from the mentioned `lang-uk` repo.

The script `create_random_split.py` was used to create 5 random splits and all created data can be found
in the `./ner_experiments` folder in this repo.

Results (Development set, best hyper-param config):

| Model                          | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| ------------------------------ | ----- | ----- | ----- | ----- | ----- | -------------- |
| bert-base-multilingual-cased   | 90.55 | 89.89 | 90.16 | 90.84 | 90.81 | 90.45 ± 0.42
| xlm-roberta-base               | 92.25 | 91.99 | 91.72 | 90.54 | 91.35 | 91.57 ± 0.67
| Ukrainian ELECTRA (1M)         | 94.17 | 92.13 | 92.74 | 91.45 | 92.23 | **92.54** ± 1.02

Results (Test set, best hyper-param config on Development set incl. development data):

| Model                          | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| ------------------------------ | ----- | ----- | ----- | ----- | ----- | -------------- |
| bert-base-multilingual-cased   | 84.20 | 85.61 | 85.11 | 85.17 | 83.90 | 84.80 ± 0.72
| xlm-roberta-base               | 87.85 | 87.39 | 87.31 | 88.15 | 86.19 | 87.38 ± 0.75
| facebook/xlm-v-base            | 86.00 | 86.25 | 86.22 | 87.05 | 86.34 | 86.37 ± 0.4
| Ukrainian ELECTRA (1M)         | 88.16 | 87.96 | 88.39 | 88.14 | 87.68 | **88.07** ± 0.26

# Model usage

The Ukrainian ELECTRA model can be used from the [lang-uk](https://github.com/lang-uk) Hugging Face [model hub page](https://huggingface.co/lang-uk).

As ELECTRA is trainined with an generator and discriminator model, both models are available. The generator model is usually used for masked
language modeling, whereas the discriminator model is used for fine-tuning on downstream tasks like token or sequence classification.

The following model names can be used:

* Ukrainian ELECTRA (discriminator): `lang-uk/electra-base-ukrainian-cased-discriminator` - [model hub page](https://huggingface.co/lang-uk/electra-base-ukrainian-cased-discriminator)
* Ukrainian ELECTRA (generator): `lang-uk/electra-base-ukrainian-cased-generator` - [model hub page](https://huggingface.co/lang-uk/electra-base-ukrainian-cased-generator)

Example usage with 🤗 Transformers:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "lang-uk/electra-base-ukrainian-cased-generator"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelWithLMHead.from_pretrained(model_name)
```

# License

All models are licensed under [MIT](LICENSE).

# Contact (Bugs, Feedback, Contribution and more)

For questions about our Ukrainian ELECTRA model just open an issue in
[this repo](https://github.com/stefan-it/ukrainian-electra/issues/new) 🤗

# Citation

You can use the following BibTeX entry for citation:

```bibtex
@software{stefan_schweter_2020_4267880,
  author       = {Stefan Schweter},
  title        = {Ukrainian ELECTRA model},
  month        = nov,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.4267880},
  url          = {https://doi.org/10.5281/zenodo.4267880}
}
```

# Acknowledgments

Research supported with Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).
Thanks for providing access to the TRC ❤️

Thanks to the generous support from the [Hugging Face](https://huggingface.co/) team,
it is possible to download both cased and uncased models from their S3 storage 🤗
