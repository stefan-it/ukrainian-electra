# 🇺🇦 Ukrainian ELECTRA model

In this repository we monitor all experiments for our trained [DBMDZ model](https://github.com/dbmdz/berts)
for Ukrainian. We use the awesome 🤗 Transformers library to fine-tune models.

Made with 🤗, 🥨 and ❤️ from Munich.

# Changelog

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

We use the awesome 🤗 Transformers library for all fine-tuning experiments.

Please star and watch [Transformers](https://github.com/huggingface/transformers) on GitHub!

All JSON-based configuration files for our experiments can be found in the
[configuration](https://github.com/stefan-it/ukrainian-electra/tree/main/configs) folder
in this repository. To replicate the results, just clone the latest version of Transformers, `cd`
into the `examples/token-classification` folder and run `python3 run_ner_old.py <configuration.json>`.

## PoS Tagging

### Ukrainian-IU

Description:

> UD Ukrainian comprises 122K tokens in 7000 sentences of fiction, news, opinion articles, Wikipedia,
> legal documents, letters, posts, and comments — from the last 15 years, as well as from the first half
> of the 20th century.

Details:

* [Ukrainian-IU Repository](https://github.com/UniversalDependencies/UD_Ukrainian-IU)
* Commit `758bdd3`

For a better reproducibility, the `download_prepare_data_ud.sh` shell script can be used to download and
preprocessing the data.

Results (Development set)

| Model                          | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| ------------------------------ | ----- | ----- | ----- | ----- | ----- | -------------- |
| bert-base-multilingual-cased   | 98.12 | 98.03 | 98.04 | 98.08 | 98.17 | 98.09 ± 0.05
| bert-base-multilingual-uncased | 97.93 | 98.01 | 97.90 | 97.87 | 98.00 | 97.94 ± 0.05
| xlm-roberta-base               | 98.54 | 98.63 | 98.59 | 98.54 | 98.51 | 98.56 ± 0.04
| Ukrainian ELECTRA (1M)         | 98.78 | 98.59 | 98.75 | 98.68 | 98.65 | **98.69** ± 0.07

Results (Test set)

| Model                          | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| ------------------------------ | ----- | ----- | ----- | ----- | ----- | -------------- |
| bert-base-multilingual-cased   | 97.96 | 97.98 | 97.88 | 97.90 | 97.99 | 97.94 ± 0.04
| bert-base-multilingual-uncased | 97.89 | 97.69 | 97.73 | 97.87 | 97.77 | 97.79 ± 0.08
| xlm-roberta-base               | 98.51 | 98.48 | 98.48 | 98.41 | 98.51 | 98.48 ± 0.04
| Ukrainian ELECTRA (1M)         | 98.70 | 98.61 | 98.66 | 98.50 | 98.73 | **98.64** ± 0.08

## NER

For NER we mainly use the data provided by `lang-uk`'s great `ner-uk` annotations, that can be found in
[this](https://github.com/lang-uk/ner-uk) repository. The annotations are in BRAT-compatible format, so
we use the `stanza-lang-uk` conversion tool from [Andrew Garkavyi's](https://github.com/gawy) awesome
[repository](https://github.com/gawy/stanza-lang-uk). This tool mainly converts the `ner-uk` data into
IOB-compatible tagging scheme incl. nice training, development and test splits.

Just use the `download_prepare_data_ner.sh` script to reproduce the preprocessing steps.

Results (Development set)

| Model                          | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| ------------------------------ | ----- | ----- | ----- | ----- | ----- | -------------- |
| bert-base-multilingual-cased   | 86.07 | 86.23 | 85.83 | 85.29 | 85.86 | 85.86 ± 0.32
| bert-base-multilingual-uncased | 78.86 | 77.56 | 79.11 | 79.18 | 79.93 | 78.93 ± 0.77
| xlm-roberta-base               | 88.32 | 85.31 | 87.15 | 86.37 | 86.75 | 86.78 ± 0.98
| Ukrainian ELECTRA (1M)         | 88.01 | 87.63 | 88.29 | 87.24 | 88.63 | **87.96** ± 0.49

Results (Test set)

| Model                          | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.
| ------------------------------ | ----- | ----- | ----- | ----- | ----- | -------------- |
| bert-base-multilingual-cased   | 84.72 | 84.99 | 85.49 | 85.69 | 85.40 | 85.26 ± 0.35
| bert-base-multilingual-uncased | 75.50 | 74.02 | 73.64 | 72.42 | 71.86 | 73.49 ± 1.28
| xlm-roberta-base               | 87.31 | 85.93 | 89.39 | 85.40 | 85.71 | 86.75 ± 1.47
| Ukrainian ELECTRA (1M)         | 87.38 | 89.32 | 87.61 | 86.98 | 88.72 | **88.00** ± 0.88

Notice: Maybe accent stripping and lowercasing the input is not a good idea when using NER tasks
for Ukrainian with an uncased model (like uncased mBERT)!

# Model usage

The Ukrainian ELECTRA model can be used from the [DBMDZ](https://github.com/dbmdz) Hugging Face [model hub page](https://huggingface.co/dbmdz).

As ELECTRA is trainined with an generator and discriminator model, both models are available. The generator model is usually used for masked
language modeling, whereas the discriminator model is used for fine-tuning on downstream tasks like token or sequence classification.

The following model names can be used:

* Ukrainian ELECTRA (discriminator): `dbmdz/electra-base-ukrainian-cased-discriminator` - [model hub page](https://huggingface.co/dbmdz/electra-base-ukrainian-cased-discriminator)
* Ukrainian ELECTRA (generator): `dbmdz/electra-base-ukrainian-cased-generator` - [model hub page](https://huggingface.co/dbmdz/electra-base-ukrainian-cased-generator)

Example usage with 🤗 Transformers:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "dbmdz/electra-base-ukrainian-cased-discriminator"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelWithLMHead.from_pretrained(model_name)
```

# License

All models are licensed under [MIT](LICENSE).

# Contact (Bugs, Feedback, Contribution and more)

For questions about our Ukrainian ELECTRA model just open an issue
[in the DBMDZ BERT repo](https://github.com/dbmdz/berts/issues/new) or in
[this repo](https://github.com/stefan-it/uktrainian-electra/issues/new) 🤗

# Acknowledgments

Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).
Thanks for providing access to the TFRC ❤️

Thanks to the generous support from the [Hugging Face](https://huggingface.co/) team,
it is possible to download both cased and uncased models from their S3 storage 🤗
