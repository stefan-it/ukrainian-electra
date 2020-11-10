echo "Downloading preprocessing scripts"
wget https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py

mkdir -p data

cd data

# UK NER (we use Stanza-preprocessed data here)
git clone https://github.com/gawy/stanza-lang-uk.git
cd stanza-lang-uk
git clone https://github.com/lang-uk/ner-uk
python3 src/bsf_to_beios.py --src_dataset ./ner-uk/data/ -c iob --dst ./

for model in bert-base-multilingual-cased bert-base-multilingual-uncased xlm-roberta-base dbmdz/electra-base-ukrainian-cased-discriminator
do
    mkdir -p $model-data # append data, because preprocess would look into this folder for tokenizer config!!!!

    echo "Preprocessing Ukrainian-languk for $model (Phase II)"

    python3 ../../preprocess.py ./Ukrainian-languk/train.bio $model 128 > $model-data/train.txt
    python3 ../../preprocess.py ./Ukrainian-languk/dev.bio $model 128 > $model-data/dev.txt
    python3 ../../preprocess.py ./Ukrainian-languk/test.bio $model 128 > $model-data/test.txt
done

cd .. # stanza-lang-uk

cd .. # from data
