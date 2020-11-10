echo "Downloading preprocessing scripts"
wget https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py

mkdir -p data

cd data

# UD_Ukrainian-IU 
git clone https://github.com/UniversalDependencies/UD_Ukrainian-IU.git
cd UD_Ukrainian-IU
git checkout 758bdd3

echo "Preprocessing UD_Italian-ISDT (Phase I)"
cat uk_iu-ud-train.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" | tr '\t' ' ' > train.txt.tmp
cat uk_iu-ud-dev.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" | tr '\t' ' ' > dev.txt.tmp
cat uk_iu-ud-test.conllu | cut -f 2,4 | grep -v "_" | grep -v "^#" | tr '\t' ' ' > test.txt.tmp

for model in bert-base-multilingual-cased bert-base-multilingual-uncased xlm-roberta-base dbmdz/electra-base-ukrainian-cased-discriminator
do
    mkdir -p $model-data # append data, because preprocess would look into this folder for tokenizer config!!!!

    echo "Preprocessing UD_Ukrainian-IU for $model (Phase II)"

    python3 ../../preprocess.py ./train.txt.tmp $model 128 > $model-data/train.txt
    python3 ../../preprocess.py ./dev.txt.tmp $model 128 > $model-data/dev.txt
    python3 ../../preprocess.py ./test.txt.tmp $model 128 > $model-data/test.txt
done

cd .. # from UD_Ukrainian-IU

cd .. # from data

