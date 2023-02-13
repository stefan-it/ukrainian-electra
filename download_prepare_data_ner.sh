echo "Downloading preprocessing scripts"
wget https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py

mkdir -p data

cd data

# UK NER (we use Stanza-preprocessed data here)
git clone https://github.com/gawy/stanza-lang-uk.git
cd stanza-lang-uk
git clone https://github.com/lang-uk/ner-uk
python3 src/bsf_beios/bsf_to_beios.py --src_dataset ./ner-uk/data/ -c iob --dst ./

cd .. # stanza-lang-uk
cd .. # data

# Data lies under:
# ./data/stanza-lang-uk/Ukrainian-languk
