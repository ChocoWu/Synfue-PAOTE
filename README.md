# Syntax fusion Encoder for the PAOTE task (Synfue)
This repository implements the dependency parser described in the paper [Learn from Syntax: Improving Pair-wise Aspect and Opinion Terms Extraction with Rich Syntactic Knowledge]()
## Prerequisite
* [pytorch Library](https://pytorch.org/)
* [transformers](https://huggingface.co/transformers/model_doc/bert.html)
* [corenlp](https://stanfordnlp.github.io/CoreNLP/)

## Usage (by examples)
### Data
Orignal data comes from [TOWE](https://github.com/NJUNLP/TOWE/tree/master/data).


### Preprocessing
We need to obtain the dependency sturcture and POS tags for each data, and save as json format.
Pay attention to the file path and modify as needed.

#### Get Dependency and POS
To parse the dependency structure and POS tags, we employ the [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) provided by stanfordnlp.
So please download relavant files first and put it in `data/datasets/orignal`.
We use the NLTK package to obtain the dependency and POS parsing, so we need to modify the code as follows in `process.py` line 24: 
```
depparser = CoreNLPDependencyParser(url='http://127.0.0.1:9000')
```
The url is set according to the real scenario.

#### Save
```
  python preprocess.py
```
The proposed data will be sotored in the dicretory `data/datasets/towe/`.
We also provide some preprocessed examples. 

### Train
We use embedding bert-cased by [bert-base-cased](https://huggingface.co/bert-base-cased) (768d)

```
  python synfue.py train --config configs/16res_train.conf
```
### Test
```
  python synfue.py eval --config configs/16res_eval.conf
```

## Note
this code refers to the [SpERT](https://github.com/lavis-nlp/spert)
