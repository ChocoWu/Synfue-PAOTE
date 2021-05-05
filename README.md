# Synfue-PAOTE
This repository implements the dependency parser described in the paper [Learn from Syntax: Improving Pair-wise Aspect and Opinion Terms Extraction with Rich Syntactic Knowledge]()
## Prerequisite
[pytorch Library](https://pytorch.org/)
[transformers](https://huggingface.co/transformers/model_doc/bert.html)

## Usage (by examples)
### Train
We use embedding bert-cased by [bert-base-cased](https://huggingface.co/bert-base-cased) (768d)

```
  python Synfue.py train --config configs/16res_train.conf
```
### Test
```
  python Synfue.py test --config configs/16res_test.conf
```
