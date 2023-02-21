code, model and datasets of our paper "MRC-PASCL: A Few-shot **M**achine **R**eading **C**omprehension Approach via **P**ost-training and **A**nswer **S**pan-oriented **C**ntrastive **L**earning".

Our pretraining code is based on PyTorch(1.8.1) and transformers(4.21.0), while fine-tuning code is identical with [splinter](https://github.com/oriram/splinter) in requirements.

### Data
#### Downloading Fine-tuning data
```
curl -L https://www.dropbox.com/sh/pfg8j6yfpjltwdx/AAC8Oky0w8ZS-S3S5zSSAuQma?dl=1 > mrqa-few-shot.zip
unzip mrqa-few-shot.zip -d data/mrqa-few-shot
```

#### Downloading openwebtext and ccnews
Our pretraining data is based on openwebtext and ccnews, openwebtext corpus can be downloaded in https://skylion007.github.io/OpenWebTextCorpus/, ccnews can be downloaded in https://storage.googleapis.com/huggingface-nlp/datasets/cc_news/cc_news.tar.gz

### Model 
#### Downloading Splinter Model
```
curl -L https://www.dropbox.com/sh/h63xx2l2fjq8bsz/AAC5_Z_F2zBkJgX87i3IlvGca?dl=1 > splinter.zip
unzip splinter.zip -d pretrained_model/splinter 
```

#### Dowloading MRC-PASCL Model

MRC-PASCL Model can be downloaded in 
https://drive.google.com/file/d/1YAVF34GBkEYVUUqTX5NoH2--POCxhtUR/view?usp=sharing

### Pre-training

#### Preprocessing openwebtext and ccnews
we first get origin text from openwebtext and ccnews, then we select the appropriate passage while the passage token number is greater than 250

#### Making Pretraing Data
Run 
```
python pretraining/making_pretrained_data.py 
```

#### Training the MRC-PASCL Model
Run
```
bash pretraining/run_pretraining.sh
```

### Fine-tuning
Run
```
bash finetuning/finetuning.sh
```

### Acknowledgements
We are very grateful the authors of splinter for opening source code and pretrained model
