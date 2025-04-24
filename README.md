# [Retrieval Augmented Zero-Shot Text Classification (QZero)](https://arxiv.org/pdf/2406.15241)
## Description

We introduce QZero, a novel training-free approach that reformulates queries by retrieving supporting categories from Wikipedia to improve zero-shot classification performance. Our experiments across six diverse datasets demonstrate that QZero enhances performance for state-of-the-art static and contextual embedding models without the need for retraining. Notably, in News and medical topic classification tasks, QZero improves the performance of even the largest OpenAI embedding model by at least 5% and 3%, respectively. Acting as a knowledge amplifier, QZero enables small word embedding models to achieve performance levels comparable to those of larger contextual models, leading to significant computational savings. Additionally, QZero offers meaningful insights that illuminate query context and verify topic relevance, aiding in understanding model predictions. Overall, QZero improves embedding-based zero-shot classifiers while maintaining their simplicity. This makes it particularly valuable for resource-constrained environments and domains with constantly evolving information. The figure below describes the entire Zero-shot classification process:


![Overview of QZero](QZero.jpg)

## Requirements

Python 3.9.9,
Pyterrier,
BEIR,
MedCAT,

To install requirements:

```setup
pip install -r requirements.txt
```
# Datasets
Knowledge Corpus stored in wiki folder: Wikipedia dumps version: enwiki-20230820-pages-articles-multistream.xml.bz2

Download the full wikidumps form [wikimedia](https://dumps.wikimedia.org/)

Access the exact wiki files we used via this [link](https://drive.google.com/drive/folders/1hwUhcqI0I8k9x5t15XbIaWMKhm7d6NV3)

Test data: We provide some test datasets in the input data folder, but they can be accessed via the following links
1. [Ag_news](https://huggingface.co/datasets/ag_news)
2. [DBpedia](https://huggingface.co/datasets/fancyzhx/dbpedia_14)
3. [Yahoo](https://huggingface.co/datasets/yahoo_answers_topics)
4. [Ohsumed](https://disi.unitn.it/moschitti/corpora.htm)
5. [What's Cooking](https://www.kaggle.com/competitions/whats-cooking/data)
6. [TagMyNews](https://github.com/AIRobotZhang/STCKA/tree/master)

To reproduce our results, you can access the reformulated queries for all datasets via: [Reformulated queries]()

The link to the reformulated queries will be provided after publication.

## To Build the Index and Reformulate Queries:
```build sparse index
python3 Sparse_Index.py /path/to/save/index /path/to/wiki/data /path/to/test/query /path/to/save/reformulated/query /keyword_extraction_method/ /top_K/
```

```build dense index
python3 Dense_Index.py /path/to/save/index /path/to/wiki/data /path/to/test/query /path/to/save/reformulated/query /keyword_extraction_method/ /top_K/
```

## To Evaluate QZero
Models we used are: Word2Vec, GloVe, Fasttext, all-mpnet-base-v2, text-embedding-3-small, text-embedding-3-large.
Download static word embedding models to the QZERO directory.
update OPEN AI API KEY in the Models.py

```build dense index
python3 Evaluate_QZero.py /embedding_model_name/ /path/to/reformulated/query/ /dataset_labels/
```
Note: All the scripts use default paths and settings. You can override these defaults by providing your own paths or settings when running the script.
For detailed information on the arguments and their expected values, use the --help flag.

## **If you find QZero useful in your work, please cite us!**

<pre> ```bibtex @inproceedings{abdullahi2024retrieval, 
 title={Retrieval augmented zero-shot text classification}, 
 author={Abdullahi, Tassallah and Singh, Ritambhara and Eickhoff, Carsten}, 
 booktitle={Proceedings of the 2024 ACM SIGIR international conference on theory of information retrieval}, 
 pages={195--203}, 
 year={2024} } ``` </pre>
