import os
import json
import csv
import pandas as pd
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch
from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import logging
import random
import re
import nltk
from nltk.corpus import stopwords
import Helpers
from Helpers import GenericDataLoader
import string
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing.preprocessing import strip_tags
from gensim.utils import simple_preprocess
import ast
import argparse
from tqdm import tqdm

# Setting up logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

tqdm.pandas()


# Define stop words
stop_words = set(stopwords.words('english'))

# Function to process text data
def process_text(text):
    """Remove stopwords and punctuations from text, then tokenize and remove special characters."""
    words = nltk.word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalnum()]
    return ' '.join(filtered_words)

def tokenize(doc):
    if isinstance(doc, float):
        # Handle the case where doc is a float (e.g., NaN)
        return ''  # or any other appropriate handling for missing values
    else:
        # Assuming strip_tags is a function to remove HTML tags
        tokenized = simple_preprocess(strip_tags(doc), deacc=True, min_len=2, max_len=15)
        return ' '.join(tokenized)

def process_file(file_path, data):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    title, categories, content = "", "", ""
    is_in_content = False

    for line in lines:
        line = line.strip()
        if line.startswith("[[") and line.endswith("]]"):
            if title:
                data.append({'title': title, 'categories': categories, 'content': content})
                title, categories, content = "", "", ""
            title = line[2:-2]
        elif line.startswith("CATEGORIES:"):
            categories = line.split(":")[1].strip()
        elif line.startswith("==") and line.endswith("=="):
            is_in_content = not is_in_content
        else:
            content += line + " "
    
    if title:
        data.append({'title': title, 'categories': categories, 'content': content})

def preprocess_dataframe(df):
    df = df.drop_duplicates(subset=['title', 'categories', 'content'])
    df = df[df['categories'].str.len() > 0]
    df = df[df['content'].apply(lambda x: len(x.split()) >= 20)]
    df['categories'] = df['categories'].apply(lambda x: [x])
    df.columns = ['title', 'categories', 'text']
    df ['categories']=df['categories'].astype(str)
    df.reset_index(drop=True, inplace=True)
    
    return df

def preprocess_data(folder_path):
    data, doc_ids = [], []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            process_file(file_path, data)
    
    data = pd.DataFrame(data)
    data = preprocess_dataframe(data)

    stop_words = set(stopwords.words('english'))
    data['text'] = data['text'].apply(lambda x: ' '.join([word.lower() for word in nltk.word_tokenize(x) 
                                                         if word.lower() not in stop_words and word.lower() not in string.punctuation]))
    data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    data = data[data['text'].str.split().str.len() > 5].dropna().reset_index(drop=True)
    data['_id'] = data.index.to_series().apply(lambda x: str(x + 1))
    data = data[['_id', 'title', 'text', 'categories']]
    data.columns=["_id","title","text","metadata"]
    output_file = "/Users/tabdull1/Python_Projects/RAZL_Projects/QZERO_PAPER/Wiki_corpus.jsonl"
    data.to_json(output_file, orient="records", lines=True)
    print("Done Creating wiki json files")
    return output_file

# Load and preprocess wiki dataframe
def preprocess_dataX(filepath):
    stop_words = set(stopwords.words('english'))
    data = pd.read_csv(filepath).head(300)
    data['text'] = data['text'].apply(lambda x: ' '.join([word.lower() for word in nltk.word_tokenize(x) 
                                                         if word.lower() not in stop_words and word.lower() not in string.punctuation]))
    data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    data = data[data['text'].str.split().str.len() > 5].dropna().reset_index(drop=True)
    data['_id'] = data.index.to_series().apply(lambda x: str(x + 1))
    data = data[['_id', 'title', 'text', 'categories']]
    data.columns=["_id","title","text","metadata"]
    output_file = "/Users/tabdull1/Python_Projects/RAZL_Projects/QZERO_PAPER/Wiki_corpus.jsonl"
    data.to_json(output_file, orient="records", lines=True)
    print("Done Creating wiki json files")
    return output_file


def preprocess_queries_and_save(file_path):
    query_df = pd.read_csv(file_path).head(100)
    query_df['text'] = query_df['text'].apply(tokenize)
    query_df['processed'] = query_df['text'].apply(process_text)
    query_df = query_df[query_df['processed'].str.split().str.len() >= 1].reset_index(drop=True)
    query_df['_id'] = [str(i) for i in range(1, len(query_df) + 1)]
    queries_dict = query_df[["_id", "processed"]].rename(columns={'processed': 'text'})
    output_file = "/Users/tabdull1/Python_Projects/RAZL_Projects/QZERO_PAPER/Wiki_queries.jsonl"
    queries_dict.to_json(output_file, orient="records", lines=True)
    return output_file

def preprocess_test_df(file_path):
    query_df = pd.read_csv(file_path).head(100)
    query_df['text'] = query_df['text'].apply(tokenize)
    query_df['tokenized'] = query_df['text'].apply(process_text)
    query_df = query_df[query_df['tokenized'].str.split().str.len() >= 1].reset_index(drop=True)
    query_df['qid'] = [str(i) for i in range(1, len(query_df) + 1)]
    test_df= query_df[['qid', 'text', 'label', 'category_name', 'tokenized']]
    #convert colunm to int
    test_df['qid'] = test_df['qid'].astype(int)
    return test_df



# Dense Retrieval using Different Faiss Indexes (Flat or ANN) ####
    # Provide any Sentence-Transformer or Dense Retriever model.

def evaluate_retrieval(model_path, corpus, queries, index_dir, prefix="my-index", ext="flat", top_k=10):

    # Initialize the SentenceBERT model from BEIR.

    # Provide any Sentence-Transformer or Dense Retriever model.
    
    model = models.SentenceBERT(model_path)

    faiss_search = FlatIPFaissSearch(model,batch_size=128)
  
    model = models.SentenceBERT(model_path)
    # Initialize the FAISS search (using FlatIP index in this example).
    faiss_search = FlatIPFaissSearch(model, batch_size=128)
    
    if os.path.exists(os.path.join(index_dir, "{}.{}.faiss".format(prefix, ext))): 
        faiss_search.load(input_dir=index_dir, prefix=prefix, ext=ext)
        print("loading indexx")

    # Set up the retrieval evaluation, including the scoring function and the list of k values to evaluate.
    retriever = EvaluateRetrieval(faiss_search, score_function="dot",k_values=[top_k]) # or "cos_sim"/"dot"
    results = retriever.retrieve(corpus, queries)

    # Create the output directory if it doesn't exist
    os.makedirs(index_dir, exist_ok=True)

    # Construct the full index file path
    index_file = os.path.join(index_dir, "{}.{}.faiss".format(prefix, ext))

    # Only save the index if the file doesn't already exist
    if not os.path.exists(index_file):
        faiss_search.save(output_dir=index_dir, prefix=prefix, ext=ext)
        print('saving index')
    else:
        print("Index already exists, skipping saving.")  # Optional message

    
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    return results

# Store results in DataFrame
def store_results(results, corpus, top_k):
    dfs = []
    
    for query_id, ranking_scores in results.items():
        scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
        dataX = []
        for rank in range(top_k):
            doc_id = scores_sorted[rank][0]
            title = corpus.get(doc_id, {}).get("title", "Title Not Available")
            meta = corpus.get(doc_id, {}).get("metadata", "Meta Not Available")
            dataX.append([query_id, rank+1, doc_id, title, meta])
        query_dfX = pd.DataFrame(dataX, columns=["qid", "Rank", "Doc_ID", "Title", "categories"])
        dfs.append(query_dfX)

    # Concatenate all query-specific DataFrames into a single DataFrame
    dfX = pd.concat(dfs, ignore_index=True)
    return dfX

def main():
    parser = argparse.ArgumentParser(description='Run information retrieval processes with Dense retriever')
    parser.add_argument('--wiki_data_path', type=str, default='wiki', help='Path to the Wikipedia data files')
    parser.add_argument('--query_path', type=str, default='input_data/ag_test.csv', help='Path to the query file')
    parser.add_argument('--model_path', type=str, default='facebook/contriever', help='Model path for retrieval')
    parser.add_argument('--index_dir', type=str, default='faiss-index-wiki', help='Directory for storing FAISS index')
    parser.add_argument('--noun_type', type=str, default='spacy', choices=['proper', 'spacy', 'medical'], help='Type of noun extraction to perform')
    parser.add_argument('--result_file', type=str, default='retrieved_results/results_dpr.csv', help='Filename to store the final results')
    parser.add_argument('--top_k', type=int, default=50, help='Top K results to retrieve')

    args = parser.parse_args()

    # Data preprocessing
    corpus_file = preprocess_data(args.wiki_data_path)
    query_file = preprocess_queries_and_save(args.query_path)
    print("Loaded queries")
    
    data_loader = GenericDataLoader(data_folder='/path/to/project/jsonfiles/QZERO', corpus_file=corpus_file, query_file=query_file)
    corpus, queries, qrels = data_loader.load_custom()
    print("Done loading files")
    print(args.index_dir)
    # Perform indexing and retrieval
    ret_results = evaluate_retrieval(args.model_path, corpus, queries, index_dir=args.index_dir, top_k=args.top_k)
    results = store_results(ret_results, corpus, args.top_k)
    print("Retrieval process completed and results saved.")

    # Prepare reformed queries classification dataframe
    test_df = preprocess_test_df(args.query_path)
    results = Helpers.handle_incomplete_patterns(results)
    n_categories = 50 #categories to merge from 
    merged_results = Helpers.select_N_merge(results, n_categories=n_categories, tests=test_df)

    # Noun type extraction based on user choice
    print(f"Using {args.noun_type} type for keyword extraction")
    if args.noun_type == 'proper':
        merged_results["top_nouns"] = merged_results["sentence"].apply(lambda x: Helpers.extract_proper_nouns(x))
    elif args.noun_type == 'spacy':
        merged_results["top_nouns"] = merged_results["sentence"].progress_apply(lambda x: Helpers.extract_spacy_nouns(x))
    elif args.noun_type == 'medical':
        medcat_path = "umls_sm_pt2ch_533bab5115c6c2d6.zip" #'path_to_medcat_model.zip'
        merged_results = Helpers.get_med_nouns(merged_results, 'categories', medcat_path)

    merged_results.to_csv(args.result_file, index=False)
    print("Queries reformulated, saved, and ready for classification.")

if __name__ == '__main__':
    tqdm.pandas()
    main()

