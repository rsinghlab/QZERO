import os
import pandas as pd
import argparse
import pyterrier as pt
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
import spacy
import Helpers

# Download required data for nltk
import re
import nltk
import string
# nltk.download('punkt')
# nltk.download('stopwords')
import ast
from tqdm import tqdm
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


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
    df.reset_index(drop=True, inplace=True)
    return df

def prepare_wiki_data(folder_path):
    data, doc_ids = [], []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            process_file(file_path, data)
    
    df = pd.DataFrame(data)
    df = preprocess_dataframe(df)

    for i in range(1, len(df) + 1):
        doc_ids.append(f'doc{i:02d}')
    
    df['docno'] = doc_ids
    df.columns = ['title', 'categories', 'text', 'docno']
    df['categories'] = df['categories'].apply(lambda x: [x])
    return df

def build_index(df, index_dir):
    if not pt.started():
        pt.init()
    print("Pyterrirer started")

    indexer = pt.IterDictIndexer(index_dir, verbose=True, meta={'docno': 20, "title": 256, "categories": 4096})
    indexer.index(df.fillna("").to_dict(orient='records'), fields=['title', 'text'])
    print('Done with building index')
    index = pt.IndexFactory.of(index_dir)
    print(index.getCollectionStatistics().toString())
    return index


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
    
def preprocess_queries(file_path):
    query_df = pd.read_csv(file_path)
    query_df['text'] = query_df['text'].apply(tokenize)
    query_df['tokenized'] = query_df['text'].apply(process_text)
    query_df = query_df[query_df['tokenized'].str.split().str.len() >= 1].reset_index(drop=True)
    query = query_df[["tokenized"]]
    query.reset_index(inplace=True)
    query.columns= ['qid', 'query']
    return query

def preprocess_test_df(file_path):
    query_df = pd.read_csv(file_path)
    query_df['text'] = query_df['text'].apply(tokenize)
    query_df['tokenized'] = query_df['text'].apply(process_text)
    query_df = query_df[query_df['tokenized'].str.split().str.len() >= 1].reset_index(drop=True)
    query_df['qid'] = [str(i) for i in range(1, len(query_df) + 1)]
    test_df= query_df[['qid', 'text', 'label', 'category_name', 'tokenized']]
    #convert colunm to int
    test_df['qid'] = test_df['qid'].astype(int)
    return test_df

def retrieve_and_save(index,query_df,top_k):
    bm25 = pt.BatchRetrieve(index, num_results=top_k, wmodel="BM25", metadata=["docno", 'title', "categories"]).parallel(2)
    bm25_news = bm25.transform(query_df)
    print("Done")
    return bm25_news

def main():
    parser = argparse.ArgumentParser(description='Run information retrieval processes with Sparse retriever.')
    parser.add_argument('--index_dir', type=str, default='Sparse_wiki_index')
    parser.add_argument('--wiki_folder_path', type=str, default='wiki')
    parser.add_argument('--query_folder', type=str, default='input_data/ag_test.csv')
    parser.add_argument('--output_dir', type=str, default='retrieved_results/results_bm25.csv')
    parser.add_argument('--noun_type', type=str, default='spacy', choices=['proper', 'spacy', 'medical'])

    args = parser.parse_args()

    # Initialize pyterrier if not already done
    if not pt.started():
        pt.init(mem=20000)
    index_dir = args.index_dir
    wiki_path = args.wiki_folder_path
    print(index_dir)
    # Check if index already exists
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        print("No existing index found, preparing data and building index...")
        df = prepare_wiki_data(wiki_path)
       
        index = build_index(df,index_dir)
    else:
        print("Loading ... Index already exists.")
        index = pt.IndexFactory.of(index_dir)
        print(index.getCollectionStatistics().toString())

    queries = preprocess_queries(args.query_folder)
    results = retrieve_and_save(index=index, query_df=queries, top_k=10)
    results.to_csv(args.output_dir, index=False)
    print("Retrieval process completed and results saved.")

    test_df = preprocess_test_df(args.query_folder)
    results = Helpers.handle_incomplete_patterns(results)
    n_categories = 50
    merged_results = Helpers.select_N_merge(results, n_categories=n_categories, tests=test_df)

    print(f"Using {args.noun_type} type for keyword extraction")
    if args.noun_type == 'proper':
        merged_results["top_nouns"] = merged_results["sentence"].apply(lambda x: Helpers.extract_proper_nouns(x))
    elif args.noun_type == 'spacy':
        tqdm.pandas()  # Initialize tqdm for pandas progress_apply
        merged_results["top_nouns"] = merged_results["sentence"].progress_apply(lambda x: Helpers.extract_spacy_nouns(x))
    elif args.noun_type == 'medical':
        medcat_path = 'path_to_medcat_model.zip'
        merged_results = Helpers.get_med_nouns(merged_results, 'categories', medcat_path)

    merged_results.to_csv(args.output_dir, index=False)
    print("Queries reformulated, saved and ready for classification.")

if __name__ == '__main__':
    main()
