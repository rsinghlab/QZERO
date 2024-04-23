
import spacy.cli
from tqdm import tqdm
import re
import pandas as pd
import ast

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
import os

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import PQFaissSearch, HNSWFaissSearch, FlatIPFaissSearch, HNSWSQFaissSearch 
import spacy   

import logging
import pathlib, os
import random
from collections import Counter

from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging
import csv
import ast

import pandas as pd
from collections import Counter
import ast
from medcat.cat import CAT


logger = logging.getLogger(__name__)


from time import time


import logging
import pathlib, os
import random
import pandas as pd


class GenericDataLoader:

    def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl",
                 qrels_folder: str = "qrels", qrels_file: str = None):  # Set default value to None
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not fIn:  # Add this line to skip the check if file is None
            return  # Add this line to skip the check if file is None
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))


    
    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:


        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if self.qrels_file and os.path.exists(self.qrels_file):  # Modified line
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels


    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if self.qrels_file and os.path.exists(self.qrels_file):  # Modified line
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels

    
    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus
    
    def _load_corpus(self):
    
        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                    "metadata":line.get("metadata")
                }
    
    def _load_queries(self):
        
        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")
        
    def _load_qrels(self):
        
        reader = csv.reader(open(self.qrels_file, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)
        
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            
            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score
# The qrels variable will be empty since the qrels file does not exist



def extract_proper_nouns(sentence):
#   """Extracts proper nouns from a sentence, counts the number of times each one is mentioned, and sorts the list items according to count.

#   Args:
#     sentence: The sentence to extract proper nouns from.

#   Returns:
#     A list of tuples of the form (proper_noun, count), sorted by count in descending order.
#   """

  patterns =[re.compile(r"\b[A-Z][a-z]+\b"), re.compile(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b")]

  proper_nouns_counts = {}
  for pattern in patterns:
    matches = pattern.finditer(sentence)
    for match in matches:
      proper_noun = match.group()
      if proper_noun not in proper_nouns_counts:
        proper_nouns_counts[proper_noun] = 0
      proper_nouns_counts[proper_noun] += 1

  sorted_proper_nouns_counts = sorted(
      proper_nouns_counts.items(), key=lambda x: x[1], reverse=True)

  return sorted_proper_nouns_counts


#Load spaCy model once and disable unnecessary components
nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "lemmatizer"])

def extract_spacy_nouns(sentence):
    # Process the sentence with spaCy NLP pipeline
    doc = nlp(sentence)

    # Initialize a dictionary to store noun counts
    noun_counts = {}

    # Iterate through tokens
    for token in doc:
        if token.pos_ == "NOUN": # or token.pos_ == "PROPN":
            # Convert the noun to lowercase for consistent counting
            noun = token.text.lower()
            noun_counts[noun] = noun_counts.get(noun, 0) + 1

    # Get the top noun from the sentence
    top_nouns = sorted(noun_counts.items(), key=lambda x: x[1], reverse=True)#[0:5]
    return top_nouns


def handle_incomplete_patterns(results):

        # Identify rows with incomplete patterns using regular expressions
    incomplete_pattern_rows =results[
                                results['categories'].str.count('\[') != results['categories'].str.count('\]')]


    incomplete_pattern_rows = incomplete_pattern_rows[incomplete_pattern_rows['categories'].str.contains(r'^\[".*[^]]$', regex=True)]


        # # Update incomplete patterns using vectorized operations
    results.loc[incomplete_pattern_rows.index, 'categories'] = results.loc[incomplete_pattern_rows.index, 'categories'].apply(lambda x: x + '"]')

        #convert colunm to int
    results['qid'] = results['qid'].astype(int)


        # Convert the strings to actual lists using ast.literal_eval
    results['categories'] = results['categories'].apply(ast.literal_eval)

    return results

    
def list_to_sentence(string_list):
    return " ".join(string_list)

     # Select the first 50 rows of each 'qid' group
def select_N_merge(results,n_categories, tests):
    first_n_per_group = results.groupby('qid').head(n_categories)
        # Now, group by 'qid' again and aggregate 'categories' by sum
    news_grouped = first_n_per_group.groupby('qid', as_index=False)['categories'].agg('sum')

    ids = news_grouped.qid.values.tolist()
    full_range = set(range(1, len(tests)))
    missing_ids = [num for num in full_range if num not in ids]
    
    filtered_df = tests.copy()

        # Filter out rows where '_id' values are present in 'missing_ids' list
    filtered_df = filtered_df[~filtered_df['qid'].isin(missing_ids)]
        #reset index of filtered_df
    filtered_df = filtered_df.reset_index(drop=True)

    filtered_df['qid'] = filtered_df['qid'].astype(int)


        # Merge both DataFrames on 'qid' while maintaining the ordering
    merged_df = pd.merge_ordered(news_grouped, filtered_df, on='qid',how='inner')
    merged_df = merged_df.reset_index(drop=True)

        # Apply the function to the DataFrame column
    merged_df['sentence'] = merged_df['categories'].apply(list_to_sentence)

    return merged_df


def get_med_nouns(df, categories_column, model_path):
    # Load the CAT model
    cat = CAT.load_model_pack(model_path)

    # Helper function to process each category into a sentence
    def process_category_string(input_string):
        list_of_strings = ast.literal_eval(input_string)
        return ", ".join(list_of_strings)

    # Function to get entities using CAT
    def get_entities(text):
        return cat.get_entities(text)

    # Function to extract names from entity information
    def extract_names(entities_dict):
        return [entity_info['source_value'] for entity_id, entity_info in entities_dict['entities'].items() if 'source_value' in entity_info]

    # Function to count word occurrences in a list
    def count_word_occurrences(words):
        word_counts = Counter([word.lower() for word in words])
        return sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Process the categories column to create a 'sentence' column
    df['sentence'] = df[categories_column].apply(process_category_string)

    # Extract entities from the 'sentence' column
    df['Entities'] = df['sentence'].apply(get_entities)

    # Extract medical information from entities
    df['med_info'] = df['Entities'].apply(extract_names)

    # Count occurrences of each word in 'med_info'
    df['top_nouns'] = df['med_info'].apply(count_word_occurrences)

    # Join words into sentences
    df['sentence'] = df['med_info'].apply(lambda x: ', '.join(x))

    return df
