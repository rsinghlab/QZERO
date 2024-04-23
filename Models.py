import random
import numpy as np
import torch
import os
seed = 2
random.seed(seed)
np.random.seed(seed) 
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from gensim.models import KeyedVectors
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from gensim.models import KeyedVectors
from collections import Counter
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model, load_facebook_vectors
from gensim.test.utils import datapath
import gensim.downloader as api
import ast
from collections import defaultdict
from numpy.linalg import norm
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from transformers import GPT2Tokenizer
import openai
import re
from tqdm import tqdm

def load_models(model_name):
    # Load specific model based on the user input
    if model_name == 'glove':
        print("Loading glove")
        glove = KeyedVectors.load_word2vec_format('glove.840B.300d.txt', binary=False, no_header=True)
        return glove
        
    elif model_name == 'word2vec':
        print("Loading word2vec")
        word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
        return word2vec

    elif model_name == 'fasttext':
        print("Loading fasttext")
        fasttext = load_facebook_vectors('wiki.en/wiki.en.bin')
        # cap_path = datapath("wiki.en/wiki.en.bin")
        # return load_facebook_vectors(cap_path)
        return fasttext

    elif model_name == 'sbert':
        print("Loading sbert")
        sbert_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
        sbert_model.max_seq_length = 500
        return sbert_model

    elif model_name  == 'gpt_small':
        gpt_small = "text-embedding-3-small"
        return gpt_small

    elif model_name  == 'gpt_large':
    
        gpt_large = "text-embedding-3-large"
        return gpt_large

    else:
        raise ValueError("Invalid model name provided. Choose from 'glove', 'word2vec', 'fasttext', 'sbert', 'gpt-small', 'gpt_large'. ")


def SBERT_Model(input_sentence, sentences, model):
    # Compute embeddings
    input_embedding = model.encode(input_sentence, convert_to_tensor=True)
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute cosine-similarities for the input sentence with each other sentence
    cosine_scores = util.pytorch_cos_sim(input_embedding, embeddings)[0]

    # Find the index of the maximum similarity score
    most_similar_index = torch.argmax(cosine_scores).item()

    return most_similar_index
    
def average_vector_representation(words,model):
    # Check if the input is a single word
    if len(words) == 1 and words[0] in model:
        # Return the vector representation of the single word
        return model[words[0]]

    # For phrases: Only consider words that are in the model's vocabulary
    valid_words = [word for word in words if word in model]

    # Get vectors for the valid words
    vectors = [model[word] for word in valid_words]

    # If vectors is empty, return None (or any other default value you prefer)
    if not vectors:
        return None

    # Otherwise, compute and return the average vector
    average_vector = sum(vectors) / len(vectors)
    return average_vector

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def QZero_Word_Model(top,category_dicts,model):
    category_list = [item for sublist in category_dicts.values() for item in sublist]#[:-1]
    category_score_sums = defaultdict(float)

    for item, count in top:
        item_vector = average_vector_representation([item],model)  # Get vector representation

        # Calculate similarity with each category
        for category in category_list:
            category_words = category.lower().split()  # Split multi-word category names
            category_vector = average_vector_representation(category_words,model)

            if item_vector is not None and category_vector is not None:
                # Calculate similarity using cosine similarity
                similarity = cosine_similarity(item_vector, category_vector)
                
                # Multiply by count and add to the sum for that category
                category_score_sums[category] += similarity * count

    # Find the category with the highest score
    best_category = max(category_score_sums, key=category_score_sums.get, default='No common category found')
    return best_category


def Word_Model(text,category_dicts, model):
    category_list = [item for sublist in category_dicts.values() for item in sublist]#[:-1]
    words = text.lower().split()
    word_counts = defaultdict(int)
    for word in words:
        word_counts[word] += 1

    text_avg_vector = average_vector_representation(words,model)

    # Calculate the valid_word_counts outside of the loop
    valid_word_counts = sum(word_counts[word] for word in words if word in model)

    # If none of the words in the text are present in the model's vocabulary
    if valid_word_counts == 0:
        return "No common category found"  # Default value

    category_score_sums = defaultdict(float)

    for category in category_list:
        category_words = category.lower().split()  # Split multi-word category names
        category_avg_vector = average_vector_representation(category_words,model)

        if category_avg_vector is not None:
            # Calculate similarity using cosine similarity
            similarity = cosine_similarity(text_avg_vector, category_avg_vector)

            # Multiply similarity with valid_word_counts
            weighted_similarity = similarity * valid_word_counts
            category_score_sums[category] += weighted_similarity

    # Find the category with the highest score
    best_category = max(category_score_sums, key=category_score_sums.get, default=99)

    return best_category

def get_embedding(text, model_name, client):
    """Generate embeddings for a single text."""
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model_name)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error in generating embedding: {e}")
        return None

def cosine_similarity(a, b):
    if np.any(a) and np.any(b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 0

def QZERO_Open_AI(df, model_name, category_dicts, tokenize=True, max_tokens=500, batch_size=100):
    OPENAI_API_KEY = ''  # Consider securely fetching this
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    if model_name == 'gpt_small':
        model_name = "text-embedding-3-small"
    else:
        model_name = "text-embedding-3-large"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2") if tokenize else None
    if tokenizer:
        tokenizer.pad_token = tokenizer.eos_token

    def truncate_texts_to_gpt2_tokens(texts):
        if tokenizer:
            encoded_inputs = tokenizer(texts, truncation=True, max_length=max_tokens, padding=True, return_tensors='pt')
            return [tokenizer.decode(enc_input, skip_special_tokens=True) for enc_input in encoded_inputs["input_ids"]]
        return texts

    processed_texts = [truncate_texts_to_gpt2_tokens([text]) for text in tqdm(df['sentence'])]
    df['processed_text'] = [text[0] for text in processed_texts]  # Flatten list

    embeddings = [get_embedding(text, model_name, client) for text in tqdm(df['processed_text'])]
    df['new_embedding'] = embeddings

    label_embeddings = {key: get_embedding(' '.join(labels), model_name, client) for key, labels in category_dicts.items()}

    def classify_row(row_embedding):
        if row_embedding is None:
            return None
        similarities = {key: cosine_similarity(row_embedding, label_embedding) for key, label_embedding in label_embeddings.items()}
        return max(similarities, key=similarities.get)

    df['Predicted'] = df['new_embedding'].apply(classify_row)

    return df

def Base_Open_AI(df, model_name, category_dicts, max_tokens=500, batch_size=100):
    OPENAI_API_KEY = ''  # Consider securely fetching this
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    if model_name == 'gpt_small':
        model_name = "text-embedding-3-small"
    else:
        model_name = "text-embedding-3-large"

   
    embeddings = [get_embedding(text, model_name, client) for text in tqdm(df['tokenized'])]
    df['new_embedding'] = embeddings

    label_embeddings = {key: get_embedding(' '.join(labels), model_name, client) for key, labels in category_dicts.items()}

    def classify_row(row_embedding):
        if row_embedding is None:
            return None
        similarities = {key: cosine_similarity(row_embedding, label_embedding) for key, label_embedding in label_embeddings.items()}
        return max(similarities, key=similarities.get)

    df['Predicted'] = df['new_embedding'].apply(classify_row)

    return df


def evaluate_Base_model(model, model_type, merged_df, category_dicts):
    print("Evaluating model:", model_type)
    if model_type == "sbert":
        print("Using SBERT Model")
        all_values = [item for sublist in category_dicts.values() for item in sublist]
        all_values = [item.lower() for item in all_values]
        descriptions = [' '.join(value) for key, value in category_dicts.items()]

        merged_df['Predicted'] = merged_df['tokenized'].apply(lambda x: SBERT_Model(x, descriptions, model))
        
    elif model_type in ("word2vec", "glove", "fasttext"):
        print("Using word embedding model")
        merged_df['most_common_category'] = merged_df['tokenized'].apply(lambda x: Word_Model(x, category_dicts, model))
        # Create a reversed dictionary to map words to numbers
        reversed_dict = {word: key for key, word_list in category_dicts.items() for word in word_list}
        merged_df['Predicted'] = merged_df['most_common_category'].apply(lambda x: reversed_dict.get(x, x))
        merged_df['Predicted'] = merged_df['Predicted'].replace('No common category found', 99)
        
    elif model_type in ('gpt_small', 'gpt_large'):
        merged_df = Base_Open_AI(merged_df, model_type, category_dicts)
       
    else:
        print("Model type not recognized")
        return None

    merged_df['results'] = merged_df.apply(lambda row: row['Predicted'] == row['label'], axis=1)
    accuracy = merged_df['results'].mean()

    # Calculate F1 scores
    predicted_classes = merged_df.Predicted.values
    ground_truth_classes = merged_df.label.values
    micro_f1 = f1_score(ground_truth_classes, predicted_classes, average='micro')
    print("Results of Base model without QZero")
    print("Micro-Averaged F1 Score:", micro_f1)
    print(accuracy)
    print("-------------------------")
    return micro_f1
    

def evaluate_QZero(model, model_type, merged_df, category_dicts):
    print("Evaluating model:", model_type)
    if model_type == "sbert":
        print("Using SBERT Model")
        all_values = [item for sublist in category_dicts.values() for item in sublist]
        all_values = [item.lower() for item in all_values]
        descriptions = [' '.join(value) for key, value in category_dicts.items()]

        merged_df['Predicted'] = merged_df['sentence'].apply(lambda x: SBERT_Model(x, descriptions, model))
        
    elif model_type in ("word2vec", "glove", "fasttext"):
        print("Using word embedding model")
        merged_df['top_nouns'] = merged_df['top_nouns'].apply(ast.literal_eval)
        merged_df['most_common_category'] = merged_df['top_nouns'].apply(lambda x: QZero_Word_Model(x, category_dicts, model))
        # Create a reversed dictionary to map words to numbers
        reversed_dict = {word: key for key, word_list in category_dicts.items() for word in word_list}
        merged_df['Predicted'] = merged_df['most_common_category'].apply(lambda x: reversed_dict.get(x, x))
        merged_df['Predicted'] = merged_df['Predicted'].replace('No common category found', 99)
        
    elif model_type in ('gpt_small', 'gpt_large'):
        merged_df = QZERO_Open_AI(merged_df, model_type, category_dicts)
        
    else:
        print("Model type not recognized")
        return None

    merged_df['results'] = merged_df.apply(lambda row: row['Predicted'] == row['label'], axis=1)
    accuracy = merged_df['results'].mean()

    # Calculate F1 scores
    predicted_classes = merged_df.Predicted.values
    ground_truth_classes = merged_df.label.values
    micro_f1 = f1_score(ground_truth_classes, predicted_classes, average='micro')
    print("Results using the QZero Pipeline")
    print("Micro-Averaged F1 Score:", micro_f1)
    print(accuracy)
    print("-------------------------")
    return micro_f1

