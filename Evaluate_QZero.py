################### MAIN ##########
import argparse
import pandas as pd
from tqdm import tqdm
import Models

def main():
    parser = argparse.ArgumentParser(description='Process text classification with various models and datasets.')
    parser.add_argument('--model_type', type=str, default='sbert', 
                        choices=['glove', 'word2vec', 'fasttext', 'sbert', 'gpt-small', 'gpt_large'], 
                        help='Select the type of the model to use')
    parser.add_argument('--data_file', type=str, default='QZERO_PAPER/retrieved_results/results_bm25.csv', 
                        help='File path for the dataset to evaluate')
    parser.add_argument('--category_dict', type=str, default='ag_classes',
                        choices=['tag_classes', 'ag_classes', 'dbpedia_classes', 'yahoo_classes', 'cooking_classes', 'ohsumed_classes'],
                        help='Choose the category dictionary to use for mapping classes')

    args = parser.parse_args()

    # Category dictionaries
    category_dicts = {
        'tag_classes': {0: ['sport'], 1: ['business'], 2: ['entertainment'], 3: ['America'], 4: ['politics', 'government'], 5: ['health'], 6: ['science', 'technology']},
        'ag_classes': {0: ['politics', 'government'], 1: ['sports'], 2: ['business', 'finance'], 3: ['technology']},
        'dbpedia_classes': {0: ["companies"], 1: ["schools", "university"], 2: ["artists"], 3: ["athletes"], 4: ["politics"], 5: ["transportation"], 6: ["buildings", "structures"], 7: ["mountains", "rivers", "lakes", "landforms"], 8: ["villages"], 9: ["animals"], 10: ["plants", 'tree'], 11: ["albums"], 12: ["films"], 13: ["literature", "publication", "books", "novels"]},
        'yahoo_classes': {0: ["society", "culture"], 1: ["science", "mathematics"], 2: ["health"], 3: ["education", "reference"], 4: ["internet", "computers"], 5: ["sports"], 6: ["business", "finance"], 7: ["entertainment"], 8: ["family", "relationships"], 9: ["politics", "government"]},
        'cooking_classes': {0: ["cajun", "creole"], 1: ["jamaican"], 2: ["chinese"], 3: ["french"], 4: ["vietnamese"], 5: ["filipino"], 6: ["irish"], 7: ["thai"], 8: ["indian"], 9: ["southern", "united", "states"], 10: ["moroccan"], 11: ["greek"], 12: ["italian"], 13: ["japanese"], 14: ["mexican"], 15: ["korean"], 16: ["russian"], 17: ["spanish"], 18: ["british"], 19: ["brazilian"]},
        'ohsumed_classes': {0: ['bacterial infections'], 1: ['virus diseases'], 2: ['parasitic diseases'], 3: ['neoplasms'], 4: ['musculoskeletal diseases'], 5: ['digestive system diseases'], 6: ['stomatognathic diseases'], 7: ['respiratory tract diseases'], 8: ['otorhinolaryngologic diseases'], 9: ['nervous system diseases'], 10: ['eye diseases'], 11: ['urologic male genital diseases'], 12: ['female genital diseases and pregnancy complications'], 13: ['cardiovascular diseases'], 14: ['hemic and lymphatic diseases'], 15: ['neonatal diseases'], 16: ['skin and connective tissue diseases'], 17: ['nutritional and metabolic diseases'], 18: ['endocrine diseases'], 19: ['immunologic diseases'], 20: ['environmental disorders'], 21: ['animal diseases'], 22: ['pathological conditions']}
    }

    # Load data and model
    df = pd.read_csv(args.data_file)
    model = Models.load_models(args.model_type)
    selected_category_dict = category_dicts[args.category_dict]

    # Evaluate model
    result1 = Models.evaluate_QZero(model, args.model_type, df, selected_category_dict)
    result2 = Models.evaluate_Base_model(model, args.model_type, df, selected_category_dict)

    # Output results
    print("The accuracy with QZero is : ", result1)
    print("The accuracy without QZero is : ", result2)

if __name__ == "__main__":
    main()