import argparse
import json
from tqdm import tqdm
# import transformers
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import dotenv
import pickle
import time

###### Replace from imsitu file
if __name__ == "__main__":
    dotenv.load_dotenv()
    SWIG_PATH = os.getenv('SWIG_PATH')
    parser = argparse.ArgumentParser(description='parse json file')
    # parser.add_argument('--input_file', type=str, default="/home/VIST/data/swig/global_features/test/vist/156087858.pkl")
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--output_dir', type=str, default="../data/GSR")
    args = parser.parse_args() 

    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    # model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    
    ################################

    mapping = json.load(open('../data/imsitu_space_ch.json', 'r'))

    img_feat_dir = f"{SWIG_PATH}/global_features/{args.split}/vist"
    img_feat_paths = os.listdir(img_feat_dir)
    img_feat_paths = [f"{img_feat_dir}/{p}" for p in img_feat_paths]

    for img_feat_path in tqdm(img_feat_paths):
        
        info = pickle.load(open(img_feat_path, 'rb'))
        info['verb'] = mapping['verbs'][info['verb']]['chinese']

        output_pickle_file = f"{img_feat_path.split('/')[-1].split('.')[0]}.pkl"
        output_pickle_file = f"{args.output_dir}/{args.split}/{output_pickle_file}"

        with open(output_pickle_file, 'wb') as f:
            pickle.dump(info, f)
        