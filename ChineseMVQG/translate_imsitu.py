import argparse
import json
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import dotenv
from google.cloud import translate_v2 as translate
from google.cloud import storage
import time

if __name__ == "__main__":
    dotenv.load_dotenv()
    SWIG_PATH = os.getenv('SWIG_PATH')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'hackathon-vqg-09bf509af932.json'
    # parser = argparse.ArgumentParser(description='parse json file')
    # args = parser.parse_args() 
    input_file = f'{SWIG_PATH}/SWiG_jsons/imsitu_space.json'
    translate_client = translate.Client()

    output_file = f"{input_file.split('/')[-1].split('.')[0]}_ch.json"
    output_file = f"../data/{output_file}"

    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    # model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

    with open(input_file, 'r') as f:
        mapping = json.load(f)

    ######### Translate nouns['gloss'] and verbs.key() and verbs.key()['order']

    verb_list = [k for k in mapping['verbs'].keys()]
    order_len_list = [len(v['order']) for v in mapping['verbs'].values()]
    order_text_list = []
    for v in mapping['verbs'].values():
        order_text_list += v['order']
    noun_list = [k for k in mapping['nouns'].keys()]
    noun_len_list = [len(v['gloss']) for v in mapping['nouns'].values()]
    noun_text_list = []
    for v in mapping['nouns'].values():
        noun_text_list += v['gloss']

    MAX_LEN = 100

    verb_ch = []
    for i in range(len(verb_list) // MAX_LEN + 1):
        translation = translate_client.translate(verb_list[i*MAX_LEN:(i+1)*MAX_LEN], target_language="zh-tw")
        verb_ch += [t['translatedText'] for t in translation]

    order_text_ch = []
    for i in range(len(order_text_list) // MAX_LEN + 1):
        translation = translate_client.translate(order_text_list[i*MAX_LEN:(i+1)*MAX_LEN], target_language="zh-tw")
        order_text_ch += [t['translatedText'] for t in translation]
    
    noun_text_ch = []
    for i in range(len(noun_text_list) // MAX_LEN + 1):
        translation = translate_client.translate(noun_text_list[i*MAX_LEN:(i+1)*MAX_LEN], target_language="zh-tw")
        noun_text_ch += [t['translatedText'] for t in translation]

    total = 0
    for v, c, l in zip(verb_list, verb_ch, order_len_list):
        mapping['verbs'][v]['chinese'] = c
        mapping['verbs'][v]['order_ch'] = order_text_ch[total:total+l]
        total += l

    total = 0
    for n, l in zip(noun_list, noun_len_list):
        mapping['nouns'][n]['gloss_ch'] = noun_text_ch[total:total+l]
        total += l


    f = open(output_file, 'w', encoding='utf-8')
    json.dump(mapping, f, indent=4, ensure_ascii=False)
    f.close()