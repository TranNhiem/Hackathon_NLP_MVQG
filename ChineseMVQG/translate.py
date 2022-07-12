import six
from google.cloud import translate_v2 as translate
import argparse
import json
from google.cloud import storage
from tqdm import tqdm
import time

# input command under home directory
# export GOOGLE_APPLICATION_CREDENTIALS="hackathon-vqg-09bf509af932.json" 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='parse json file')
    parser.add_argument('--input_json', type=str, default="../data/test.json")
    args = parser.parse_args() 
    json_file = args.input_json

    output_json_file = f"{json_file.split('/')[-1].split('.')[0]}_google.json"
    output_json_file = f"../data/{output_json_file}"
    output_dict = {}

    translate_client = translate.Client()

    with open(json_file, 'r') as f:
        data = json.load(f)
        for key, q_sum_list in tqdm(data.items()):
            q_sum_output = []
            for q_sum in q_sum_list:
                q_ch = translate_client.translate(q_sum["Question"], target_language="zh-tw")
                sum_ch = translate_client.translate(q_sum["Summary"], target_language="zh-tw")
                q_sum_output.append({"Question": q_ch["translatedText"], "Summary": sum_ch["translatedText"]})
            output_dict[key] = q_sum_output
            time.sleep(5)

    f = open(output_json_file, 'w', encoding='utf-8')
    json.dump(output_dict, f, indent=4, ensure_ascii=False)
    f.close()