import argparse
import json
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='parse json file')
    parser.add_argument('--input_json', type=str, default="../data/test.json")
    args = parser.parse_args() 
    json_file = args.input_json

    output_json_file = f"{json_file.split('/')[-1].split('.')[0]}_ch.json"
    output_json_file = f"../data/{output_json_file}"
    output_dict = {}


    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

    with open(json_file, 'r') as f:
        data = json.load(f)
        for key, q_sum_list in tqdm(data.items()):
            q_sum_output = []
            # for q_sum in q_sum_list:
            tokenized_q = tokenizer.prepare_seq2seq_batch([q_sum["Question"] for q_sum in q_sum_list], return_tensors='pt')
            translation_q = model.generate(**tokenized_q)
            q_ch = tokenizer.batch_decode(translation_q, skip_special_tokens=True)[0]
            # print("CHINESE:", q_ch)
    
            tokenized_sum = tokenizer.prepare_seq2seq_batch([q_sum["Summary"] for q_sum in q_sum_list], return_tensors='pt')
            translation_sum  = model.generate(**tokenized_sum)
            sum_ch = tokenizer.batch_decode(translation_sum, skip_special_tokens=True)[0]
            
            for q, s in zip(q_ch, sum_ch):
                q_sum_output.append({"Question": q, "Summary": s})
            output_dict[key] = q_sum_output

    f = open(output_json_file, 'w', encoding='utf-8')
    json.dump(output_dict, f, indent=4, ensure_ascii=False)
    f.close()