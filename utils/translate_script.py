import json
from tqdm import tqdm

# 1. deep_translator :
# pip install deep_translator
# doc:https://deep-translator.readthedocs.io/en/latest/index.html
from deep_translator import GoogleTranslator
# 2. ctranslate2
# pip install ctranslate2 OpenNMT-py sentencepiece
# doc:https://opennmt.net/CTranslate2/quickstart.html
# wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz (wget en2zh model) & tar it tar xf transformer-ende-wmt-pyOnmt.tar.gz
# ct2-opennmt-py-converter --model_path averaged-10-epoch.pt --output_dir ende_ctranslate2 (convert2ct-format)
#import ctranslate2
#import sentencepiece as spm

def get_meta_dict(meta_ds_path):
    with open(meta_ds_path, 'rb') as f_ptr:
        meta_dict = json.load(f_ptr)
    return meta_dict

def wrt_bk2json(meta_dict, save_path=None):
    with open(save_path, 'w+') as f_ptr:
        json.dump(meta_dict, f_ptr)


def qs_from_eng2ch(meta_dict, translator):
    
    for ims_id in tqdm(meta_dict.keys()):
        qs_lst = meta_dict[ims_id]
        q_lst = []
        s_lst = []  
        for qs_dict in qs_lst:
            q_lst.append(qs_dict['Question'])
            s_lst.append(qs_dict['Summary'])
            
        qc_lst = translator.translate_batch(q_lst)
        sc_lst = translator.translate_batch(s_lst)
        for idx, qs_dict in enumerate(qs_lst):
            qs_dict['Question'] = qc_lst[idx]
            qs_dict['Summary'] = sc_lst[idx]
    breakpoint()
    return meta_dict


def meta_js_translation():
    # plz help me change the meta_ds_path
    meta_js_path = "/workspace/data/test.json"
    save_js_path = "/workspace/data/test_ch.json"

    meta_dict = get_meta_dict(meta_js_path)

    #translator = ctranslate2.Translator("ende_ctranslate2/", device="gpu")
    #sp = spm.SentencePieceProcessor("sentencepiece.model")
    #input_tokens = sp.encode(input_text, out_type=str)
    #results = translator.translate_batch([input_tokens])
    #output_tokens = results[0].hypotheses[0]
    #output_text = sp.decode(output_tokens)
    #print(output_text)

    translator = GoogleTranslator(source='en', target='zh-TW')
    
    meta_dict = qs_from_eng2ch(meta_dict, translator)

    wrt_bk2json(meta_dict, save_js_path)
    


if __name__ == "__main__":
    meta_js_translation()