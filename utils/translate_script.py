import json
from tqdm import tqdm

# 1. deep_translator :
# pip install deep_translator
# doc:https://deep-translator.readthedocs.io/en/latest/index.html
from deep_translator import GoogleTranslator
# 2. dl-translate
# pip install dl-translate
# doc:https://github.com/xhluca/dl-translate
#import dl_translate as dlt

def get_meta_dict(meta_ds_path):
    with open(meta_ds_path, 'rb') as f_ptr:
        meta_dict = json.load(f_ptr)
    return meta_dict

def wrt_bk2json(meta_dict, save_path=None):
    with open(save_path, 'w+') as f_ptr:
        json.dump(meta_dict, f_ptr)


def qs_from_eng2ch(meta_dict, translator):
    def eng2zh(sentences):
        #translator.translate(sentences, source='en', target='zh')
        sentences = translator.translate_batch(sentences)
        return sentences

    for ims_id in tqdm(meta_dict.keys()):
        qs_lst = meta_dict[ims_id]
        q_lst = []
        s_lst = []  
        for qs_dict in qs_lst:
            q_lst.append(qs_dict['Question'])
            s_lst.append(qs_dict['Summary'])
            
        qc_lst, sc_lst = eng2zh(q_lst), eng2zh(s_lst)
        for idx, qs_dict in enumerate(qs_lst):
            qs_dict['Question'] = qc_lst[idx]
            qs_dict['Summary'] = sc_lst[idx]

    breakpoint()
    return meta_dict


def meta_js_translation():
    # plz help me change the meta_ds_path
    meta_js_path = "/workspace/data/test.json"
    save_js_path = "/workspace/data/test_zh.json"

    meta_dict = get_meta_dict(meta_js_path)

    translator = GoogleTranslator(source='en', target='zh-TW')
    #translator = dlt.TranslationModel("m2m100", device="gpu")  # default
    
    meta_dict = qs_from_eng2ch(meta_dict, translator)

    wrt_bk2json(meta_dict, save_js_path)

if __name__ == "__main__":
    meta_js_translation()