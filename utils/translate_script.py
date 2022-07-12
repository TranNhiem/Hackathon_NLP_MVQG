import json

def get_meta_dict(meta_ds_path):
    with open(meta_ds_path, 'rb') as f_ptr:
        meta_dict = json.load(f_ptr)
    return meta_dict

def wrt_bk2json(meta_dict, save_path=None):
    with open(save_path, 'w+') as f_ptr:
        json.dump(meta_dict, f_ptr)

def qs_from_eng2ch(meta_dict, translator):

    def eng_phase2ch_phase(phase):
        # maynot have preprocessing
        preproc_phase = phase
        return translator(preproc_phase)
    
    # translate all text in json
    for ims_id in meta_dict.keys():
        qs_lst = meta_dict[ims_id]
        for qs_dict in qs_lst:
            # change cnt to effect the shallow obj
            qs_dict['Question'] = eng_phase2ch_phase(qs_dict['Question'], translator)
            qs_dict['Summary'] = eng_phase2ch_phase(qs_dict['Summary'], translator)
            
    return meta_dict


def integrate_test():
    # plz help me change the meta_ds_path
    meta_js_path = "/workspace/data/test.json"
    save_js_path = "/workspace/data/test_ch.json"

    meta_dict = get_meta_dict(meta_js_path)
    translator = lambda x : x + "TESTTEST"

    meta_dict = qs_from_eng2ch(meta_dict, translator)

    wrt_bk2json(meta_dict, save_js_path)
    

if __name__ == "__main__":
    integrate_test()