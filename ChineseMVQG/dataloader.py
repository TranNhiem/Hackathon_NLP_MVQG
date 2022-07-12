import json
import random
import dotenv
import logging
import os
import pickle
# import pickle5 as pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

dotenv.load_dotenv()

SWIG_PATH = os.getenv('SWIG_PATH')
QUESTION_PATH = os.getenv('QUESTION_PATH')

def load_img_feat(img_feat_dir):
    img_feat_paths = os.listdir(img_feat_dir)
    img_feat_paths = [f"{img_feat_dir}/{p}" for p in img_feat_paths]
    
    valid_img_id_list, img_feats, objects_feats, verbs, objects, bbox, widths, heights = [], [], [], [], [], [], [], []
    for path in img_feat_paths:
        with open(path, 'rb') as f:
            info = pickle.load(f)
        valid_img_id_list.append(path.split('/')[-1][:-4])
        img_feats.append(info['global_feature'])
        objects_feats.append(info['features'])
        verbs.append(info['verb'])
        objects.append(info['objects'])
        
        temp = []
        for i, box in enumerate(info['bbox']):
            temp.append(list(box) if box is not None else [0, 0, info['width'], info['height']])
        bbox.append(temp)

        widths.append(info['width'])
        heights.append(info['height'])

    return valid_img_id_list, img_feats, objects_feats, verbs, objects, bbox, widths, heights


def pad_label(tokenizer, labels, max_seq_len=512):
    pad_len = max_seq_len - len(labels)
    assert pad_len >= 0, print(len(labels))

    labels += [tokenizer.pad_token] * pad_len

    return labels

def pad_bbox_mask(features, boxes, num_image=1):
    """
        input:
            VIST -> num_image = 5
            VQG -> num_image = 1
        return:
            padded_bbox
            padded_feat
            padded_noun
            padded_verb
            mask
    """
    num_boxes = len(boxes)
    num_max_boxes = 7 * num_image
    assert num_max_boxes >= num_boxes

    d = len(features[0])
    padded_features = np.concatenate((features, np.zeros((num_max_boxes - num_boxes, d))))
    padded_boxes = np.concatenate((boxes, np.zeros((num_max_boxes - num_boxes, 4))))

    padded_mask = np.concatenate((np.ones(num_boxes), np.zeros(num_max_boxes - num_boxes)), axis=0)
    

    return padded_features, padded_boxes, padded_mask

class Dataset():
    """
    input
        question
        situations: verb and nouns predicted by SWiG
    return 
        tokens: padded img+obj+situation+label
        label: ground truth question
        img_feature: 1 x 2048
        obj_features: #obj(pad to 6) x 2048
        token_type_id: shape==shape of tokens
    """
    def __init__(self, 
                args,
                tokenizer,
                split='train',
                max_role=6,
                cache_dir='./fortest',
                eval_only=False
            ):
        self.max_role     = max_role
        self.max_seq_len  = args.max_seq_len
        self.max_img_len  = args.n_images
        self.max_img_feat = self.max_role + 1
        self.tokenizer    = tokenizer
        self.postfix      = args.postfix
        self.split        = split
        self.eval_only    = eval_only

        random.seed(args.seed)

        overwrite_cache = args.overwrite_cache
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        cached_dataset = f"cached_{split}_{self.postfix}"
        cached_dataset = os.path.join(cache_dir, cached_dataset)

        self._load_img_feat() # temporary, need to change test as [train, val, test]

        if not os.path.exists(cached_dataset) or overwrite_cache:
            data = self._load_data()

            # cache questions here
            with open(cached_dataset, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # load from cache
            with open(cached_dataset, 'rb') as f:
                data = pickle.load(f)

        self.tokens, self.text_mask, self.labels, self.image_order_ids, self.region_order_ids, self.text_order_ids, self.image_ids = data['tokens'], data['text_mask'], data['labels'], data['image_order_ids'], data['region_order_ids'], data['text_order_ids'], data['image_ids']

    def _load_img_feat(self):
        valid_img_id_list, img_feats, objects_feats, verbs, objects, bbox, widths, heights = [], [], [], [], [], [], [], []
        for folder in ["train", "val", "test"]:
            img_feat_dir = f"{SWIG_PATH}/global_features/{folder}/vist"
            id_list, img_f, objects_f, v, o, b, w, h = load_img_feat(img_feat_dir)
            valid_img_id_list += id_list
            img_feats += img_f
            objects_feats += objects_f
            verbs += v
            objects += o
            bbox += b
            widths += w
            heights += h

        valid_img_id_dict = dict(zip(valid_img_id_list, range(len(valid_img_id_list))))

        self.image_feat = {
            "valid_img_id_list": valid_img_id_list, 
            "valid_img_id_dict": valid_img_id_dict,
            "img_feats":     img_feats, 
            "objects_feats": objects_feats, 
            "verbs":   verbs, 
            "objects": objects, 
            "bbox":    bbox, 
            "widths":  widths, 
            "heights": heights
        }

    def _load_data(self):

        with open(f'{SWIG_PATH}/SWiG_jsons/imsitu_space.json', 'r') as f:
            mapping = json.load(f)
            noun_mapping = mapping['nouns']
            verb_role_mapping = mapping['verbs']

        vist_path = f"{QUESTION_PATH}/{self.split}.json"
        with open(vist_path, 'r') as f:
            corpus = json.load(f)

        tokens, labels, text_order_ids, image_order_ids, region_order_ids, text_mask, image_list = [], [], [], [], [], [], []

        task_token = self.tokenizer.tokenize(f'generate question:')

        # for i, data in tqdm(enumerate(corpus)):
        for k, v in tqdm(corpus.items()):
            text_tokens, _image_order_ids, _text_order_ids, _region_order_ids = [], [], [], []

            img_ids = k.split("_")
            # assert len(img_ids) == 5

            for i, img_id in enumerate(img_ids): # 5 imgs in one story
                # construct image token 
                img_index = self.image_feat["valid_img_id_dict"][img_id]
                verb      = self.image_feat["verbs"][img_index]
                roles     = verb_role_mapping[verb]['order']

                noun_id_list = self.image_feat["objects"][img_index]
                nouns = []
                for noun_id in noun_id_list:
                    nouns.append(noun_mapping[noun_id]['gloss'][0] if noun_id != '' and noun_id != 'Pad' and noun_id != 'oov' else '')

                sub_text_tokens, sub_image_order_ids, sub_text_order_ids, sub_region_order_ids = \
                    self._img_to_token(
                        verb, 
                        nouns, 
                        roles,
                        image_order=i+1,
                        num_feat=len(self.image_feat["objects_feats"][img_index] + 1)
                    ) # <b_img> ... <e_img> <b_verb> v <e_verb> <b_obj> ... <e_obj>

                text_tokens       += sub_text_tokens
                _image_order_ids  += sub_image_order_ids
                _text_order_ids   += sub_text_order_ids
                _region_order_ids += sub_region_order_ids
                    
            text_tokens = task_token + text_tokens + [self.tokenizer.eos_token]
            _text_order_ids = [0] * len(task_token) + _text_order_ids + [0]
            padded_tokens, padded_text_order_ids, padded_mask = self._pad_text_token(text_tokens, _text_order_ids, self.max_seq_len)
            padded_image_order_ids, padded_region_order_ids = self._pad_image_token(_image_order_ids, _region_order_ids, self.max_img_len * self.max_img_feat)
            padded_tokens = self.tokenizer.convert_tokens_to_ids(padded_tokens)

            assert len(padded_tokens)          == self.max_seq_len
            assert len(padded_text_order_ids)  == self.max_seq_len
            assert len(padded_image_order_ids) == self.max_img_len * self.max_img_feat
            assert len(padded_region_order_ids) == self.max_img_len * self.max_img_feat

            if self.eval_only:
                v = v[:1]
                
            _tokens, _text_order_ids, _image_order_ids, _region_order_ids, _text_mask, _labels = [], [], [], [], [], []
            for question in v:
                label_tokens = self._label_to_token(question["Question"])
            
                _tokens.append(padded_tokens)
                _text_order_ids.append(padded_text_order_ids)
                _image_order_ids.append(padded_image_order_ids)
                _region_order_ids.append(padded_region_order_ids)
                _text_mask.append(padded_mask)
            
                label = pad_label(self.tokenizer, label_tokens, max_seq_len=self.max_seq_len)
                label = self.tokenizer.convert_tokens_to_ids(label)
                assert len(label) == self.max_seq_len
                _labels.append(label)

            tokens.append(_tokens)
            text_order_ids.append(_text_order_ids)
            image_order_ids.append(_image_order_ids)
            region_order_ids.append(_region_order_ids)
            text_mask.append(_text_mask)
            labels.append(_labels)
            image_list.append(img_ids)
            
        return {
            'tokens':    tokens,
            'text_mask': text_mask,
            'labels':    labels,
            'text_order_ids':   text_order_ids,
            'image_order_ids':  image_order_ids,
            'region_order_ids': region_order_ids,
            'image_ids': image_list,
        }

    def _img_to_token(self, verb, nouns, roles, image_order=1, num_feat=7):
        text_instance = [self.tokenizer.begin_verb] + self.tokenizer.tokenize(verb) + [self.tokenizer.end_verb]
        
        # <role> noun <role>
        for i, noun in enumerate(nouns):
            if noun != '':
                tag = self.tokenizer.situ_role_tokens[roles[i]]
                text_instance += [tag[0]] + self.tokenizer.tokenize(noun) + [tag[1]]

        text_order_ids   = [image_order] * len(text_instance)
        image_order_ids  = [image_order] * num_feat
        region_order_ids = list(range(1, num_feat + 1))

        return text_instance, image_order_ids, text_order_ids, region_order_ids

    def _label_to_token(self, target):
        return self.tokenizer.tokenize(target) + [self.tokenizer.eos_token]

    def _pad_text_token(self, tokens, order_ids, max_seq_len):
        pad_len = max_seq_len - len(tokens)
        assert pad_len >= 0, print(len(tokens))

        mask = [1] * len(tokens) + [0] * pad_len

        tokens    += [self.tokenizer.pad_token] * pad_len
        order_ids += [0] * pad_len
        
        return tokens, order_ids, mask
        
    def _pad_image_token(self, _image_ids, _region_ids, max_seq_len):
        pad_len = max_seq_len - len(_image_ids)
        assert pad_len >= 0, print(len(_image_ids))

        _image_ids  += [0] * pad_len
        _region_ids += [0] * pad_len

        return _image_ids, _region_ids

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index):
        labels    = torch.tensor(self.labels[index])
        tokens    = torch.tensor(self.tokens[index]) 
        text_mask = torch.tensor(self.text_mask[index]).long()

        image_ids = self.image_ids[index]

        features, boxes = [], []
        for image_id in image_ids:
            image_index = self.image_feat["valid_img_id_dict"][image_id]

            img_feature = np.expand_dims(self.image_feat["img_feats"][image_index], 0)
            filtered_features = self.image_feat["objects_feats"][image_index]
            filtered_features = np.row_stack((img_feature, filtered_features))
            features.append(filtered_features)
            
            w = self.image_feat["widths"][image_index]
            h = self.image_feat["heights"][image_index]
            filtered_bbox = np.row_stack((np.array([0, 0, w, h]), self.image_feat["bbox"][image_index]))

            boxes.append(filtered_bbox)

        padded_features, padded_boxes, padded_mask = \
            pad_bbox_mask(
                features=np.concatenate(features), 
                boxes=np.concatenate(boxes), 
                num_image=5
            )
        
        text_order_ids   = torch.tensor(self.text_order_ids[index]).long()
        image_order_ids  = torch.tensor(self.image_order_ids[index]).long()
        region_order_ids = torch.tensor(self.region_order_ids[index]).long()

        # print(len(self.labels[index]))
        # print(torch.tensor(padded_features).shape)
        
        return_dict = {
            'roi_features':       torch.tensor(padded_features).repeat(len(self.labels[index]), 1, 1).float(),
            'bbox_coordinates':   torch.tensor(padded_boxes).repeat(len(self.labels[index]), 1, 1).float(),
            'vis_attention_mask': torch.tensor(padded_mask).repeat(len(self.labels[index]), 1).long(),
            'text_order_ids':   text_order_ids,
            'image_order_ids':  image_order_ids,
            'region_order_ids': region_order_ids,
            'input_ids':      tokens,
            'attention_mask': text_mask,
            'labels': labels,
        }

        # for k, v in return_dict.items():
        #     print(k, v.shape)

        if self.eval_only:
            return_dict['item_id'] = '_'.join(image_ids)

        return return_dict

    def collate_fn(self, data):
        return_dict = {
            'roi_features':       torch.cat([d["roi_features"] for d in data]),
            'bbox_coordinates':   torch.cat([d["bbox_coordinates"] for d in data]),
            'vis_attention_mask': torch.cat([d["vis_attention_mask"] for d in data]),
            'text_order_ids':   torch.cat([d["text_order_ids"] for d in data]),
            'image_order_ids':  torch.cat([d["image_order_ids"] for d in data]),
            'region_order_ids': torch.cat([d["region_order_ids"] for d in data]),
            'input_ids':      torch.cat([d["input_ids"] for d in data]),
            'attention_mask': torch.cat([d["attention_mask"] for d in data]),
            'labels': torch.cat([d["labels"] for d in data]),
        }

        if 'item_id' in data[0]:
            return_dict['item_id'] = []
            for d in data:
                for i in range(d['input_ids'].shape[0]):
                    return_dict['item_id'].append(d['item_id'])

        return return_dict