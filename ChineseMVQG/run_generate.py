#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
# from _typeshed import NoneType

import os
import json
import pickle
import dotenv
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader

from transformers import OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, T5Config


from tokenizer import VQGT5Tokenizer
from dataloader import Dataset
from model import VLT5

# from model import ImageSentenceEmbeddingNetwork

dotenv.load_dotenv()

RESULT_FOLDER = os.getenv('RESULT_FOLDER')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in
                #   (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    't5_vc': (T5Config, VLT5, VQGT5Tokenizer),
}

def load_and_cache_vqg_examples(args, tokenizer, split):
    dataset = Dataset(
        args,
        tokenizer,
        split=split,
        eval_only=True,
    )
    return dataset

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty

def beam_search_sequence(model, length, context, img_feats, obj_feats, end_token_id, pad_token_id, token_type_ids=None, rationale_type_idx=None, num_beams=1, temperature=1, device=None):
    """ Generate sequences for each example with beam search.
    """
    if device is None:
        device = torch.device("cpu")

    if img_feats is not None:
        img_feats = img_feats.repeat(num_beams, 1, 1, 1)
        obj_feats = obj_feats.repeat(num_beams, 1, 1, 1)

    generated = context

    # generated hypotheses
    generated_hyps = BeamHypotheses(num_beams, length, 1, early_stopping=False)

    # scores for each sentence in the beam
    beam_scores = torch.zeros((1,num_beams), dtype=torch.float, device=device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    with torch.no_grad():
        for cur_len in range(length):

            inputs = {'input_ids': generated }

            if img_feats is not None:
                inputs.update({
                    'img_feats': img_feats,
                    'obj_feats': obj_feats,
                })
            if token_type_ids is not None:
                inputs.update({
                    'token_type_ids': token_type_ids
                })

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            vocab_size = next_token_logits.size(-1)
            scores = F.log_softmax(next_token_logits, dim=-1)  # (num_beams, vocab_size)

            # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (num_beams, vocab_size)
            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            _scores = _scores.view(-1)
            next_scores, next_words = torch.topk(_scores, 2 * num_beams, dim=0, largest=True, sorted=True)

            # next sentence beam content
            next_sent_beam = []

            # next words for this sentence
            for idx, score in zip(next_words, next_scores):

                # get beam and word IDs
                beam_id = idx // vocab_size
                word_id = idx % vocab_size

                # end of sentence, or next word
                if word_id.item() == end_token_id or cur_len + 1 == length:
                    generated_hyps.add(generated[beam_id].clone(), score.item())
                else:
                    next_sent_beam.append((score, word_id, beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break

            # update next beam content
            assert len(next_sent_beam) == 0 if cur_len + 1 == length else num_beams
            if len(next_sent_beam) == 0:
                next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch

            # sanity check / prepare next batch
            beam_scores = beam_scores.new([x[0] for x in next_sent_beam])
            beam_words = generated.new([x[1] for x in next_sent_beam])
            beam_idx = generated.new([x[2] for x in next_sent_beam])

            # re-order batch
            generated = generated[beam_idx, :]
            generated = torch.cat([generated, beam_words.unsqueeze(1)], dim=-1)
            if rationale_type_idx is not None:
                token_type_ids = torch.cat((token_type_ids, rationale_type_idx), dim=1)

    # select the best hypotheses
    tgt_len = generated.new(num_beams)
    best = []

    for i, hypotheses in enumerate(generated_hyps.hyp):
        best_hyp = hypotheses[1]
        tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
        best.append(best_hyp)

    # generate target batch
    decoded = generated.new(num_beams, tgt_len.max().item()).fill_(pad_token_id)
    for i, hypo in enumerate(best):
        decoded[i, : tgt_len[i] - 1] = hypo
        decoded[i, tgt_len[i] - 1] = end_token_id

    return decoded

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        probs = F.softmax(logits, dim=1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)

        _cumsum = sorted_probs.cumsum(1)
        mask = _cumsum < top_p
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], 1)
        sorted_probs = sorted_probs * mask.float()
        sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)

        logits.scatter_(1, sorted_indices, sorted_probs.log())

    return logits

def sample_sequence(model, length, context, end_token, pad_token, 
                    roi_features=None, bbox_coordinates=None, 
                    attention_mask=None, vis_attention_mask=None, text_order_ids=None, image_order_ids=None,
                    region_order_ids = None, token_type_ids=None, rationale_type_idx=None,
                    do_sample=True, num_samples=1, temperature=1, top_k=0, top_p=0.0, device=None, encoder=None, decode_input_ids=None):
    if not do_sample:
        return beam_search_sequence(model, length, context, img_feats, obj_feats, end_token, pad_token, token_type_ids=None, rationale_type_idx=None, num_beams=num_samples, temperature=1, device=device)
    if device is None:
        device = torch.device("cpu")

    inputs = {}
    inputs['roi_features'] = roi_features.repeat(num_samples, 1, 1)
    inputs['bbox_coordinates'] = bbox_coordinates.repeat(num_samples, 1, 1)        
    inputs['text_order_ids'] = text_order_ids.repeat(num_samples, 1)
    inputs['image_order_ids'] = image_order_ids.repeat(num_samples, 1)
    inputs['region_order_ids'] = region_order_ids.repeat(num_samples, 1)
    inputs['attention_mask'] = attention_mask.repeat(num_samples, 1)
    inputs['vis_attention_mask'] = vis_attention_mask.repeat(num_samples, 1)

    inputs['input_ids'] = context
    encoder_outputs = encoder(**inputs, return_dict=True)
    generated = decode_input_ids

    with torch.no_grad():
        for tok_idx in range(length):
            outputs = model(
                decoder_input_ids=generated,
                encoder_outputs=encoder_outputs,
                attention_mask=inputs['attention_mask'],
                vis_attention_mask=inputs['vis_attention_mask']
            )
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            if do_sample:
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            generated = torch.cat((generated, next_token), dim=-1)
            if rationale_type_idx is not None:
                token_type_ids = torch.cat((token_type_ids, rationale_type_idx), dim=1)
    return generated


def main():
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument("--prefix", type=str, default="", help="prefix for the output files")
    parser.add_argument("--model_type", default='gpt2_vc', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='gpt2', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--data_dir", default='./', type=str,
                        help="Directory containing train, val, test files")
    parser.add_argument("--split", default=None, type=str, required=True, choices=['train', 'val', 'test'],
                        help="split to use for generation (val/test)")
    parser.add_argument("--task_token_mode", default="normal", type=str)
    parser.add_argument("--vist_image_num", default=5, type=int)
    parser.add_argument("--n_images", default=5, type=int)

    parser.add_argument("--output_file", type=str, default=RESULT_FOLDER,
                        help="File to generate inferences; otherwise, created in the same directpry in the model")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument('--max_seq_len', type=int, default=-1, help="input sequence length after tokenization.")
    parser.add_argument("--no_image", dest='include_image', action='store_false',
                        help="Do not use image context to generate the inference sentences.")
    parser.add_argument("--no_text", dest='include_text', action='store_false',
                        help="Do not use text event and place to generate the inference sentences.")
    parser.add_argument("--use_all_dets", action='store_true',
                        help="Use all detections.")
    parser.add_argument("--eval_only", type=bool, default=True,
                        help="Whether to run training.")
    # sampling based parameters
    parser.add_argument("--length", type=int, default=20, help='max length of sequence to generate')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--inference_type", default='all', type=str, choices=['all', 'intent', 'need', 'react'],
                        help="inference type to generate")
    parser.add_argument("--do_sample", type=int, default=1)
    parser.add_argument("--num_samples", default=5, type=int, help="No. of samples to obtain.")
    parser.add_argument("--gen_batch_size", default=1, type=int, help="No. of instances per batch (for now, it only supports batch size 1).")

    # misc
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--cache_postfix', type=str, default=None,
                       help="postfix for cache")

    parser.add_argument("--feature_dir", default="./feature", type=str)
    parser.add_argument('--image_enc_model', type=str, default=None, help="Load the image encoding model")              
    parser.add_argument("--cache_dir", default="./", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)")
    parser.add_argument('--postfix', type=str, default='', help="For caching")
    parser.add_argument('--n_max_images', type=int, default=10, help="config.n_max_images")
    parser.add_argument('--output_prefix', type=str, default='')

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    #train_args = torch.load(os.path.join(args.model_name_or_path, 'training_args.bin'))
    train_args = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'))
    config, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    tokenizer.add_special_tokens({
        "additional_special_tokens": tokenizer.special_tokens
    })

    if args.max_seq_len <= 0:
        args.max_seq_len = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.max_seq_len = min(args.max_seq_len, tokenizer.max_len_single_sentence)

    config.n_images = 5

    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)
    model.eval()

    # if args.length < 0 and model.config.max_position_embeddings > 0:
    #     args.length = model.config.max_position_embeddings
    # elif 0 < model.config.max_position_embeddings < args.length:
    #     args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    # elif args.length < 0:
    #     args.length = MAX_LENGTH  # avoid infinite loop
    # args.length = MAX_LENGTH  # avoid infinite loop

    print(args)

    # def _prompt_to_gen(context_orig, img_feats):
    def _prompt_to_gen(
        context_orig, 
        roi_features=None,
        bbox_coordinates=None, 
        vis_attention_mask=None, 
        attention_mask=None, 
        text_order_ids=None, 
        image_order_ids=None,
        region_order_ids=None,
        encoder=None,
        decode_input_ids=None,
        label=None
    ):

        if args.model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raise Exception("Not supported")

        context = context_orig.repeat(args.num_samples, 1)
        text_gen = [{} for _ in range(args.num_samples)]

        # set the start token to signal when to start generation
        end_token = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
            

        context_input = context
        rationale_type_idx = None

        # begin sampling sequence starting from context_input
        out = sample_sequence(
            model=model,
            context=context_input,
            pad_token=pad_token,
            end_token=end_token,
            roi_features=roi_features,
            bbox_coordinates=bbox_coordinates,
            vis_attention_mask=vis_attention_mask,
            attention_mask=attention_mask,
            text_order_ids=text_order_ids,
            image_order_ids=image_order_ids,
            region_order_ids=region_order_ids,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            device=args.device,
            num_samples=args.num_samples,
            encoder=encoder,
            decode_input_ids=decode_input_ids,
        )

        # ensure to end the sequence with end token, and pad the rest of the sequence.
    
        out[:,-1] = end_token
        ending_idx = (out == end_token).nonzero()
        processed = []
        for i in range(ending_idx.size(0)):
            sample_idx = ending_idx[i][0].item()
            if sample_idx not in processed:
                processed.append(sample_idx)
                end_idx = ending_idx[i][1].item()
                if end_idx < out.size(1) - 1:
                    out[sample_idx,end_idx+1:] = pad_token

        end_token = tokenizer.eos_token
        # decode the sequence to text
        text_gen = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True, end_token=end_token) for o in out.tolist()]
        # orig_gen = tokenizer.decode(label.tolist()[0], skip_special_tokens=True, clean_up_tokenization_spaces=True, end_token=end_token)
        # print(orig_gen)
        return text_gen

    # output file to store the generations

    output_name = args.output_file if args.output_file is not None else args.model_name_or_path
    output_file = '{}/{}_{}_sample_{}_num_{}_top_k_{}_top_p_{}_{}.json'.format(
        output_name,
        args.output_prefix,
        args.split,
        args.do_sample,
        args.num_samples,
        args.top_k,
        args.top_p,
        args.prefix
        )
    print(output_file)

    # Get Dataset Loader
    eval_dataset = load_and_cache_vqg_examples(args, tokenizer, args.split)
    # all_records = eval_dataset.records # TODO: replace this line to img ids

    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset),
                                 batch_size=args.gen_batch_size, collate_fn=eval_dataset.collate_fn)

    encoder = model.get_encoder()
    decode_input_ids = torch.ones((args.num_samples, 1), device=model.device, dtype=torch.long)
    decode_input_ids = decode_input_ids * model.config.decoder_start_token_id
        
    results = []
    idx = 0
    context_inputs = set()
    for data_input in tqdm.tqdm(eval_dataloader):
        for k, v in data_input.items():
            if isinstance(v, torch.Tensor):
                data_input[k] = v.to(args.device)
        # Skip if we have processed this image, event, and inference type.
        # context = input_record['img_fn'] + input_record['event'] + input_record['inference_relation']
        # if context not in context_inputs:
        #     context_inputs.add(context)
        # else:
        #     continue

        # Now, generate the inferences and decode using original ids.
        generations = _prompt_to_gen(
            context_orig=data_input['input_ids'], 
            roi_features=data_input['roi_features'],
            bbox_coordinates=data_input['bbox_coordinates'],
            vis_attention_mask=data_input['vis_attention_mask'], 
            attention_mask=data_input['attention_mask'], 
            text_order_ids=data_input['text_order_ids'], 
            image_order_ids=data_input['image_order_ids'],
            region_order_ids=data_input['region_order_ids'],
            encoder=encoder,
            decode_input_ids=decode_input_ids,
            label=data_input['labels']
        )
        # for i in range(len(generations)):
        #     generations[i] = replace_names(generations[i], input_record['name2person'])
        # output_record = {k: input_record[k] for k in output_keys}
        output_record = { 'item_id': data_input['item_id'][0] }
        output_record['generations'] = generations
        results.append(output_record)

        if idx < 30:
            print("Item: {}".format(data_input['item_id'][0]))
            print("Inference Generations: {}".format(generations))
        idx += 1

    json.dump(results, open(output_file,'w'), ensure_ascii=False)
    print('Saved to', output_file)

if __name__ == '__main__':
    main()
