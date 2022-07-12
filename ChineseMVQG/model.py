import copy
import warnings
import torch
from torch import nn
from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions, BaseModelOutput, ModelOutput
from transformers import T5PreTrainedModel, T5ForConditionalGeneration
# # from transformers.modeling_bert import BertLayerNorm as LayerNorm
from transformers.models.gpt2.modeling_gpt2 import Block as GPT2_Block
from transformers.models.t5.modeling_t5 import T5Block
from transformers.models.t5.modeling_t5 import T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF
from transformers.models.t5.modeling_t5 import T5LayerNorm

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from dataclasses import dataclass

# from transformers.modeling_bert import BertLayerNorm as LayerNorm
# from transformers.modeling_gpt2 import Block

def assign_visual_embed(input_embeds, visual_embeds, starts, num_feats):
    b = input_embeds.size(0)
    for i in range(b):
        s = 0
        for j, start in enumerate(starts[i]):
            input_embeds[i, start:start+num_feats[i][j], :] = visual_embeds[i, s:s+num_feats[i][j], :]
            s += num_feats[i][j]

    return input_embeds

def pad_sequence(sequence, lengths):
    """
    :param sequence: [\sum b, .....] sequence
    :param lengths: [b1, b2, b3...] that sum to \sum b
    :return: [len(lengths), maxlen(b), .....] tensor
    """
    output = sequence.new_zeros(len(lengths), max(lengths), *sequence.shape[1:])
    start = 0
    for i, diff in enumerate(lengths):
        if diff > 0:
            output[i, :diff] = sequence[start:(start + diff)]
        start += diff
    return output

class SimpleDetector(nn.Module):
    def __init__(self, pretrained=True, use_bbox=True, semantic=False, final_dim=768, layer_norm_epsilon=1e-12):
        """
        :param average_pool: whether or not to average pool the representations
        :param pretrained: Whether we need to load from scratch
        :param semantic: Whether or not we want to introduce the mask and the class label early on (default Yes)
        """
        super(SimpleDetector, self).__init__()

        self.ln_f = nn.LayerNorm(final_dim, eps=layer_norm_epsilon)

        self.semantic = semantic

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2048 + (512 + 512 if semantic else 0), final_dim),
            torch.nn.ReLU(inplace=True),
        )

        self.use_bbox = use_bbox
        if use_bbox:
            self.bbox_upsample = torch.nn.Sequential(
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(4, final_dim),
                torch.nn.ReLU(inplace=True),
            )
    
    def forward(self,
                roi_aligned_feats: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                ):
        """
        max_num_objects = 6 in JSL
        :param roi_aligned_feats: [batch_size, max_num_objects, 2048]
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        
        :return: object reps [batch_size, max_num_objects, dim]
        """

        box_inds = box_mask.nonzero() 
        rois = roi_aligned_feats[box_inds[:, 0], box_inds[:, 1]]

        roi_aligned_feats = self.ln_f(self.obj_downsample(rois))
        # roi_aligned_feats = self.obj_downsample(rois)

        if self.use_bbox:
            bboxes = boxes[box_inds[:, 0], box_inds[:, 1]]
            box_feats = self.ln_f(self.bbox_upsample(bboxes))
            box_feats = self.bbox_upsample(bboxes)
            # roi_aligned_feats = self.ln_f(roi_aligned_feats + box_feats)
            roi_aligned_feats = roi_aligned_feats + box_feats
        

        # if image_order_ids is not None:
        # roi_aligned_feats += image_order_ids
        # # Add some regularization, encouraging the model to keep giving decent enough predictions
        # obj_logits = self.noun_regularizing_predictor(roi_aligned_feats)
        # obj_labels = noun_labels[box_inds[:, 0], box_inds[:, 1]]
        # obj_cnn_regularization = F.cross_entropy(obj_logits, obj_labels, size_average=True)[None]

        # verb_logits = self.verb_regularizing_predictor(roi_aligned_feats)
        # verb_labels = verb_labels[box_inds[:, 0], box_inds[:, 1]]
        # verb_cnn_regularization = F.cross_entropy(verb_logits, verb_labels, size_average=True)[None]

        # Reshape into a padded sequence - this is expensive and annoying but easier to implement and debug...
        obj_reps = pad_sequence(roi_aligned_feats, box_mask.sum(1).tolist())
        return {
            'obj_reps': obj_reps,
            # 'obj_logits': obj_logits,
            # 'obj_labels': obj_labels,
            # 'verb_cnn_regularization_loss': None # verb_cnn_regularization
            # 'obj_cnn_regularization_loss': None # obj_cnn_regularization_loss
        }



class VisualGPT2(GPT2PreTrainedModel):
    def __init__(self, config):
        super(VisualGPT2, self).__init__(config)
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # vocab embedding
        self.wpe = nn.Embedding(config.n_positions, config.n_embd) # position embedding
        self.i_wpe = nn.Embedding(config.n_max_images+1, config.n_embd, padding_idx=0) # image order embedding
        self.i_wpe.weight.data[1:config.n_max_images+1].copy_(self.wpe.weight[1:config.n_max_images+1].data)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2_Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.down_sampler = SimpleDetector(pretrained=True, use_bbox=True, semantic=False, final_dim=config.n_embd)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        """
        Wrapper around PreTrainedModel._get_resized_embeddings
        Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end.
        """
        self.wte = self._get_resized_embeddings(self.wte, new_num_tokens)
        return self.wte

    def forward(self, 
                input_ids,
                position_ids=None,
                token_type_ids=None,
                past_key_values=None,
                head_mask=None,
                attention_mask=None,
                img_feats=None,
                obj_feats=None,
                segments=None,
                inputs_embeds=None,
                roi_features=None,
                bbox_coordinates=None,
                box_masks=None,
                image_order_ids=None,
                num_feat_each_image=None,
                image_emb_locat=None,
                image_order_ids_mask=None
        ):

        output_attentions = self.config.output_attentions
        output_hidden_states = (self.config.output_hidden_states)
        use_cache = self.config.use_cache
        return_dict = self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        visual_embeds = self.down_sampler(roi_features, bbox_coordinates, box_masks)['obj_reps']
        inputs_embeds = assign_visual_embed(inputs_embeds, visual_embeds, image_emb_locat, num_feat_each_image)
        if image_order_ids is not None:
            image_order_ids_embed = self.i_wpe(image_order_ids)

        position_embeds = self.wpe(position_ids)
        if image_order_ids is not None:
            hidden_states = inputs_embeds + position_embeds + image_order_ids_embed
        else:
            hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)


        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_self_attentions, )
        # if img_feats is not None:
        #     outputs = outputs + (vision_obj_loss, )
        return outputs  # last hidden state, presents, (all hidden_states), (attentions)

class VisualGPT2LMHead(GPT2PreTrainedModel):
    def __init__(self, config):
        super(VisualGPT2LMHead, self).__init__(config)
        self.transformer = VisualGPT2(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte)

    def forward(self,
                input_ids,
                position_ids=None,
                token_type_ids=None,
                past_key_values=None,
                head_mask=None,
                img_feats=None,
                obj_feats=None,
                segments=None,
                get_rationale=False,
                labels=None,
                attention_mask=None,
                roi_features=None,
                bbox_coordinates=None,
                box_masks=None,
                image_order_ids=None,
                num_feat_each_image=None,
                image_emb_locat=None,
                image_order_ids_mask=None
        ):
        transformer_outputs = self.transformer(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            img_feats=img_feats,
            obj_feats=obj_feats,
            segments=segments,
            attention_mask=attention_mask,
            roi_features=roi_features,
            bbox_coordinates=bbox_coordinates,
            box_masks=box_masks,
            image_order_ids=image_order_ids,
            num_feat_each_image=num_feat_each_image,
            image_emb_locat=image_emb_locat,
            image_order_ids_mask=image_order_ids_mask
        )
        hidden_states = transformer_outputs[0] # last hidden state
        if get_rationale:
            return hidden_states[:,-1]
        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,) + transformer_outputs[1:] # presents, (all hidden_states), (attentions)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction = "sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

            num = shift_labels.ne(-1).long().sum().item()
            loss /= num
            
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions), (vision_loss)

    def _resize_token_embeddings(self, new_num_tokens):
        self.transformer.resize_token_embeddings(new_num_tokens)


##################################################
## T5 Model with Adapter
##################################################

def freeze_model_parameters(model, exception_keyword="adapter"):
    """
    Freeze parameters of the layers that do not contain exception_keyword in their layer names.
    """
    for name, param in model.named_parameters():
        if exception_keyword not in name and "layer_norm" not in name:
            param.requires_grad = False


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation functions.
    Modified from https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapters/modeling.py
    """
    def __init__(self, hidden_act):
        super().__init__()

        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":
            def swish(x):
                return x * torch.sigmoid(x)
            
            self.f = swish
        elif hidden_act.lower() == "gelu":
            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                import math
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu

    def forward(self, x):
        return self.f(x)


class Adapter(nn.Module):
    """
    Implementation of a single Adapter block.
    Modified from https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapters/modeling.py
    """
    def __init__(
        self,
        input_size,
        down_sample=None,
        non_linearity="relu",
        add_layer_norm_before=True,
        add_layer_norm_after=False,
        residual_before_ln=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual_before_ln = residual_before_ln

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2

        # Linear down projection of the input
        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        
    def forward(self, x, residual_input):
        down = self.adapter_down(x)

        up = self.adapter_up(down)

        output = up

        # apply residual connection before layer norm if configured in this way
        if self.residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # if residual should be applied after layer norm, apply it here
        if not self.residual_before_ln:
            output = output + residual_input

        return output, down, up


class T5_Block(nn.Module):
    """
    Implementation of a T5 block with adapter layer.
    Modified from https://github.com/huggingface/transformers/blob/v4.5.0-release/src/transformers/models/t5/modeling_t5.py
    """
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

        # include adapter module
        self.layer.append(
            Adapter(
                input_size=config.d_model, 
                non_linearity="gelu", 
                add_layer_norm_before=False, 
                add_layer_norm_after=True, 
                residual_before_ln=False
            )
        )

        self.vqg_adapter = Adapter(
                input_size=config.d_model, 
                non_linearity="gelu", 
                add_layer_norm_before=False, 
                add_layer_norm_after=True, 
                residual_before_ln=False
            )

        self.skip_adapter_layer = config.skip_adapter_layer
        self.skip_vqg_adapter_layer = config.skip_vqg_adapter_layer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        encoder_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention' if expected_num_past_key_values == 4 else ''}."
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-2](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        if not self.skip_adapter_layer:
            # Apply adapter layer
            hidden_states = self.layer[-1](hidden_states, hidden_states)[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        if not self.skip_vqg_adapter_layer:
            # Apply adapter layer
            hidden_states = self.vqg_adapter(hidden_states, hidden_states)[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class VisualT5Stack(T5PreTrainedModel):
    """
    Modified from https://github.com/huggingface/transformers/blob/v4.5.0-release/src/transformers/models/t5/modeling_t5.py
    """
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        if config.use_adapter:
            self.block = nn.ModuleList(
                [T5_Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
            )
        else:
            self.block = nn.ModuleList(
                [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
            )
        
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.down_sampler = SimpleDetector(pretrained=True, use_bbox=True, semantic=False, final_dim=config.d_model)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                inputs_embeds=None,
                head_mask=None,
                encoder_head_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                roi_features=None,
                bbox_coordinates=None,
                box_masks=None,
                num_feat_each_image=None,
                image_emb_locat=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        if not self.is_decoder:
            visual_embeds = self.down_sampler(roi_features, bbox_coordinates, box_masks)['obj_reps']
            inputs_embeds = assign_visual_embed(inputs_embeds, visual_embeds, image_emb_locat, num_feat_each_image)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        encoder_head_mask = self.get_head_mask(encoder_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            encoder_layer_head_mask = encoder_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                encoder_layer_head_mask=encoder_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

class VisualEmbedding(nn.Module):
    """
    Modified from https://github.com/j-min/VL-T5/blob/main/VL-T5/src/modeling_t5.py
    """
    def __init__(self, config, obj_order_embedding, img_order_embedding):
        super().__init__()
        self.config = config
        feat_dim = config.feat_dim
        pos_dim = config.pos_dim
        # n_objs = config.n_objs

        if self.config.individual_vis_layer_norm:

            # Object feature encoding
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            if self.config.use_vis_layer_norm:
                feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.feat_embedding = nn.Sequential(*feat_embedding)

            # self.relative_vis_pos_embedding = nn.Linear(pos_dim + 1, config.num_heads)
            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            if self.config.use_vis_layer_norm:
                absolute_vis_pos_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
            # self.absolute_vis_pos_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

            if self.config.use_vis_order_embedding:
                # self.obj_order_embedding = nn.Embedding(n_objs, config.d_model)
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = img_order_embedding

        else:
            # Object feature encoding
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            # if self.config.use_vis_layer_norm:
            #     feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.feat_embedding = nn.Sequential(*feat_embedding)

            # self.relative_vis_pos_embedding = nn.Linear(pos_dim + 1, config.num_heads)
            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            # if self.config.use_vis_layer_norm:
            #     absolute_vis_pos_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
            # self.absolute_vis_pos_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

            if self.config.use_vis_order_embedding:
                # self.obj_order_embedding = nn.Embedding(n_objs, config.d_model)
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = img_order_embedding

            if self.config.use_vis_layer_norm:
                self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        # [B, N]
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area


    def forward(self, feats, pos, img_order_ids=None, obj_order_ids=None):
        """
        Args
            feats: [B, N, feat_dim]
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            relative_vis_pos_embedding: [B, N, N, n_heads]
            absolute_vis_pos_embedding: # [B, N, d_model]
        """

        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)

        feat_embedding = self.feat_embedding(feats)

        device = feats.device
        dtype = feats.dtype

        area = self.get_area(pos).unsqueeze(2) # [B, N, 1]
        pos = torch.cat([pos, area], dim=2) # [B, N, 5]

        # [B, N, d_model]
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)
        # absolute_vis_pos_embedding = self.absolute_vis_pos_layer_norm(absolute_vis_pos_embedding)

        if self.config.use_vis_order_embedding:
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0) #.expand(B, -1)
            img_order_embedding = self.img_order_embedding(img_order_ids)

            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0) #.expand(B,-1)
            # assert obj_order_ids.max().item() < 32200, obj_order_ids
            obj_order_ids = self.obj_order_embedding.num_embeddings - obj_order_ids - 1
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)

            vis_embedding = feat_embedding + absolute_vis_pos_embedding + \
                img_order_embedding + obj_order_embedding

        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding

        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)

        return vis_embedding

class JointEncoder(T5PreTrainedModel):
    """
    Modified from https://github.com/j-min/VL-T5/blob/main/VL-T5/src/modeling_t5.py
    """
    def __init__(self, config, embed_tokens=None):
        super(T5PreTrainedModel, self).__init__(config)
        self.config = config

        self.embed_tokens = embed_tokens
        self.is_decoder = self.config.is_decoder
        assert self.config.is_decoder is False

        self.shared_order_embedding = nn.Embedding(config.n_images + 1, config.d_model)

        self.visual_embedding = VisualEmbedding(self.config, embed_tokens, self.shared_order_embedding)

        if config.use_adapter:
            self.block = nn.ModuleList(
                [T5_Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
            )
        else:
            self.block = nn.ModuleList(
                [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
            )

        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        self.model_parallel = False
        self.device_map = None

    def extend_order_embedding(self, n_images):
        new_shared = nn.Embedding(n_images, self.config.d_model)
        old_weight = self.shared_order_embedding.weight.data.detach().clone()
        old_size = old_weight.size(0)
        new_shared.weight.data[:old_size, :] = old_weight
        self.shared_order_embedding = new_shared
        self.visual_embedding.img_order_embedding = new_shared

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
        self.visual_embedding.obj_order_embedding = new_embeddings

    def forward(self,
                input_ids=None,
                attention_mask=None,

                roi_features=None,
                bbox_coordinates=None,
                text_order_ids=None,
                image_order_ids=None,
                region_order_ids=None,
                vis_attention_mask=None,

                inputs_embeds=None,
                head_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
    ):
        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids) + self.shared_order_embedding(text_order_ids)

        B, L = inputs_embeds.size()[:-1]

        vis_embeds = self.visual_embedding(roi_features, bbox_coordinates, image_order_ids, region_order_ids)

        V_L = vis_embeds.size(1)

        inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        if vis_attention_mask is None:
            vis_attention_mask = attention_mask.new_ones(B, V_L)

        attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            (B, L + V_L),
            inputs_embeds.device)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        # position_bias = None
        # encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        if self.config.num_layers > 0:

            assert self.block[0].layer[0].SelfAttention.has_relative_attention_bias

            seq_length = L + V_L
            q_len = seq_length
            k_len = seq_length

            # [1, n_heads, Q_len, K_len]
            text_position_bias = self.block[0].layer[0].SelfAttention.compute_bias(
                L, L)
            num_heads = text_position_bias.size(1)
            position_bias = text_position_bias.new_zeros(
                1, num_heads, seq_length, seq_length)
            position_bias[:, :, :L, :L] = text_position_bias

            # print('position_bias size', position_bias.size())
            # print('attention_mask size', attention_mask.size())
            # print('extended_attention_mask size', extended_attention_mask.size())
            # relative position bias only between Text <-> Text
            # no relative position bias Text -> Vision
            # no relative position bias Vision -> Text
            # no relative position bias Vision <-> Vision
            # position_bias[:, :, L:, :] = 0
            # position_bias[:, :, :, L:] = 0
            position_bias = position_bias + extended_attention_mask

            for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):

                # if output_hidden_states:
                #     all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    encoder_decoder_position_bias=None,
                    layer_head_mask=head_mask[i],
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                hidden_states, present_key_value_state = layer_outputs[:2]

                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights),
                # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[2]

                # append next layer key value states
                if use_cache:
                    present_key_value_states = present_key_value_states + \
                        (present_key_value_state,)

                # if output_attentions:
                #     all_attentions = all_attentions + (layer_outputs[3],)
                #     if self.is_decoder:
                #         all_cross_attentions = all_cross_attentions + \
                #             (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

class VLT5(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super(T5ForConditionalGeneration, self).__init__(config)

        self.config = config

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        #---- Modified ----#
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = JointEncoder(encoder_config, self.shared)
        #------------------#

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

        self.decoder = VisualT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def extend_vocab(self, vocab_size):
        new_shared = nn.Embedding(vocab_size, self.config.d_model)
        old_weight = self.shared.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_shared.weight.data[:old_vocab_size, :] = old_weight
        self.shared = new_shared

        new_lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        old_weight = self.lm_head.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_lm_head.weight.data[:old_vocab_size, :] = old_weight
        self.lm_head = new_lm_head

        self.encoder.visual_embedding.obj_order_embedding = self.shared

        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        self.lm_head.weight = self.shared.weight

        self.config.vocab_size = vocab_size
        self.encoder.config.vocab_size = vocab_size
        self.decoder.config.vocab_size = vocab_size


    # @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                roi_features=None,
                bbox_coordinates=None,
                text_order_ids=None,
                image_order_ids=None,
                region_order_ids=None,
                vis_attention_mask=None,

                reduce_loss=True,

                return_hidden_state=False,

                **kwargs,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,

                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,

                roi_features=roi_features,
                bbox_coordinates=bbox_coordinates,
                text_order_ids=text_order_ids,
                image_order_ids=image_order_ids,
                region_order_ids=region_order_ids,
                vis_attention_mask=vis_attention_mask,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=hidden_states.dtype, device=hidden_states.device)
        if vis_attention_mask is None:
            B, L = attention_mask.size()
            V_L = encoder_outputs[0].size(1) - L
            vis_attention_mask = attention_mask.new_ones(B, V_L)
        encoder_attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,

            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print('decoder_outputs')
        # print(decoder_outputs)

        sequence_output = decoder_outputs[0]

        assert self.config.tie_word_embeddings is True

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        if return_hidden_state:
            return sequence_output

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(
            #     lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            if reduce_loss:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1))

            # print('loss')
            # print(loss)

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        return VLSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            # decoder_attentions=decoder_outputs.attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            # vis_encoder_last_hidden_state=vis_encoder_outputs.last_hidden_state,
            # vis_encoder_hidden_states=vis_encoder_outputs.hidden_states,
            # vis_encoder_attentions=vis_encoder_outputs.attentions,
            # cross_encoder_outputs=cross_encoder_outputs
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None,
        encoder_outputs=None,
        **kwargs):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        output = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

        if 'vis_attention_mask' in kwargs:
            output['vis_attention_mask'] = kwargs['vis_attention_mask']

        return output

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if model_kwargs.get("vis_attention_mask", None) is not None:
            model_kwargs['vis_attention_mask'] = model_kwargs['vis_attention_mask'].index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs

@dataclass
class VLSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Languaged modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see ``past_key_values`` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    vis_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    vis_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vis_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None