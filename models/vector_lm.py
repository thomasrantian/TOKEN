# pylint: skip-file
from typing import List, Optional, Tuple, Union

import torch
from peft import PeftModelForCausalLM
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.BEV_encoder import BEVEncoder, BEVEncoderConfig
import torch.nn as nn
import numpy as np
import json
import math
from torch.nn import functional as F

class AuxSelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd,n_head, edge_dim, aux_edge_func=None, attn_pdrop=0,resid_pdrop=0,PE_len = None):
        super().__init__()
        assert n_embd % n_head == 0
        # assert edge_dim % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.q_net = nn.Linear(n_embd, n_embd)
        self.k_net = nn.Linear(n_embd+edge_dim, n_embd)
        self.v_net = nn.Linear(n_embd+edge_dim, n_embd)
        self.aux_edge_func = aux_edge_func
        
        self.PE_len = PE_len
        if PE_len is not None:
            self.PE_q = nn.Embedding(PE_len, n_embd)
            self.PE_k = nn.Embedding(PE_len, n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.attn_pdrop = attn_pdrop
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x, aux_x, mask=None,edge=None,frame_indices=None):
        # mask: (B,T,T)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_net(x)
        if edge is not None or self.aux_edge_func is not None:
            #T = aux_x.size(1)
            if edge is None:
                edge = self.aux_edge_func(aux_x,aux_x) # (B,T,T,aux_vardim)
            if self.k_net.weight.dtype==torch.float16:
                edge = edge.half()
            aug_x = torch.cat([x.unsqueeze(2).repeat_interleave(T,2), edge], dim=-1)
            k = self.k_net(aug_x)
            v = self.v_net(aug_x)
            k = k.view(B, T, T, self.n_head, C // self.n_head).permute(0, 3, 1, 2,4) # (B, nh, T, T, hs)
            q = q.view(B, T, self.n_head, 1, C // self.n_head).transpose(1, 2).repeat_interleave(T,3) # (B, nh, T, T, hs)
            v = v.view(B, T, T, self.n_head, C // self.n_head).permute(0, 3, 1, 2,4) # (B, nh, T, T, hs)
            if self.PE_len is not None:
                q = q + self.PE_q(frame_indices).view(B, T, self.n_head, 1, C // self.n_head).transpose(1, 2).repeat_interleave(T,3)
                k = k + self.PE_k(frame_indices).view(B, T, self.n_head, C // self.n_head).transpose(1,2).unsqueeze(2).repeat_interleave(T,2)
            # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q * k).sum(-1) * (1.0 / math.sqrt(k.size(-1)))
        else:
            aug_x = torch.cat([x, aux_x], dim=-1)
            k = self.k_net(aug_x)
            v = self.v_net(aug_x)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            if self.PE_len is not None:
                q = q + self.PE_q(frame_indices).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                k = k + self.PE_k(frame_indices).view(B, T, self.n_head, C // self.n_head).transpose(1,2)
            # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
        if mask is not None:
            att = att.masked_fill(mask[:,None] == 0, float('-inf'))
            att = att.masked_fill((mask==0).all(-1)[:,None,:,None], 0.0)
        finfo = torch.finfo(att.dtype)
        att = att.nan_to_num(nan=0.0, posinf=finfo.max, neginf=finfo.min)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # if v.ndim==4, (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) elif v.ndim==5, (B, nh, T, T) x (B, nh, T, T, hs) -> (B, nh, T, hs)
        y = att @ v if v.ndim==4 else (att.unsqueeze(-1)*v).sum(-2) 
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class LlamaForCausalLMVectorInput(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.weighted_loss_on_numbers = True
        if self.weighted_loss_on_numbers:
            number_tokens = [
                448,
                29900,
                29889,
                29896,
                29906,
                29941,
                29946,
                29945,
                29953,
                29955,
                29947,
                29929,
            ]  # -0.123456789
            weighted_mask = torch.ones(self.config.vocab_size)
            weighted_mask[number_tokens] = 3.0
            self.register_buffer("weighted_mask", weighted_mask)
        else:
            self.register_buffer("weighted_mask", None)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        lane_geom_descriptors=None,
        lane_topo_descriptors=None,
        agents_descriptors=None,
        ego_vehicle_descriptor=None,
        bev_descriptors=None,
        plan_query=None,
        ego_command_descriptors=None,
        query_embeds=None,
        sensor_token_mask=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        model_inputs.update(
            {
                "query_embeds": query_embeds,
                "sensor_token_mask": sensor_token_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        query_embeds: Optional[torch.FloatTensor] = None,
        sensor_token_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Ingest vectors if in generation mode (query_embeds is not None)
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        elif inputs_embeds is None and input_ids is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if query_embeds is not None and past_key_values is None:
            inputs_embeds, attention_mask, _ = ingest_vectors(
                input_ids,
                inputs_embeds,
                query_embeds,
                attention_mask,
                sensor_token_mask,
            )
            position_ids = None

        # from modeling_llama.py
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # Clip the logits to avoid overflow
        #print(logits)
        #logits = torch.clamp(logits, -10000, 10000)

        loss = None
        if labels is not None:
            def get_token_pred_loss(labels, logits):
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(weight=self.weighted_mask)
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                return loss
            loss = get_token_pred_loss(labels, logits)
 
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VectorLMWithLoRA(PeftModelForCausalLM):
    def __init__(self, model, peft_config, num_vector_tokens=64, feature_mode="", adapter_mode = "",  use_ego_state_info = "", root_data_path="", freeze_LORA=False):
        super().__init__(model, peft_config)
        
        if freeze_LORA:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False
        # print all the trainable parameters in self.model
        for name, param in self.model.named_parameters():
           if param.requires_grad:
               print(name)

        self.num_vector_tokens = num_vector_tokens
        self.feature_mode = feature_mode
        self.adapter_mode = adapter_mode
        self.use_ego_state_info = use_ego_state_info

        bev_id_to_sample_token_map_train_path = 'cached_dicts/bev_id_to_sample_token_map_train.json'
        bev_id_to_sample_token_map_val_path = 'cached_dicts/bev_id_to_sample_token_map_val.json'
        # appendd the self.root_data_path to the path
        bev_id_to_sample_token_map_train_path = root_data_path + bev_id_to_sample_token_map_train_path
        bev_id_to_sample_token_map_val_path = root_data_path + bev_id_to_sample_token_map_val_path
        with open(bev_id_to_sample_token_map_train_path) as f:
            self.bev_id_to_sample_token_map_train = json.load(f)
        with open(bev_id_to_sample_token_map_val_path) as f:
            self.bev_id_to_sample_token_map_val = json.load(f)
        
        # CTT_paradrive_agent_id_map_path = 'agent_level_feature_index_2_CTT_agent_id_map.json'
        # CTT_paradrive_agent_id_map_path = root_data_path + CTT_paradrive_agent_id_map_path
        # with open(CTT_paradrive_agent_id_map_path) as f:
        #     self.CTT_paradrive_agent_id_map = json.load(f)
        
        # sample_token_2_scee_id_ts_map_path = 'sample_token_2_scee_id_ts_map.json'
        # sample_token_2_scee_id_ts_map_path = root_data_path + sample_token_2_scee_id_ts_map_path
        # with open(sample_token_2_scee_id_ts_map_path) as f:
        #     self.sample_token_2_scee_id_ts_map = json.load(f)

        # Only use the ego state vector as the latent token (project to 1x4096 token), directly inject to LLM
        if self.feature_mode == 'ego_only':
            self.ego_info_proj = torch.nn.Linear(
                4, 4096
            )
            self.modules_to_save = ["ego_info_proj"]
        
        elif self.feature_mode == "object_decode_pretrain":
            print('Train a simple MLP to decode the object level tokens')
            self.object_latent_proj = torch.nn.Linear(
                    1026, 4096
                )
            self.modules_to_save = ["object_latent_proj"]
        
        elif self.feature_mode in['object_level_token_visual_state_with_map',
                                  'object_level_token_visual_state_without_map', # without paradrive map token 
                                  'object_level_token_visual_only_with_map', 
                                  'object_level_token_visual_only_without_map']:
            
            print('Train a simple MLP to decode the object level tokens')

            self.object_latent_proj = torch.nn.Linear(
                    1026, 4096
                )
            
            # Project the ego query (default to be track + traj, 1024) to 1024
            self.ego_query_proj = torch.nn.Linear(
                    1024, 4096
                )
            self.modules_to_save = ["object_latent_proj", "ego_query_proj"]
            
            # Project the map token to the text token embedding, and inject to LLM
            if 'with_map' in self.feature_mode:
                self.map_latent_proj = torch.nn.Linear(
                        256, 4096
                    )
                self.modules_to_save.append("map_latent_proj")
            
            # Use the agent2lane token
            if 'visual_state' in self.feature_mode:
                # We also inject the state tokens
                self.agent_lane_token_extractor = AuxSelfAttention(256, 4, 21, attn_pdrop=0.1, resid_pdrop=0.1)
                self.agent_lane_token_proj = torch.nn.Linear(8192, 4096)
                self.modules_to_save.append("agent_lane_token_extractor")
                self.modules_to_save.append("agent_lane_token_proj")
            
            if self.use_ego_state_info: # we add another project to project the ego state to 4096
                self.ego_state_proj = torch.nn.Linear(
                    4, 4096
                )
                self.modules_to_save.append("ego_state_proj")
        
        elif self.feature_mode == 'object_only':
            
            if self.adapter_mode == 'tf_decoder':
                print('Using transformer decoder as the adapter to process the object level features')
                attn_module_layer = nn.TransformerDecoderLayer(
                    1024,
                    8,
                    dim_feedforward=1024 * 2,
                    dropout=0.1,
                    batch_first=True,
                )
                # Project the object feature to 256 for the transformer decoder
                self.object_latent_proj = torch.nn.Linear(
                   1024, 1024
                )  
                self.bev_Q_former = nn.TransformerDecoder(attn_module_layer, 3)

                # Project the ego query (default to be track + traj, 1024) to 1024
                self.ego_query_proj = torch.nn.Linear(
                    1024, 1024
                )
                self.modules_to_save = ["object_latent_proj", "bev_Q_former", "ego_query_proj"]
                # In ego pred only task, we could use the command query to guide the transformer decoder for verification
                # if self.use_ego_state_info:
                #     print('Using command query to guide the transformer decoder')
                #     self.navi_embed = nn.Embedding(3, 1024)
                #     # Here we change the ego_query_proj to 2048 since we augment the ego_query with the command query
                #     self.ego_query_proj = torch.nn.Linear(
                #         2048, 1024
                #     )
                #     self.modules_to_save.append("navi_embed")
                
                # Project the final latent token to 4096
                self.final_latent_proj = torch.nn.Linear(
                    1024, 4096
                )
                self.modules_to_save.append("final_latent_proj")
            
            elif self.adapter_mode == 'direct_inject':
                print('Directly inject the object level features (object tokens + ego tokens)to LLM')
                # Directly project the object tokens to 4096, andd inject to LLM
                self.object_latent_proj = torch.nn.Linear(
                    1026, 4096
                )
                self.ego_query_proj = torch.nn.Linear(
                    1024, 4096
                )
                # Project the map token to the text token embedding, and inject to LLM
                self.map_latent_proj = torch.nn.Linear(
                    256, 4096
                )
                self.modules_to_save = ["object_latent_proj", "ego_query_proj", "map_latent_proj"]  

            elif self.adapter_mode == 'perceiver':
                print('Using perceiver as the adapter to process the object level features')
                self.bev_Q_former = BEVEncoder(BEVEncoderConfig(), num_vector_tokens)
            
            
            elif self.adapter_mode == 'none': # We directly project the object token to 4096 and inject to LLM
                pass
        
        elif self.feature_mode == 'bev_ego':
            n_vector_query = num_vector_tokens + 1
            self.ego_info_proj = torch.nn.Linear(
                4, 256
            )
        else:
            print('Undefined mode!')
            return None

        self.root_data_path = root_data_path
        
        self.to(model.device)
       
        self.generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=10,
            use_cache=False,
            do_sample=True,
            max_length=384,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            _from_model_config=False,
        )

    def fill_bev_descriptors(self, plan_query, split_folder):
        bev_descriptors = []
        for i in range(plan_query.shape[0]):
            sample_token = plan_query[i].detach().cpu().numpy()
            # convert to string
            sample_token = str(sample_token)
            bev_feature_path = self.root_data_path + split_folder + str(sample_token) + '/latent_feature.npy'
            bev_descriptor = np.load(bev_feature_path)
            bev_descriptors.append(bev_descriptor)
        # convert to numpy array first then to tensor
        bev_descriptors = np.array(bev_descriptors)
        bev_descriptors = torch.tensor(bev_descriptors).squeeze(2)
        return bev_descriptors
    
    def construct_a2l_token(self, sample_token, target_agent_index):
        # get the scene_id and ts from the sample_token
        train_mode_scene_ts = self.sample_token_2_scee_id_ts_map[sample_token]
        train_mode = train_mode_scene_ts[0]
        scene_id = train_mode_scene_ts[1]
        scene_ts = train_mode_scene_ts[2]
        CTT_object_feature_path = '/home/rant/media/rant/CTT_mode_data/' + train_mode + '/3s/' + scene_id + '/' + scene_ts + '/CTT_object_features.npy'
        CTT_object_feature = np.load(CTT_object_feature_path, allow_pickle=True).item()
        a2l_feature = CTT_object_feature['a2l'].squeeze(0)
        l2l_feature = CTT_object_feature['l2l'].squeeze(0)
        ego_agent_index = 0
        # concat the a2l feature of the ego agent and the target agent
        x = np.concatenate([a2l_feature[ego_agent_index], a2l_feature[target_agent_index]], axis=1)
        x = torch.tensor(x).unsqueeze(0).float()

        # pad x to 1x32x256 with zeros
        x = F.pad(x, (0, 0, 0, 32-x.size(1)), 'constant', 0)

        l2l_feature = torch.tensor(l2l_feature).float() # 10 x 10 x 21
        # Pad l2l_feature to 10 x 32 x 21 with zeros
        padding_left = 0
        padding_right = 0
        padding_top = 0
        padding_bottom = 32 - l2l_feature.size(1)
        l2l_feature = F.pad(l2l_feature, (padding_left, padding_right, padding_top, padding_bottom), 'constant', 0)

        n_needed = 32 - l2l_feature.size(0)
        # construct a neededd x 32 x 21 tensor with zeros
        pad = torch.zeros(n_needed, 32, 21)
        # concatenate l2l_feature with pad
        l2l_feature = torch.cat([l2l_feature, pad], dim=0)
        l2l_feature = l2l_feature.unsqueeze(0)


        # construct the mask
        mask = torch.ones(1, 4, 32, 32)

        # make the 1 x 4 x needed:, needed: tensor to be zeros
        mask[:, :, :n_needed, :n_needed] = 0

        # move x edge and mask to the device of the model
        x = x.to(self.model.device)
        l2l_feature = l2l_feature.to(self.model.device)
        mask = mask.to(self.model.device)
        output = self.agent_lane_token_extractor(x, None, edge=l2l_feature, mask=mask) # 1 x 32 x 256
        return output


    def fill_object_levelfeature(self, plan_query, valid_object_index = None, object_speeds = None):
        
        object_level_features = []
        ego_queries = []
        map_queries = []
        CTT_object_level_features = []

        for i in range(plan_query.shape[0]):
            sample_token = plan_query[i].detach().cpu().numpy()
            sample_token_actual = None
            if sample_token > 0:
                split_folder = 'cached_object_tokens/'
                sample_token_actual = self.bev_id_to_sample_token_map_train[str(int(sample_token))]
            else:
                split_folder = 'cached_object_tokens/'
                sample_token_actual = self.bev_id_to_sample_token_map_val[str(abs(int(sample_token)))]
            sample_token = abs(sample_token)
            # convert to string
            sample_token = str(sample_token)
            object_levelfeature_path = self.root_data_path + split_folder + str(sample_token) + '/object_level_feature.npy'
            object_levelfeature_dict = np.load(object_levelfeature_path, allow_pickle=True).item()
            
            ## Process the object features
            object_track_query = torch.tensor(object_levelfeature_dict['track']).squeeze()
            # Check if the object_traj_query_pooled dim size is 3
            if len(object_track_query.shape) != 2: # This could happen when there is only one agent
                object_track_query = object_track_query.unsqueeze(0) # Insert the agent dim
            object_traj_query = torch.tensor(object_levelfeature_dict['traj'])
            object_traj_query_pooled = torch.mean(object_traj_query, 3).squeeze() # To do: Why 3? should be 2
            # Check if the object_traj_query_pooled dim size is 3
            if len(object_traj_query_pooled.shape) != 3: # This could happen when there is only one agent
                object_traj_query_pooled = object_traj_query_pooled.unsqueeze(1) # Insert the agent dim
            object_traj_query_stacked = torch.cat((object_traj_query_pooled[0], object_traj_query_pooled[1], object_traj_query_pooled[2]), 1)
            object_feature = torch.cat((object_track_query, object_traj_query_stacked), 1) # N_agent x 1024
                
            # Get the map query
            map_query = torch.tensor(object_levelfeature_dict['map_query']).squeeze() # (M+1) x 256

            # We only keep the first 10 rows of the map query
            map_query = map_query[:10]
            
            # We only use the valid object index
            if valid_object_index is not None:
                # filter out the elements that larger than 0 in the valid_object_index
                cut_index = len(valid_object_index[0])
                for j in range(len(valid_object_index[0])):
                    if valid_object_index[i, j] < 0:
                        cut_index = j
                        break
                current_valid_object_index = valid_object_index[i, :cut_index].cpu().long()
                if len(current_valid_object_index) == 0:
                    current_valid_object_index = range(len(object_feature))
                object_feature = object_feature[current_valid_object_index]
            
            # If the number of agents is less than 50, fill the empty space with zeros
            if object_feature.shape[0] < 30:
                n_needed_rows = 30 - object_feature.shape[0]
                temp = torch.zeros((n_needed_rows, 1024))
                object_feature = torch.cat((object_feature, temp))
            elif object_feature.shape[0] > 30:
                object_feature = object_feature[:30]

             # Concat with the object speed
            if object_speeds is not None:
                object_feature = torch.cat((object_feature, object_speeds[i].cpu()), 1)
            
            object_level_features.append(object_feature)

            if 'visual_state' in self.feature_mode:
                # For all the valid paradrive agent index, we construct the state token for each agent
                CTT_object_level_feature = []
                if sample_token_actual in self.CTT_paradrive_agent_id_map:
                    for valid_agent_index in current_valid_object_index:
                        if str(int(valid_agent_index)) in self.CTT_paradrive_agent_id_map[sample_token_actual]:
                            CTT_agent_index = self.CTT_paradrive_agent_id_map[sample_token_actual][str(int(valid_agent_index))]
                            agent_2_lane_token = self.construct_a2l_token(sample_token_actual, int(CTT_agent_index)) # 1 x 32 x 256
                            CTT_object_level_feature.append(agent_2_lane_token.cpu())
                        else:
                            pad_token = torch.zeros(1, 32, 256)
                            CTT_object_level_feature.append(pad_token)
                else:
                    for valid_agent_index in current_valid_object_index:
                        pad_token = torch.zeros(1, 32, 256)
                        CTT_object_level_feature.append(pad_token)
                CTT_object_level_feature = torch.cat(CTT_object_level_feature, 0) # N_agent x 32 x 256

                if CTT_object_level_feature.shape[0] < 30:
                    n_needed_rows = 30 - CTT_object_level_feature.shape[0]
                    temp = torch.zeros((n_needed_rows, 32, 256))
                    CTT_object_level_feature = torch.cat((CTT_object_level_feature, temp))
                elif CTT_object_level_feature.shape[0] > 30:
                    CTT_object_level_feature = CTT_object_level_feature[:30]
                
                CTT_object_level_features.append(CTT_object_level_feature)
 
            
            ## Process the ego query
            sdc_track_query = torch.tensor(object_levelfeature_dict['ego_track_query']).squeeze()
            if len(sdc_track_query.shape) == 1:
                sdc_track_query = sdc_track_query.unsqueeze(0) # 1 x 256
            
            sdc_traj_query = torch.tensor(object_levelfeature_dict['ego_traj_query'])
            sdc_traj_query_pooled = torch.mean(sdc_traj_query, 2).squeeze() # To do: Why 3? should be 2
            if len(sdc_traj_query_pooled.shape) == 2:
                sdc_traj_query_pooled = sdc_traj_query_pooled.unsqueeze(1)
            sdc_track_query_stacked = torch.cat((sdc_traj_query_pooled[0], sdc_traj_query_pooled[1], sdc_traj_query_pooled[2]), 1)
            
            ego_feature = torch.cat((sdc_track_query, sdc_track_query_stacked), 1) # 1 x 1024
            ego_queries.append(ego_feature)
            map_queries.append(map_query)

        # Stack the object_level_features list along the first dimension to form a tensor
        object_level_features = torch.stack(object_level_features)
        ego_queries = torch.stack(ego_queries)
        map_queries = torch.stack(map_queries)
        if 'visual_state' in self.feature_mode:
            CTT_object_level_features = torch.stack(CTT_object_level_features)
            return object_level_features, ego_queries, map_queries, CTT_object_level_features
        elif 'visual_only' in self.feature_mode:
            return object_level_features, ego_queries, map_queries, None
        else:
            return object_level_features, ego_queries, map_queries, None
                    
    def embed_vector_and_prompt(
        self,
        input_ids,
        question_input_ids,
        attention_mask,
        labels,
        lane_geom_descriptors,
        lane_topo_descriptors,
        agents_descriptors,
        ego_vehicle_descriptor,
        bev_descriptors,
        plan_query,
        ego_command_descriptors,
    ):

        # Generate token embeddings (full user input, question + answer)
        inputs_embeds = self.model.model.embed_tokens(input_ids) # B x inputs_token_length x 4096

        

        if self.feature_mode == 'ego_only':
            ego_info_query = self.ego_info_proj(ego_vehicle_descriptor).unsqueeze(1) # B x 1 x 4096
            sensor_token_embeds = ego_info_query
        
        elif self.feature_mode == 'object_decode_pretrain':
            # bev_descriptors = bev_descriptors.to(ego_vehicle_descriptor.device)
            # if len(bev_descriptors.shape) == 2:
            #     bev_descriptors = bev_descriptors.unsqueeze(1) # B x 1 x 1026 [we insert the agent dim]
            # sensor_token_embeds = self.object_latent_proj(bev_descriptors) # B x 1 x 4096

            valid_object_index = bev_descriptors.squeeze()
            # First load the latent features 
            objects_queries, ego_queries, map_queries, CTT_object_level_features = self.fill_object_levelfeature(plan_query, valid_object_index, agents_descriptors)
            objects_queries = objects_queries[:, 0, :].unsqueeze(1).to(ego_vehicle_descriptor.device) # We only keep the first object token           
            # build the objects_queries mask from objects_queries that masks out the padding zeros
            objects_queries_mask = torch.ones(objects_queries.shape[0], objects_queries.shape[1])
            for i in range(objects_queries.shape[0]):
                for j in range(objects_queries.shape[1]):
                    if torch.sum(objects_queries[i, j]) == 0:
                        objects_queries_mask[i, j] = 0

            sensor_token_embeds = self.object_latent_proj(objects_queries) # B x 1 x 4096
            sensor_token_mask = objects_queries_mask.to(ego_vehicle_descriptor.device)
           
        elif self.feature_mode in['object_level_token_visual_state_with_map',
                                  'object_level_token_visual_state_without_map', # without paradrive map token 
                                  'object_level_token_visual_only_with_map', 
                                  'object_level_token_visual_only_without_map']:
            valid_object_index = bev_descriptors.squeeze()
            # First load the latent features 
            objects_queries, ego_queries, map_queries, CTT_object_level_features = self.fill_object_levelfeature(plan_query, valid_object_index, agents_descriptors)
            objects_queries = objects_queries.to(ego_vehicle_descriptor.device)
            # build the objects_queries mask from objects_queries that masks out the padding zeros
            objects_queries_mask = torch.ones(objects_queries.shape[0], objects_queries.shape[1])
            for i in range(objects_queries.shape[0]):
                for j in range(objects_queries.shape[1]):
                    if torch.sum(objects_queries[i, j]) == 0:
                        objects_queries_mask[i, j] = 0

            objects_queries = self.object_latent_proj(objects_queries) # B x N x 4096
            ego_queries = ego_queries.to(ego_vehicle_descriptor.device)
            # we always have the ego query
            ego_queries_mask = torch.ones(ego_queries.shape[0], ego_queries.shape[1])
            ego_queries = self.ego_query_proj(ego_queries) # B x 1 x 1024
            
            # Process the map tokens
            if 'with_map' in self.feature_mode:
                map_queries = map_queries.to(ego_vehicle_descriptor.device)
                map_queries_mask = torch.ones(map_queries.shape[0], map_queries.shape[1])
                map_queries = self.map_latent_proj(map_queries) # B x 101 x 4096

            ego_info_query = None
            ego_info_query_mask = None
            if self.use_ego_state_info:
                ego_info_query = self.ego_state_proj(ego_vehicle_descriptor).unsqueeze(1) # B x 1 x 4096
                ego_info_query_mask = torch.ones(ego_info_query.shape[0], ego_info_query.shape[1])

            if 'visual_state' in self.feature_mode:
            
                CTT_object_level_features = CTT_object_level_features.to(ego_vehicle_descriptor.device)
                # flat the last two dimensions of the CTT_object_level_features
                shape = CTT_object_level_features.shape
                CTT_object_level_features = CTT_object_level_features.reshape(*shape[:-2], shape[-2] * shape[-1])

                # build the CTT_object_level_features mask from CTT_object_level_features that masks out the padding zeros
                CTT_object_level_features_mask = torch.ones(CTT_object_level_features.shape[0], CTT_object_level_features.shape[1])
                for i in range(CTT_object_level_features.shape[0]):
                    for j in range(CTT_object_level_features.shape[1]):
                        if torch.sum(CTT_object_level_features[i, j]) == 0:
                            CTT_object_level_features_mask[i, j] = 0
                CTT_object_level_features = self.agent_lane_token_proj(CTT_object_level_features) # B x n_obs x 4096
                
                if 'with_map' in self.feature_mode:
                    if self.use_ego_state_info:
                        sensor_token_embeds = torch.cat((ego_queries, ego_info_query, objects_queries, map_queries, CTT_object_level_features), 1)
                        sensor_token_mask = torch.cat((ego_queries_mask, ego_info_query_mask, objects_queries_mask, map_queries_mask, CTT_object_level_features_mask), 1).to(ego_vehicle_descriptor.device) # B x (n_obs + 1 + 101 + 1)
                    else:
                        sensor_token_embeds = torch.cat((ego_queries, objects_queries, map_queries, CTT_object_level_features), 1) # B x (n_obs + 1 + 101) x 4096
                        sensor_token_mask = torch.cat((ego_queries_mask, objects_queries_mask, map_queries_mask, CTT_object_level_features_mask), 1).to(ego_vehicle_descriptor.device) # B x (n_obs + 1 + 101)
                
                elif 'without_map' in self.feature_mode:
                    sensor_token_embeds = torch.cat((ego_queries, objects_queries, CTT_object_level_features), 1)
                    sensor_token_mask = torch.cat((ego_queries_mask, objects_queries_mask, CTT_object_level_features_mask), 1).to(ego_vehicle_descriptor.device)
        
            elif 'visual_only' in self.feature_mode:
                if 'with_map' in self.feature_mode:
                    
                    if self.use_ego_state_info:
                        sensor_token_embeds = torch.cat((ego_queries, ego_info_query, objects_queries, map_queries), 1)
                        sensor_token_mask = torch.cat((ego_queries_mask, ego_info_query_mask, objects_queries_mask, map_queries_mask), 1).to(ego_vehicle_descriptor.device)
                    else:
                        sensor_token_embeds = torch.cat((ego_queries, objects_queries, map_queries), 1)
                        sensor_token_mask = torch.cat((ego_queries_mask, objects_queries_mask, map_queries_mask), 1).to(ego_vehicle_descriptor.device)

                elif 'without_map' in self.feature_mode:
                    if self.use_ego_state_info:
                        sensor_token_embeds = torch.cat((ego_queries, ego_info_query, objects_queries), 1)
                        sensor_token_mask = torch.cat((ego_queries_mask, ego_info_query_mask, objects_queries_mask), 1).to(ego_vehicle_descriptor.device)
                    else:
                        sensor_token_embeds = torch.cat((ego_queries, objects_queries), 1)
                        sensor_token_mask = torch.cat((ego_queries_mask, objects_queries_mask), 1).to(ego_vehicle_descriptor.device)
        
        final_query_embeds = sensor_token_embeds
        final_query_embeds_mask = sensor_token_mask

        # Concatenate the vector embeddings with the token embeddings [Do we concatenate the question tokens here?]
        new_inputs_embeds, new_attention_mask, new_labels = ingest_vectors(
            input_ids,
            inputs_embeds,
            final_query_embeds,
            attention_mask,
            final_query_embeds_mask,
            labels,
        )

        return new_inputs_embeds, new_attention_mask, new_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        lane_geom_descriptors=None,
        lane_topo_descriptors=None,
        agents_descriptors=None,
        ego_vehicle_descriptor=None,
        bev_descriptors=None,
        plan_query=None,
        ego_command_descriptors=None,
        instruction = None,
        **kwargs,  # those are 'user_input_ids', 'user_attention_mask'
    ):
        inputs_embeds, attention_mask, labels = self.embed_vector_and_prompt(
            input_ids,
            instruction,
            attention_mask,
            labels,
            lane_geom_descriptors,
            lane_topo_descriptors,
            agents_descriptors,
            ego_vehicle_descriptor,
            bev_descriptors,
            plan_query,
            ego_command_descriptors,
        )

        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        loss = outputs.loss

        return {"loss": loss}

    def generate(self, **kwargs):
        lane_geom_descriptors = kwargs["lane_geom_descriptors"]
        lane_topo_descriptors = kwargs["lane_topo_descriptors"]
        agents_descriptors = kwargs["agents_descriptors"]
        ego_vehicle_descriptor = kwargs["ego_vehicle_descriptor"]
        bev_descriptors = kwargs["bev_descriptors"]
        plan_query = kwargs["plan_query"]
        ego_command_descriptors = kwargs["ego_command_descriptors"]

        if self.feature_mode == 'ego_only':
            ego_info_query = self.ego_info_proj(ego_vehicle_descriptor).unsqueeze(1) # B x 1 x 4096
            sensor_token_embeds = ego_info_query
        
        elif self.feature_mode == 'object_decode_pretrain':
            # bev_descriptors = bev_descriptors.to(ego_vehicle_descriptor.device)
            # if len(bev_descriptors.shape) == 2:
            #     bev_descriptors = bev_descriptors.unsqueeze(1) # B x 1 x 1026 [we insert the agent dim]
            # sensor_token_embeds = self.object_latent_proj(bev_descriptors) # B x 1 x 4096

            valid_object_index = bev_descriptors.squeeze()
            # First load the latent features 
            objects_queries, ego_queries, map_queries, CTT_object_level_features = self.fill_object_levelfeature(plan_query, valid_object_index, agents_descriptors)
            objects_queries = objects_queries[:, 0, :].unsqueeze(1).to(ego_vehicle_descriptor.device) # We only keep the first object token           
            # build the objects_queries mask from objects_queries that masks out the padding zeros
            objects_queries_mask = torch.ones(objects_queries.shape[0], objects_queries.shape[1])
            for i in range(objects_queries.shape[0]):
                for j in range(objects_queries.shape[1]):
                    if torch.sum(objects_queries[i, j]) == 0:
                        objects_queries_mask[i, j] = 0

            sensor_token_embeds = self.object_latent_proj(objects_queries) # B x 1 x 4096
            sensor_token_mask = objects_queries_mask.to(ego_vehicle_descriptor.device)

        
        elif self.feature_mode in['object_level_token_visual_state_with_map',
                                  'object_level_token_visual_state_without_map', # without paradrive map token 
                                  'object_level_token_visual_only_with_map', 
                                  'object_level_token_visual_only_without_map']:
            valid_object_index = bev_descriptors.squeeze()
            # Firt load the latent features 
            objects_queries, ego_queries, map_queries, CTT_object_level_features = self.fill_object_levelfeature(plan_query, valid_object_index, agents_descriptors)
            objects_queries = objects_queries.to(ego_vehicle_descriptor.device)
            # build the objects_queries mask from objects_queries that masks out the padding zeros
            objects_queries_mask = torch.ones(objects_queries.shape[0], objects_queries.shape[1])
            for i in range(objects_queries.shape[0]):
                for j in range(objects_queries.shape[1]):
                    if torch.sum(objects_queries[i, j]) == 0:
                        objects_queries_mask[i, j] = 0

            objects_queries = self.object_latent_proj(objects_queries) # B x N x 4096
            ego_queries = ego_queries.to(ego_vehicle_descriptor.device)
            # we always have the ego query
            ego_queries_mask = torch.ones(ego_queries.shape[0], ego_queries.shape[1])
            ego_queries = self.ego_query_proj(ego_queries) # B x 1 x 1024
            
            ego_info_query = None
            ego_info_query_mask = None
            if self.use_ego_state_info:
                ego_info_query = self.ego_state_proj(ego_vehicle_descriptor).unsqueeze(1) # B x 1 x 4096
                ego_info_query_mask = torch.ones(ego_info_query.shape[0], ego_info_query.shape[1])

            # Process the map tokens
            if 'with_map' in self.feature_mode:
                map_queries = map_queries.to(ego_vehicle_descriptor.device)
                map_queries_mask = torch.ones(map_queries.shape[0], map_queries.shape[1])
                map_queries = self.map_latent_proj(map_queries) # B x 101 x 4096

            if 'visual_state' in self.feature_mode:
            
                CTT_object_level_features = CTT_object_level_features.to(ego_vehicle_descriptor.device)
                # flat the last two dimensions of the CTT_object_level_features
                shape = CTT_object_level_features.shape
                CTT_object_level_features = CTT_object_level_features.reshape(*shape[:-2], shape[-2] * shape[-1])

                # build the CTT_object_level_features mask from CTT_object_level_features that masks out the padding zeros
                CTT_object_level_features_mask = torch.ones(CTT_object_level_features.shape[0], CTT_object_level_features.shape[1])
                for i in range(CTT_object_level_features.shape[0]):
                    for j in range(CTT_object_level_features.shape[1]):
                        if torch.sum(CTT_object_level_features[i, j]) == 0:
                            CTT_object_level_features_mask[i, j] = 0
                CTT_object_level_features = self.agent_lane_token_proj(CTT_object_level_features) # B x n_obs x 4096
                
                if 'with_map' in self.feature_mode:
                    sensor_token_embeds = torch.cat((ego_queries, objects_queries, map_queries, CTT_object_level_features), 1) # B x (n_obs + 1 + 101) x 4096
                    sensor_token_mask = torch.cat((ego_queries_mask, objects_queries_mask, map_queries_mask, CTT_object_level_features_mask), 1).to(ego_vehicle_descriptor.device) # B x (n_obs + 1 + 101)
                elif 'without_map' in self.feature_mode:
                    sensor_token_embeds = torch.cat((ego_queries, objects_queries, CTT_object_level_features), 1)
                    sensor_token_mask = torch.cat((ego_queries_mask, objects_queries_mask, CTT_object_level_features_mask), 1).to(ego_vehicle_descriptor.device)
        
            elif 'visual_only' in self.feature_mode:
                if 'with_map' in self.feature_mode:
                    
                    if self.use_ego_state_info:
                        sensor_token_embeds = torch.cat((ego_queries, ego_info_query, objects_queries, map_queries), 1)
                        sensor_token_mask = torch.cat((ego_queries_mask, ego_info_query_mask, objects_queries_mask, map_queries_mask), 1).to(ego_vehicle_descriptor.device)
                    else:
                        sensor_token_embeds = torch.cat((ego_queries, objects_queries, map_queries), 1)
                        sensor_token_mask = torch.cat((ego_queries_mask, objects_queries_mask, map_queries_mask), 1).to(ego_vehicle_descriptor.device)

                elif 'without_map' in self.feature_mode:
                    if self.use_ego_state_info:
                        sensor_token_embeds = torch.cat((ego_queries, ego_info_query, objects_queries), 1)
                        sensor_token_mask = torch.cat((ego_queries_mask, ego_info_query_mask, objects_queries_mask), 1).to(ego_vehicle_descriptor.device)
                    else:
                        sensor_token_embeds = torch.cat((ego_queries, objects_queries), 1)
                        sensor_token_mask = torch.cat((ego_queries_mask, objects_queries_mask), 1).to(ego_vehicle_descriptor.device)      

        final_query_embeds = 1.0 * sensor_token_embeds

        kwargs["query_embeds"] = final_query_embeds
        kwargs["input_ids"] = kwargs.pop("user_input_ids")
        kwargs["attention_mask"] = kwargs.pop("user_attention_mask")
        kwargs.pop("instruction")
        kwargs["sensor_token_mask"] = sensor_token_mask
        if "generation_config" not in kwargs:
            kwargs[
                "generation_config"
            ] = (
                self.generation_config
            )  # Override the generation config to make the padding tokens correct
        outputs = self.base_model.generate(**kwargs)
        return outputs


def ingest_vectors(
    input_ids, inputs_embeds, input_vectors, attention_mask, final_query_embeds_mask, labels=None
):
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    vector_length = input_vectors.shape[1]
    # Find the position of the specific token sequence (10567 and 29901) (Input:) for each instance in the batch
    token_sequence = torch.tensor([10567, 29901], device=input_ids.device)
    positions = (input_ids[:, :-1] == token_sequence[0]) & (
        input_ids[:, 1:] == token_sequence[1]
    )

    # Add 3 to get the vector insertion positions, and handle cases where the sequence is not found
    vector_input_positions = torch.argmax(positions.float(), dim=1) + 3
    vector_input_positions[vector_input_positions == 3] = 0
    vector_input_positions[vector_input_positions > seq_length] = seq_length
    # Create tensors to hold the updated inputs_embeds, attention_mask, and labels
    new_inputs_embeds = torch.zeros(
        batch_size,
        seq_length + vector_length,
        inputs_embeds.shape[2],
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    new_attention_mask = torch.zeros(
        batch_size,
        seq_length + vector_length,
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    new_labels = (
        torch.zeros(
            batch_size,
            seq_length + vector_length,
            device=labels.device,
            dtype=labels.dtype,
        )
        if labels is not None
        else None
    )
    for b in range(batch_size):
        vector_input_position = vector_input_positions[b]
        if vector_input_position == 0:
            vector_input_position = 1  # Insert the vector embeddings at position 1 if the token_sequence is not found (avoid the bos_token)
        new_inputs_embeds[b, :vector_input_position] = inputs_embeds[
            b, :vector_input_position
        ]
        new_inputs_embeds[
            b, vector_input_position : vector_input_position + vector_length
        ] = input_vectors[b]
        new_inputs_embeds[b, vector_input_position + vector_length :] = inputs_embeds[
            b, vector_input_position:
        ]

        new_attention_mask[b, :vector_input_position] = attention_mask[
            b, :vector_input_position
        ]
        new_attention_mask[
            b, vector_input_position : vector_input_position + vector_length
        ] = final_query_embeds_mask[b]
        new_attention_mask[b, vector_input_position + vector_length :] = attention_mask[
            b, vector_input_position:
        ]

        if labels is not None:
            new_labels[b, :vector_input_position] = labels[b, :vector_input_position]
            new_labels[
                b, vector_input_position : vector_input_position + vector_length
            ] = -100
            new_labels[b, vector_input_position + vector_length :] = labels[
                b, vector_input_position:
            ]

    return new_inputs_embeds, new_attention_mask, new_labels
