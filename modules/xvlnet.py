from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch
from torch import nn
from utils.pretrained_model import CLIP4ClipPreTrainedModel,Setter
from utils.loss import CrossEn, KL_Divergence
from utils.utilities import AllGather,show_log,update_attr,check_attr
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip
from modules.module_clip import CLIP, convert_weights
from modules.finegrainedtransformer import finegrainedTransformer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


logger = logging.getLogger(__name__)
allgather = AllGather.apply


class X_VLNet(CLIP4ClipPreTrainedModel):
    def __init__(self,clip_state_dict,configuration):
        super(X_VLNet,self).__init__(configuration.cross_config)
        self.configuration = configuration
        
        assert self.configuration.max_words + self.configuration.max_frames <= self.configuration.cross_config.max_position_embeddings
        
        
        ## 1. CLIP Initialization
        self.setter = Setter(clip_state_dict)
        clip_configuration = self.setter.set_clip(configuration)
        #eyes_ = self.setter.set_eyes(configuration,clip_configuration)
        self.clip = CLIP(
            clip_configuration['embed_dim'],
            clip_configuration['image_resolution'],
            clip_configuration['vision_layers'],
            clip_configuration['vision_width'],
            clip_configuration['vision_patch_size'],
            clip_configuration['context_length'], 
            clip_configuration['vocab_size'],
            clip_configuration['transformer_width'],
            clip_configuration['transformer_heads'],
            clip_configuration['transformer_layers'], 
            linear_patch='2d'
        ).float()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]
                
        convert_weights(self.clip)
        ##
        
        ## 2. Cross Initialization
        configuration.cross_config.max_position_embeddings = clip_configuration['context_length'] #maybe error raise 
        self.frame_position_embeddings = nn.Embedding(configuration.cross_config.max_position_embeddings, configuration.cross_config.hidden_size)
        self.transformerClip = TransformerClip(width=clip_configuration['transformer_width'], layers=self.configuration.cross_num_hidden_layers,
                                                   heads=clip_configuration['transformer_heads'],)
        
        
        ##
        self.s2f_transformer = finegrainedTransformer(clip_configuration['embed_dim'],clip_configuration['transformer_heads'])
        self.v2w_transformer = finegrainedTransformer(clip_configuration['embed_dim'],clip_configuration['transformer_heads'])
        
        
        # for coarse-grained constrast weights
        '''
            The following code would be deleted if the performance is worse than expected. 
        '''
        self.global_v2tmat_weight = nn.parameter.Parameter(torch.eye(clip_configuration['embed_dim']), requires_grad=True)
        self.global_t2vmat_weight = nn.parameter.Parameter(torch.eye(clip_configuration['embed_dim']), requires_grad=True)
        self.local_mat_weight = nn.parameter.Parameter(torch.eye(clip_configuration['embed_dim']), requires_grad=True)
        self.frame_mat_weight = nn.parameter.Parameter(torch.eye(configuration.max_frames), requires_grad=True)
        self.word_mat_weight = nn.parameter.Parameter(torch.eye(configuration.max_words), requires_grad=True)
        self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(configuration.max_frames), requires_grad=True)
        self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(configuration.max_words), requires_grad=True)

        
        self.logit_scale = self.clip.logit_scale.exp()
        self.loss_fct = CrossEn()
        self.kld = KL_Divergence()
        self.negative_w = 0.8
        self.temp_w = 0.0035
        self.score_threshold = 0.7
        self.temperature = 0.03
        
        self.cross_modal_attention_coefficient = configuration.cross_modal_attention_coefficient
        self.afi_clr_coefficient = configuration.afi_clr_coefficient
        
        
        self.apply(self.init_weights)
        
    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        input_ids = input_ids.view(-1,input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1,token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1,attention_mask.shape[-1])
        video_mask = video_mask.view(-1,video_mask.shape[-1])
        
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape 
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts 
        
        (sequence_output,seq_features), visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True, video_frame=video_frame)

        if self.training:
            loss = 0.
            sim_matrix,global_video_sentence_loss,fine_grained_logits, *_tmp = self.get_similarity_logits(sequence_output,seq_features, visual_output, attention_mask, video_mask,
                                                    shaped=True)
            sim_matrix +=fine_grained_logits
            
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            #loss = (global_video_sentence_loss+sim_loss)/2
            loss = sim_loss + self.afi_clr_coefficient * global_video_sentence_loss
            return loss
        else:
            return None 
        
    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden,seq_features = self.clip.encode_text(input_ids,return_hidden=True)
        sequence_hidden,seq_features = sequence_hidden.float(),seq_features.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden,seq_features    
    
    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden
    
    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output,seq_features = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return (sequence_output,seq_features), visual_output
    
    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out
    
    def get_seqential_frame_feats(self,visual_output,video_mask):
        visual_output_original = visual_output
        seq_length = visual_output.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        visual_output = visual_output + frame_position_embeddings

        extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
        visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
        visual_output = self.transformerClip(visual_output, extended_video_mask)
        visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
        visual_output = visual_output + visual_output_original
        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()
        frame_features = visual_output_original / visual_output_original.norm(dim=-1,keepdim=True)

        return frame_features
    
    def get_similarity_logits(self, sequence_output,seq_features, visual_output, attention_mask, video_mask, shaped=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1,attention_mask.shape[-1])
            video_mask = video_mask.view(-1,video_mask.shape[-1])
        
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        frame_features = self.get_seqential_frame_feats(self,visual_output,video_mask)
        
        finegrained_logits = self.get_Finegrained_logits(self,sequence_output,seq_features,frame_features)
        cross_clr_logits = self.get_AFICLR_logits()
        
            
            