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

        self.apply(self.init_weights)
        
        
    def forward(self):
        pass 
        
        