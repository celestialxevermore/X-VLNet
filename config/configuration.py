import os 
import argparse
from config.base_config import baseconfig


class Configuration(baseconfig):
    def __init__(self):
        super(Configuration,self).__init__()
        
    def parse_args(self):
        description = 'Text Video Retrieval Task'
        parser = argparse.ArgumentParser(description=description)
        
        # data parameters 
        parser.add_argument("--do_train", action='store_true',help="Whether to run training")
        parser.add_argument("do_eval",action='store_true',help="Whether to run eval on the dev set")
        
        parser.add_argument("dataset_name",type=str,default='MSRVTT',help="Dataset name")
        
        parser.add_argument("--pretrained_clip_name",default="ViT-B/32",type=str)
        parser.add_argument("--world_size",default=0,type=int, help="distributed training")
        parser.add_argument("--local_rank",default=0,type=int,help="distributed training")
        
        parser.add_argument("--epochs",type=int,default=5,help='upper epoch limit')
        parser.add_argument("--batch_size",type=int,help="batch_size")
        parser.add_argument("--output_dir",default=None,type=str,required=True)
        parser.add_argument("--cross_modal_attention_coefficient",default=0.8,type=float,required=True,help="coefficient for balancing the fine grained cross-attention similarity scores")
        parser.add_argument("--afi_clr_coefficient",default=0.5,type=float,required=True,help="balancing for the total loss")