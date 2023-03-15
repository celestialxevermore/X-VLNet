import os 
import torch 
import random 
import numpy as np 
from config.configuration import Configuration
from torch.utils.tensorboard.writer import SummaryWriter
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_factory import DataDirector 
from utils.utilities import parallel_apply, get_logger, prep_optimizer
from utils.setting import Initializer
from utils.train_test import Trainer,Tester

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
torch.distributed.init_process_group(backend="nccl")

def main():
    global logger 
    configuration = Configuration()
    initializer = Initializer()
    if not configuration.no_tensorboard:
        writer = SummaryWriter(log_dir = configuration.tb_log_dir)
    else:
        writer = None 
    
    tokenizer = ClipTokenizer()
    configuration,logger = initializer.set_seed_logger(configuration)
    device,n_gpu = initializer.init_device(configuration,logger,configuration.local_rank)
    
    model = initializer.init_model(configuration, device, n_gpu, configuration.local_rank)
    
    
    ## ####################################
    # train and eval
    ## ####################################
    
    if configuration.do_train:
        train_dataloader,train_length,train_sampler = DataDirector.get_dataloader(configuration,tokenizer,split_type=configuration.do_train)
        num_train_optimization_steps = (int(len(train_dataloader) + configuration.gradient_accumulation_steps - 1)
                                        / configuration.gradient_accumulation_steps) * configuration.epochs

        optimizer,scheduler,model = initializer.prep_optimizer(configuration,model,num_train_optimization_steps,device,n_gpu,configuration.local_rank,coef_lr=configuration.coef_lr)
        if configuration.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", configuration.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * configuration.gradient_accumulation_steps)

        #if configuration.resume_model:
        trainer = Trainer(model,optimizer,configuration,train_dataloader,scheduler,device,train_sampler,logger)
        
    
    elif configuration.do_eval:
        test_Dataloader,test_length = DataDirector.get_dataloader(configuration,tokenizer,configuration.do_eval)

        