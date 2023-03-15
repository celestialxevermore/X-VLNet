import os 
import torch 
import numpy as np 
import random
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from utilities import parallel_apply,get_logger
from modules.optimization import BertAdam
from modules.xvlnet import X_VLNet

class Initializer():
    def __init__(self):
        super(Initializer,self).__init__()
        
    def set_seed_logger(self,configuration):
        # predefining random initial seeds
        random.seed(configuration.seed)
        os.environ['PYTHONHASHSEED'] = str(configuration.seed)
        np.random.seed(configuration.seed)
        torch.manual_seed(configuration.seed)
        torch.cuda.manual_seed(configuration.seed)
        torch.cuda.manual_seed_all(configuration.seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(configuration.local_rank)
        configuration.world_size = world_size
        rank = torch.distributed.get_rank()
        configuration.rank = rank

        if not os.path.exists(configuration.output_dir):
            os.makedirs(configuration.output_dir, exist_ok=True)

        logger = get_logger(os.path.join(configuration.output_dir, "log.txt"))

        if configuration.local_rank == 0:
            logger.info("Effective parameters:")
            for key in sorted(configuration.__dict__):
                logger.info("  <<< {}: {}".format(key, configuration.__dict__[key]))

        return configuration,logger

    def init_device(self,configuration,logger, local_rank):
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

        n_gpu = torch.cuda.device_count()
        logger.info("device: {} n_gpu: {}".format(device, n_gpu))
        configuration.n_gpu = n_gpu

        if configuration.batch_size % configuration.n_gpu != 0 or configuration.batch_size_val % configuration.n_gpu != 0:
            raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
                configuration.batch_size, configuration.n_gpu, configuration.batch_size_val, configuration.n_gpu))

        return device, n_gpu

    def init_model(self,configuration, device, n_gpu, local_rank):

        if configuration.init_model:
            model_state_dict = torch.load(configuration.init_model, map_location='cpu')
        else:
            model_state_dict = None

        # Prepare model
        cache_dir = configuration.cache_dir if configuration.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = X_VLNet.from_pretrained(cache_dir=cache_dir, state_dict=model_state_dict, task_config=configuration)

        model.to(device)

        return model

    def prep_optimizer(self,configuration, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

        if hasattr(model, 'module'):
            model = model.module

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
        no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

        decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
        decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

        no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
        no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

        weight_decay = 0.2
        optimizer_grouped_parameters = [
            {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': configuration.lr * coef_lr},
            {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
            {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': configuration.lr * coef_lr},
            {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
        ]

        self.scheduler = None
        optimizer = BertAdam(optimizer_grouped_parameters, lr=configuration.lr, warmup=configuration.warmup_proportion,
                            schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                            t_total=num_train_optimization_steps, weight_decay=weight_decay,
                            max_grad_norm=1.0)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                        output_device=local_rank, find_unused_parameters=True)

        return optimizer, self.scheduler, model



    