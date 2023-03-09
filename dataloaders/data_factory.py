import torch 
from config.configuration import configuration
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader
from dataloaders.dataloader_msvd_retrieval import MSVD_DataLoader
from dataloaders.dataloader_lsmdc_retrieval import LSMDC_DataLoader
from dataloaders.dataloader_activitynet_retrieval import ActivityNet_DataLoader
from dataloaders.dataloader_didemo_retrieval import DiDeMo_DataLoader
from torch.utils.data import DataLoader

#train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            
class DataDirector:
    
    @staticmethod
    def get_dataloader(configuration : configuration, tokenizer, split_type='train'):
        if split_type =='train':
            
            if configuration.dataset_name == "MSRVTT":
                #global dataset 
                msrvtt_dataset = MSRVTT_TrainDataLoader(
                    csv_path = configuration.msrvtt_train_csv,
                    json_path = configuration.msrvtt_data_path,
                    features_path = configuration.msrvtt_features_path,
                    max_words = configuration.max_words,
                    feature_framerate=configuration.feature_framerate,
                    tokenizer=tokenizer,
                    unfold_sentences=configuration.expand_msrvtt_sentences,
                    frame_order=configuration.train_frame_order,
                    slice_framepos=configuration.slice_framepos,
                )
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                dataloader = DataLoader(
                    msrvtt_dataset,
                    batch_size = configuration.batch_size // configuration.n_gpu,
                    num_workers = configuration.num_thread_reader, 
                    pin_memory=False,
                    shuffle=(train_sampler is None),
                    sampler = train_sampler,
                    drop_last=True,
                )
                return dataloader,len(msrvtt_dataset), train_sampler
            
            if configuration.dataset_name == "MSVD":
                msvd_dataset = MSVD_DataLoader(
                    subset = "train",
                    data_path=configuration.msvd_data_path,
                    features_path=configuration.msvd_features_path,
                    max_words=configuration.max_words,
                    feature_framerate=configuration.feature_framerate,
                    tokenizer=tokenizer,
                    max_frames=configuration.max_frames,
                    frame_order=configuration.train_frame_order,
                    slice_framepos=configuration.slice_framepos,
                )
                train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
                dataloader = DataLoader(
                    msvd_dataset,
                    batch_size=configuration.batch_size // configuration.n_gpu,
                    num_workers=configuration.num_thread_reader,
                    pin_memory=False,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                    drop_last=True,
                )
                return dataloader, len(msvd_dataset), train_sampler
            
            if configuration.dataset_name == "LSMDC":
                lsmdc_dataset = LSMDC_DataLoader(
                    subset="train",
                    data_path=configuration.data_path,
                    features_path=configuration.features_path,
                    max_words=configuration.max_words,
                    feature_framerate=configuration.feature_framerate,
                    tokenizer=tokenizer,
                    max_frames=configuration.max_frames,
                    frame_order=configuration.train_frame_order,
                    slice_framepos=configuration.slice_framepos,
                )
                train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
                
                dataloader = DataLoader(
                    lsmdc_dataset,
                    batch_size=configuration.batch_size // configuration.n_gpu,
                    num_workers=configuration.num_thread_reader,
                    pin_memory=False,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                    drop_last=True,
                )
                return dataloader, len(lsmdc_dataset), train_sampler
            
            if configuration.dataset_name =="ActivityNet":
                dataset = ActivityNet_DataLoader()
                dataloader = DataLoader()
                
            if configuration.dataset_name == "DiDemo":
                dataset = DiDeMo_DataLoader()
                dataloader = DataLoader()
                
            
                
        elif split_type =='test':
            if configuration.dataset_name == "MSRVTT":
                msrvtt_testset = MSRVTT_DataLoader(
                    csv_path=configuration.msrvtt_val_csv,
                    features_path=configuration.msrvtt_features_path,
                    max_words=configuration.max_words,
                    feature_framerate=configuration.feature_framerate,
                    tokenizer=tokenizer,
                    max_frames=configuration.max_frames,
                    frame_order=configuration.eval_frame_order,
                    slice_framepos=configuration.slice_framepos,
                )
                dataloader_msrvtt = DataLoader(
                    msrvtt_testset,
                    batch_size=configuration.batch_size_val,
                    num_workers=configuration.num_thread_reader,
                    shuffle=False,
                    drop_last=False,
                )
                return dataloader_msrvtt,len(msrvtt_testset)
            
            if configuration.dataset_name == "MSVD":
                msvd_testset = MSVD_DataLoader(
                    subset=split_type,
                    data_path=configuration.msvd_data_path,
                    features_path=configuration.msvd_features_path,
                    max_words=configuration.max_words,
                    feature_framerate=configuration.feature_framerate,
                    tokenizer=tokenizer,
                    max_frames=configuration.max_frames,
                    frame_order=configuration.eval_frame_order,
                    slice_framepos=configuration.slice_framepos,
                )
                dataloader_msrvtt = DataLoader(
                    msvd_testset,
                    batch_size=configuration.batch_size_val,
                    num_workers=configuration.num_thread_reader,
                    shuffle=False,
                    drop_last=False,
                )
                return dataloader_msrvtt, len(msvd_testset)
                
            if configuration.dataset_name == "LSMDC":
                lsmdc_dataloader = DataLoader()
                
            if configuration.dataset_name =="ActivityNet":
                activity_dataloader = DataLoader()
                
            if configuration.dataset_name == "DiDemo":
                didemo_dataloader = DataLoader()
            
        