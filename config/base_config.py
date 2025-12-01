from abc import abstractmethod,ABC
class baseconfig(ABC):
    def __init__(self):
        args = self.parse_args()
        
        
        self.dataset_name = args.dataset_name
        
        self.msrvtt_train_file = args.msrvtt_train_file
        self.msrvtt_train_csv = '/home/key2317/X-VLNet/datasets/MSRVTT/msrvtt_data/MSRVTT_train.9k.csv'
        self.msrvtt_val_csv = '/home/key2317/X-VLNet/datasets/MSRVTT/msrvtt_data/MSRVTT_JSFUSION_test.csv'
        self.msrvtt_features_path = '/home/key2317/X-VLNet/datasets/MSRVTT/videos/compressed'
        
        self.msvd_data_path = '/home/key2317/X-VLNet/datasets/msvd_data'
        self.msvd_features_path = '/home/key2317/X-VLNet/datasets/msvd_data/compressed'
        
        self.n_gpu = args.n_gpu
        self.n_display = 50 
        self.cross_model = "cross-base"
        self.max_frames = 12
        self.max_words = 32 
        self.output_dir = args.output_dir
        self.warmup_proportion = 0.1 
        self.weight_decay = 0.2 
        self.lr = 1e-4
        self.seed = 42
        self.feature_framerate = 1 
        self.gradient_accumulation_steps = 1 
        self.coef_lr = 1e-3 
        self.slice_framepos = 0 # choices 0,1,2
        self.eval_frame_order = 0
        self.num_thred_reader = 4
        self.expand_msrvtt_sentences = "" # 뭐지 아직 모르겠음
        self.train_frame_order = 0 # choices [0,1,2]
        self.best_score = 0.00001
        self.best_output_model_file = "None"
        self.resumed_epoch = 0 
        self.global_step=0

    @abstractmethod
    def parse_args(self):
        raise NotImplementedError