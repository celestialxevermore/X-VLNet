import torch 
import numpy as np 
import time 
#from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
from utilities import parallel_apply,run_on_single_gpu
from scipy.special import softmax

class Trainer():
    '''
        Trainer class
    '''
    def __init__(self,model,optimizer,configuration,train_dataloader,scheduler,device,train_sampler,logger):
        super(Trainer,self).__init__(model,optimizer)
        self.train_dataloader = train_dataloader
        self.configuration=configuration
        self.scheduler = scheduler
        self.model = model
        self.device = device
        self.train_sampler = train_sampler
        self.logger = logger
    
    def _train(self,epoch,global_step): #train_sampler
        
        torch.cuda.empty_cache()
        self.model.train()
        log_step = self.configuration.n_display
        start_time = time.time()
        total_loss = 0 
        
        for step, batch in enumerate(self.train_dataloader):
            if self.configuration.n_gpu == 1:
            # multi-gpu does scattering it-self
                batch = tuple(t.to(device=self.device, non_blocking=True) for t in batch)

            input_ids, input_mask, segment_ids, video, video_mask = batch
            loss = self.model(input_ids, segment_ids, input_mask, video, video_mask)

            if self.configuration.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if self.configuration.gradient_accumulation_steps > 1:
                loss = loss / self.configuration.gradient_accumulation_steps

            loss.backward()

            total_loss += float(loss)
            if (step + 1) % self.configuration.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                if self.scheduler is not None:
                    self.scheduler.step()  # Update learning rate schedule

                self.optimizer.step()
                self.optimizer.zero_grad()

                # https://github.com/openai/CLIP/issues/46
                if hasattr(self.model, 'module'):
                    torch.clamp_(self.model.module.clip.logit_scale.data, max=np.log(100))
                else:
                    torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

                global_step += 1
                if global_step % log_step == 0 and self.configuration.local_rank == 0:
                    self.logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                                self.configuration.epochs, step + 1,
                                len(self.train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))]),
                                float(loss),
                                (time.time() - start_time) / (log_step * self.configuration.gradient_accumulation_steps))
                    start_time = time.time()

        
        total_loss = total_loss / len(self.train_dataloader)
        return total_loss,global_step


class Tester():
    '''
        Test class
    '''
    def __init__(self,configuration,model,test_dataloader,device,logger):
        super(Tester,self).__init__()
        
        self.configuration = configuration
        self.test_dataloader = test_dataloader
        self.device = device
        self.model = model 
        self.logger = logger
        if hasattr(model,'module'):
            self.model = self.model.module.to(device)
        else:
            self.model = self.model.to(device)
            
            
    def _eval_epoch(self):
        
        multi_sentence_ = False
        cut_off_points_, sentence_num_, video_num_ = [], -1, -1
        if hasattr(self.test_dataloader.dataset, 'multi_sentence_per_video') \
                and self.test_dataloader.dataset.multi_sentence_per_video:
            multi_sentence_ = True
            cut_off_points_ = self.test_dataloader.dataset.cut_off_points
            sentence_num_ = self.test_dataloader.dataset.sentence_num
            video_num_ = self.test_dataloader.dataset.video_num
            cut_off_points_ = [itm - 1 for itm in cut_off_points_]

        if multi_sentence_:
            self.logger.warning("Eval under the multi-sentence per video clip setting.")
            self.logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

        self.model.eval() 
        with torch.no_grad():
            batch_list_t = []
            batch_list_v = []
            batch_sequence_output_list, batch_visual_output_list = [], []
            batch_seq_features_list=[]
            total_video_num = 0

            # ----------------------------
            # 1. cache the features
            # ----------------------------
            for bid, batch in enumerate(self.test_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, video, video_mask = batch

                if multi_sentence_:
                    # multi-sentences retrieval means: one clip has two or more descriptions.
                    b, *_t = video.shape
                    sequence_output,seq_features = self.model.get_sequence_output(input_ids, segment_ids, input_mask)
                    #print(f"sequence_output : {sequence_output.shape} seq_features : {seq_features.shape}")
                    batch_sequence_output_list.append(sequence_output)
                    batch_seq_features_list.append(seq_features)
                    batch_list_t.append((input_mask, segment_ids,))

                    s_, e_ = total_video_num, total_video_num + b
                    filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                    if len(filter_inds) > 0:
                        video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                        visual_output = self.model.get_visual_output(video, video_mask)
                        batch_visual_output_list.append(visual_output)
                        batch_list_v.append((video_mask,))
                    total_video_num += b
                else:
                    (sequence_output,seq_features), visual_output = self.model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

                    batch_sequence_output_list.append(sequence_output)
                    batch_seq_features_list.append(seq_features)
                    batch_list_t.append((input_mask, segment_ids,))
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))

                print("{}/{}\r".format(bid, len(self.test_dataloader)), end="")

            # ----------------------------------
            # 2. calculate the similarity
            # ----------------------------------
            if self.configuration.n_gpu > 1:
                device_ids = list(range(self.configuration.n_gpu))
                batch_list_t_splits = []
                batch_list_v_splits = []
                batch_t_output_splits = []
                batch_v_output_splits = []
                bacth_len = len(batch_list_t)
                split_len = (bacth_len + self.configuration.n_gpu - 1) // self.configuration.n_gpu
                for dev_id in device_ids:
                    s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                    if dev_id == 0:
                        batch_list_t_splits.append(batch_list_t[s_:e_])
                        batch_list_v_splits.append(batch_list_v)

                        batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                        batch_v_output_splits.append(batch_visual_output_list)
                    else:
                        devc = torch.device('cuda:{}'.format(str(dev_id)))
                        devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                        batch_list_t_splits.append(devc_batch_list)
                        devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                        batch_list_v_splits.append(devc_batch_list)

                        devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                        batch_t_output_splits.append(devc_batch_list)
                        devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                        batch_v_output_splits.append(devc_batch_list)

                parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                        batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in device_ids]
                parallel_outputs = parallel_apply(run_on_single_gpu, self.model, parameters_tuple_list, device_ids)
                sim_matrix = []
                for idx in range(len(parallel_outputs)):
                    sim_matrix += parallel_outputs[idx]
                sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            else:
                sim_matrix = run_on_single_gpu(self.model, batch_list_t, batch_list_v, batch_sequence_output_list,batch_seq_features_list, batch_visual_output_list)
                sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
                
            sim_matrix_dsl = sim_matrix * softmax(sim_matrix,axis=0)
            sim_matrix_dsl_ = sim_matrix * softmax(sim_matrix,axis=1)
            
            
        if multi_sentence_:
            self.logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
            cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
            max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
            sim_matrix_new = []
            for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
                sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                    np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
            sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
            self.logger.info("after reshape, sim matrix size: {} x {} x {}".
                        format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

            tv_metrics = tensor_text_to_video_metrics(sim_matrix)
            vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
            
            # dsl 
            cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
            max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
            sim_matrix_new = []
            sim_matrix_new_= []
            for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
                sim_matrix_new.append(np.concatenate((sim_matrix_dsl[s_:e_],
                                                    np.full((max_length-e_+s_, sim_matrix_dsl.shape[1]), -np.inf)), axis=0))
                sim_matrix_new_.append(np.concatenate((sim_matrix_dsl_[s_:e_],
                                                    np.full((max_length-e_+s_, sim_matrix_dsl_.shape[1]), -np.inf)), axis=0))

            sim_matrix_dsl = np.stack(tuple(sim_matrix_new), axis=0)
            sim_matrix_dsl_ = np.stack(tuple(sim_matrix_new_),axis=0)
            self.logger.info("after reshape, sim matrix size: {} x {} x {}".
                        format(sim_matrix_dsl.shape[0], sim_matrix_dsl.shape[1], sim_matrix_dsl.shape[2]))
            self.logger.info("after reshape, sim matrix size: {} x {} x {}".
                        format(sim_matrix_dsl_.shape[0], sim_matrix_dsl_.shape[1], sim_matrix_dsl_.shape[2]))

            dsl_tv_metrics = tensor_text_to_video_metrics(sim_matrix_dsl)
            dsl_vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix_dsl))
            dsl_tv_metrics_ = tensor_text_to_video_metrics(sim_matrix_dsl_)
            dsl_vt_metrics_ = compute_metrics(tensor_video_to_text_sim(sim_matrix_dsl_))
            
        else:
            self.logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
            tv_metrics = compute_metrics(sim_matrix)
            vt_metrics = compute_metrics(sim_matrix.T)
            dsl_tv_metrics = compute_metrics(sim_matrix_dsl)
            dsl_vt_metrics = compute_metrics(sim_matrix_dsl.T)
            dsl_tv_metrics_ = compute_metrics(sim_matrix_dsl_)
            dsl_vt_metrics_ = compute_metrics(sim_matrix_dsl_.T)
            
            self.logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

        # dsl output
        self.logger.info("------------------------------------------------------------")
        self.logger.info("DSL Text-to-Video:")
        self.logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(dsl_tv_metrics['R1'], dsl_tv_metrics['R5'], dsl_tv_metrics['R10'], dsl_tv_metrics['MR'], dsl_tv_metrics['MeanR']))
        self.logger.info("DSL Video-to-Text:")
        self.logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(dsl_vt_metrics_['R1'], dsl_vt_metrics_['R5'], dsl_vt_metrics_['R10'], dsl_vt_metrics_['MR'], dsl_vt_metrics_['MeanR']))

        self.logger.info("------------------------------------------------------------")
        
        self.logger.info("Text-to-Video:")
        self.logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
        self.logger.info("Video-to-Text:")
        self.logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

        R1 = tv_metrics['R1']
        return R1
        
        
        