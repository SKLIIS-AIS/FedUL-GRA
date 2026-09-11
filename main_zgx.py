import os
import json
from pathlib import Path
from utils.args import parser_args
from utils.datasets import *
import copy
import random
import dill
import datetime
from tqdm import tqdm
import numpy as np
import math
from scipy import spatial
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import time
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
import models as models

# from opacus.validators import ModuleValidator
# from opacus import PrivacyEngine
from experiments.base import Experiment
from experiments.trainer_private import TrainerPrivate, TesterPrivate
from dataset import CIFAR10, CIFAR100
# import wandb


def experiment_output_dir(root, model_name, dataset):
    output_root = Path(root).expanduser()
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    return output_root / model_name / dataset

class FederatedLearning(Experiment):
    """
    Perform federated learning
    """
    def __init__(self, args):
        compatible_models = {
            'mnist': {'lenet'},
            'cifar10': {'alexnet'},
            'cifar100': {'resnet18', 'resnet34'},
            'celeba': {'alexnet'},
        }
        if args.model_name not in compatible_models[args.dataset]:
            raise ValueError(
                'Model {} is not supported for {}. Choose one of {}.'.format(
                    args.model_name,
                    args.dataset,
                    ', '.join(sorted(compatible_models[args.dataset])),
                )
            )
        if not 0 <= args.num_ul_users <= args.num_users:
            raise ValueError('--num_ul_users must be between 0 and --num_users.')
        super().__init__(args) # define many self attributes from args
        self.criterion = torch.nn.CrossEntropyLoss()
        self.in_channels = 3
        self.optim=args.optim
        self.proportion=args.proportion
        if 'class' in args.ul_mode:
            self.proportion=1.0
        self.dp = args.dp
        self.sigma = args.sigma
        self.cosine_attack =args.cosine_attack  
        self.sigma_sgd = args.sigma_sgd
        self.grad_norm=args.grad_norm
        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.data_root = args.data_root

        self.ul_mode=args.ul_mode
        if 'client' in self.ul_mode:
            raise ValueError('Client unlearning has been removed from the open-source experiment interface.')
        self.ul_class_id=args.ul_class_id
        self.ul_clients=list(np.random.choice([i for i in range(args.num_users)], args.num_ul_users, replace=False))
 
        print('==> Preparing data...')
        self.train_set, self.test_set, self.ul_test_set, self.dict_users, self.train_idxs, self.val_idxs, self.private_samples_idxs, self.final_train_idxs = get_data(dataset=self.dataset,
                                                        data_root = self.data_root,
                                                        proportion=self.proportion,
                                                        iid = self.iid,
                                                        num_users = self.num_users,
                                                        UL_clients=self.ul_clients,
                                                        data_aug=self.args.data_augment,
                                                        noniid_beta=self.args.beta,
                                                        samples_per_user=args.samples_per_user,
                                                        ul_mode=self.ul_mode,
                                                        ul_class_id=self.ul_class_id,
                                                        celeba_attr=args.celeba_attr,
                                                        celeba_target_mode=args.celeba_target_mode
                                                        )

        if self.args.dataset == 'cifar10':
            self.num_classes = 10
            self.in_channels=3
            # self.dataset_size = 60000
        elif self.args.dataset == 'cifar100':
            self.num_classes = 100
            self.in_channels=3
            # self.dataset_size = 60000
        elif self.args.dataset == 'mnist':
            self.num_classes = 10
            self.in_channels=1
            # self.dataset_size = 60000
        elif self.args.dataset == 'celeba':
            self.num_classes = len(self.train_set.classes)
            self.in_channels=3

        self.train_idxs_cos=[]
        self.testset_idx=(50000+np.arange(10000)).astype(int) # 最后10000样本的作为test set
        self.testset_idx_cos=(50000+np.arange(1000)).astype(int)

        print('==> Preparing model...')

        self.logs = {'train_acc': [], 'train_sign_acc':[], 'train_loss': [],
                     'val_acc': [], 'val_loss': [],
                     'test_acc': [], 'test_loss': [],
                     'keys':[],

                     'best_test_acc': -np.inf,
                     'best_model': [],
                     'local_loss': [],
                     }

        self.construct_model()
        
        self.w_t = copy.deepcopy(self.model.state_dict())

        self.trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma,self.num_classes,'none')
        self.trainer_ul=TrainerPrivate(self.model_ul, self.device, self.dp, self.sigma,self.num_classes,self.ul_mode)
        self.tester = TesterPrivate(self.model, self.device)

              
    def construct_model(self):

        # model = models.__dict__[self.args.model_name](num_classes=self.num_classes*2)
        model = models.__dict__[self.args.model_name](num_classes=self.num_classes,in_channels=self.in_channels)
        # if not ModuleValidator.is_valid(model):
        #     model = ModuleValidator.fix(model)
        #model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        self.learning_model = self.model
        # ========== 【核心修改开始：创建遗忘模型】==========
        # 判断是否要使用你创建的“独立双分支”模型
        if hasattr(self.args, 'model_variant') and self.args.model_variant == 'independent':
            # 1. 打印提示，方便我们确认
            print("*** [状态] 正在构建 -> 独立双分支模型 (LeNet_UL_Independent) ***")
            # 2. 直接使用你刚刚在 lenet_ul.py 里定义的新类来创建模型
            independent_model_name = self.args.model_name + '_ul_independent'
            if independent_model_name not in models.__dict__:
                raise ValueError('Independent variant is not implemented for {}'.format(self.args.model_name))
            print("*** [Status] Building independent two-branch model: {} ***".format(independent_model_name))
            model_ul = models.__dict__[independent_model_name](num_classes=self.num_classes, in_channels=self.in_channels)
            if hasattr(model_ul, 'set_ablation'):
                model_ul.set_ablation(
                    disable_cne=getattr(self.args, 'disable_cne', False),
                    disable_gra=getattr(self.args, 'disable_gra', False),
                )
                print('[FedUL Ablation] CNE {}, GRA {}'.format(
                    'off' if getattr(self.args, 'disable_cne', False) else 'on',
                    'off' if getattr(self.args, 'disable_gra', False) else 'on',
                ))
        else:
            # 否则，使用原来的方式创建原始模型
            print("*** [状态] 正在构建 -> 原始FedAU模型 ***")
            model_ul = models.__dict__[self.args.model_name+'_ul'](num_classes=self.num_classes,in_channels=self.in_channels)
        # ========== 【核心修改结束】==========
        self.model_ul=model_ul.to(self.device)
        
        # torch.backends.cudnn.benchmark = True
        print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

    def _load_independent_unlearn_model(self, ul_state_dicts, mode):
        combined_state_dict = copy.deepcopy(self.w_t)
        aux_keys = [
            key for key in self.model_ul.state_dict().keys()
            if '_ul' in key or key.startswith('classifier_ul')
        ]

        for key in aux_keys:
            combined_state_dict[key] = sum(ul_state_dicts[idx][key] for idx in self.ul_clients) / len(self.ul_clients)

        self.model_ul.load_state_dict(combined_state_dict, strict=False)
        if hasattr(self.model_ul, 'set_ablation'):
            self.model_ul.set_ablation(
                disable_cne=getattr(self.args, 'disable_cne', False),
                disable_gra=getattr(self.args, 'disable_gra', False),
            )
        if hasattr(self.model_ul, 'set_unlearn_rule'):
            if mode == 'class':
                self.model_ul.set_unlearn_rule(
                    mode='class',
                    alpha=self.args.ul_samples_alpha,
                    beta=1.0,
                    target_class=self.ul_class_id,
                )
            else:
                self.model_ul.set_unlearn_rule(mode='samples', alpha=self.args.ul_samples_alpha, beta=1.0)

        self.model = self.model_ul
        self.trainer.model = self.model

    def _calibrate_independent_negative_expert(self, forget_ldr, retain_ldr, mode):
        if getattr(self.args, 'disable_cne', False):
            return {}
        if not hasattr(self.model, 'set_negative_expert'):
            return {}
        target_class = self.ul_class_id if mode == 'class' else None
        calibration_mode = 'class' if mode == 'class' else 'samples'
        calibration = self.trainer.calibrate_negative_expert(
            forget_ldr,
            retain_dataloader=retain_ldr,
            mode=calibration_mode,
            target_class=target_class,
        )
        if calibration:
            retain_rate = calibration.get('negative_retain_trigger_rate')
            retain_rate_text = 'None' if retain_rate is None else '{:.4f}'.format(retain_rate)
            print('[FedUL CNE] threshold {:.6f}, target {}, forget trigger {:.4f}, retain trigger {}'.format(
                calibration.get('negative_threshold', 0.0),
                calibration.get('negative_target_class', -1),
                calibration.get('negative_forget_trigger_rate', 0.0),
                retain_rate_text,
            ))
        return calibration

    def _should_run_independent_recovery(self):
        return (
            getattr(self.args, 'model_variant', 'original') == 'independent'
            and self.ul_mode.startswith('ul_')
            and getattr(self.args, 'recovery_epochs', 0) > 0
        )

    def _retained_recovery_indices(self):
        retained_idxs = set(self.final_train_idxs) - set(self.private_samples_idxs)
        return sorted(retained_idxs)

    def _blend_state_dict(self, pre_state_dict, candidate_state_dict, blend):
        blended_state_dict = {}
        for name, pre_value in pre_state_dict.items():
            candidate_value = candidate_state_dict[name]
            if torch.is_tensor(pre_value) and torch.is_floating_point(pre_value):
                blended_state_dict[name] = pre_value + (candidate_value - pre_value) * blend
            else:
                blended_state_dict[name] = copy.deepcopy(candidate_value if blend >= 0.5 else pre_value)
        return blended_state_dict

    def _run_independent_recovery(
        self,
        recovery_ldr,
        val_ldr,
        ul_ldr,
        pre_val_acc,
        pre_ul_effect,
        calibration_forget_ldr=None,
        calibration_retain_ldr=None,
        calibration_mode='samples',
    ):
        recovery_lr = self.args.recovery_lr if self.args.recovery_lr is not None else self.lr
        recovery_epochs = int(self.args.recovery_epochs)
        freeze_aux = bool(self.args.recovery_freeze_aux)
        objective = getattr(self.args, 'recovery_objective', 'unlearn')
        uses_sgr = hasattr(self.model, 'retain_adapter')
        train_objective = 'unlearn' if uses_sgr else objective
        accept_ul_drop = max(0.0, float(self.args.recovery_accept_ul_drop))

        pre_state_dict = copy.deepcopy(self.model.state_dict())
        start = time.time()
        _state_dict, recovery_loss, recovery_train_acc = self.trainer._retained_recovery_update(
            recovery_ldr,
            recovery_epochs,
            recovery_lr,
            self.optim,
            freeze_aux=freeze_aux,
            objective=train_objective,
        )
        recovery_time = time.time() - start

        candidate_calibration = self._calibrate_independent_negative_expert(
            calibration_forget_ldr,
            calibration_retain_ldr,
            calibration_mode,
        )
        candidate_state_dict = copy.deepcopy(self.model.state_dict())
        candidate_val_loss, candidate_val_acc = self.trainer.test(val_ldr)
        candidate_ul_loss, candidate_ul_effect = self.trainer.ul_test(ul_ldr)

        best_recovery = None
        blend_candidates = [
            1.0, 0.95, 0.9, 0.85, 0.8,
            0.75, 0.7, 0.65, 0.6, 0.55,
            0.5, 0.45, 0.4, 0.35, 0.3,
            0.25, 0.2, 0.15, 0.1, 0.075,
            0.05, 0.025, 0.01,
        ]
        for blend in blend_candidates:
            if blend == 1.0:
                blended_state_dict = candidate_state_dict
            else:
                blended_state_dict = self._blend_state_dict(pre_state_dict, candidate_state_dict, blend)
            self.model.load_state_dict(blended_state_dict)
            self.trainer.model = self.model
            blend_calibration = self._calibrate_independent_negative_expert(
                calibration_forget_ldr,
                calibration_retain_ldr,
                calibration_mode,
            )
            calibrated_state_dict = copy.deepcopy(self.model.state_dict())
            blend_val_loss, blend_val_acc = self.trainer.test(val_ldr)
            blend_ul_loss, blend_ul_effect = self.trainer.ul_test(ul_ldr)
            accepted = (
                blend_val_acc >= pre_val_acc
                and blend_ul_effect >= pre_ul_effect - accept_ul_drop
            )
            if not accepted:
                continue
            if (
                best_recovery is None
                or blend_val_acc > best_recovery['final_val_acc']
                or (
                    blend_val_acc == best_recovery['final_val_acc']
                    and blend_ul_effect > best_recovery['final_ul_effect']
                )
            ):
                best_recovery = {
                    'blend': blend,
                    'state_dict': calibrated_state_dict,
                    'final_val_loss': blend_val_loss,
                    'final_val_acc': blend_val_acc,
                    'final_ul_loss': blend_ul_loss,
                    'final_ul_effect': blend_ul_effect,
                    'calibration': blend_calibration,
                }

        if best_recovery is None:
            self.model.load_state_dict(pre_state_dict)
            self.trainer.model = self.model
            final_calibration = self._calibrate_independent_negative_expert(
                calibration_forget_ldr,
                calibration_retain_ldr,
                calibration_mode,
            )
            final_val_loss, final_val_acc = self.trainer.test(val_ldr)
            final_ul_loss, final_ul_effect = self.trainer.ul_test(ul_ldr)
            accepted = False
            selected_blend = 0.0
        else:
            self.model.load_state_dict(best_recovery['state_dict'])
            self.trainer.model = self.model
            final_calibration = best_recovery.get('calibration', candidate_calibration)
            final_val_loss = best_recovery['final_val_loss']
            final_val_acc = best_recovery['final_val_acc']
            final_ul_loss = best_recovery['final_ul_loss']
            final_ul_effect = best_recovery['final_ul_effect']
            accepted = True
            selected_blend = best_recovery['blend']

        retain_gate_mean = self.trainer.gra_gate_mean(recovery_ldr)
        ul_gate_mean = self.trainer.gra_gate_mean(ul_ldr)

        return {
            'accepted': accepted,
            'time': recovery_time,
            'lr': recovery_lr,
            'epochs': recovery_epochs,
            'objective': 'sgr' if uses_sgr else objective,
            'freeze_aux': freeze_aux,
            'blend': selected_blend,
            'retain_gate_mean': retain_gate_mean,
            'ul_gate_mean': ul_gate_mean,
            'loss': recovery_loss,
            'train_acc': recovery_train_acc,
            'candidate_val_loss': candidate_val_loss,
            'candidate_val_acc': candidate_val_acc,
            'candidate_ul_loss': candidate_ul_loss,
            'candidate_ul_effect': candidate_ul_effect,
            'final_val_loss': final_val_loss,
            'final_val_acc': final_val_acc,
            'final_ul_loss': final_ul_loss,
            'final_ul_effect': final_ul_effect,
            'calibration': final_calibration,
        }

    def _empty_model_update(self):
        update = {}
        for param_name, param_value in self.model.state_dict().items():
            if "weight" in param_name or "bias" in param_name:
                update[param_name] = torch.zeros_like(param_value).to(self.device)
        return update

    def _amnesiac_start_epoch(self):
        override = getattr(self.args, 'amnesiac_start_epoch', None)
        if override is not None and override >= 0:
            return min(int(override), self.epochs)
        if 'samples' in self.ul_mode:
            # Keep sample-level Amnesiac aligned across MNIST, CIFAR-10, and CelebA.
            return max(0, self.epochs - 10)
        if 'class' in self.ul_mode:
            return max(0, int(0.5 * self.epochs))
        return 0

    def _sum_amnesiac_updates(self, update_list, start_epoch, end_epoch):
        update_sum = self._empty_model_update()
        end_epoch = min(end_epoch, len(update_list) - 1)
        if start_epoch > end_epoch:
            return update_sum
        for epoch_update in update_list[start_epoch:end_epoch + 1]:
            for param_name in update_sum:
                update_sum[param_name] += epoch_update[param_name]
        return update_sum


    def train(self):
        # these dataloader would only be used in calculating accuracy and loss
        train_ldr = DataLoader(DatasetSplit(self.train_set,self.train_set.final_train_list), batch_size=self.batch_size *2, shuffle=False, num_workers=4)
        val_ldr = DataLoader(self.test_set, batch_size=self.batch_size *2, shuffle=False, num_workers=4)
        test_ldr = DataLoader(self.test_set, batch_size=self.batch_size , shuffle=False, num_workers=0)
        ul_ldr = DataLoader(self.ul_test_set, batch_size=self.batch_size *2, shuffle=False, num_workers=4)
        recovery_ldr = None
        negative_forget_ldr = None
        negative_retain_ldr = None
        recovery_retained_idxs = []
        uses_independent_negative_expert = (
            getattr(self.args, 'model_variant', 'original') == 'independent'
            and self.ul_mode.startswith('ul_')
        )
        if uses_independent_negative_expert:
            recovery_retained_idxs = self._retained_recovery_indices()
            if len(self.private_samples_idxs) > 0:
                negative_forget_ldr = DataLoader(DatasetSplit(self.train_set, self.private_samples_idxs),
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
            if len(recovery_retained_idxs) > 0:
                negative_retain_ldr = DataLoader(DatasetSplit(self.train_set, recovery_retained_idxs),
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
                if self._should_run_independent_recovery():
                    recovery_ldr = DataLoader(DatasetSplit(self.train_set, recovery_retained_idxs),
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              num_workers=2)
                    print('[FedUL Recovery] retained samples:', len(recovery_retained_idxs))
            else:
                if self._should_run_independent_recovery():
                    print('[FedUL Recovery] skipped: retained set is empty')

        if args.num_ul_users == 0:
            if not args.external_ul_dataloader:
                raise ValueError(
                    '--external_ul_dataloader is required when --num_ul_users is 0.'
                )
            with open(args.external_ul_dataloader, 'rb') as f:
                ldrs=dill.load(f)
            ul_ldr=ldrs['ul_ldr']

            
        # 不区分iid 和 non iid --unlearn code上

        # torch.backends.cudnn.benchmark = True

        local_train_ldrs = []
        if args.iid:
            for i in range(self.num_users):
                local_train_ldr = DataLoader(DatasetSplit(self.train_set, self.dict_users[i]), batch_size = self.batch_size,
                                                shuffle=True, num_workers=2)
                local_train_ldrs.append(local_train_ldr)

        else:  #copy原版的
            for i in range(self.num_users):
                
                local_train_ldr = DataLoader(DatasetSplit(self.train_set, self.dict_users[i]), batch_size = self.batch_size,
                                            shuffle=True, num_workers=2)
                local_train_ldrs.append(local_train_ldr) 

        # 保存dataloader
        today = datetime.date.today()
        save_dir = experiment_output_dir(
            self.args.log_folder_name, self.args.model_name, self.args.dataset
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        dataloader_pkl_name = "_".join(
                    ['FedUL_dataloader', f's{self.args.seed}', str(args.num_users), str(args.batch_size),str(args.lr), str(args.iid), f'{today.year}_{today.month}_{today.day}'])
        dataloader_pkl_name = save_dir / (dataloader_pkl_name + '.pkl')
        print("dataloader_pkl_name:",dataloader_pkl_name)

        dataloader_save_dict={'train_ldr':train_ldr,
                    "val_ldr":val_ldr,
                    "ul_ldr":ul_ldr,
                    "dict_users":self.dict_users,
                    "local_train_ldrs":local_train_ldrs,
                    "ul_clients":self.ul_clients
                    }
        if recovery_retained_idxs:
            dataloader_save_dict["recovery_retained_idxs"] = recovery_retained_idxs
        with open(dataloader_pkl_name,'wb') as f:
            dill.dump(dataloader_save_dict, f)

        total_time=0
        sum_learn_time=0
        sum_unlearn_time=0
        sum_recovery_time=0
        final_unlearn_state_dict=None
        time_mark=str(time.strftime("%Y_%m_%d_%H%M%S", time.localtime()))
        file_name = "_".join(
                ['FedUL', str(args.ul_mode), f's{args.seed}',str(args.num_users), str(args.batch_size),str(args.lr), str(args.lr_up), str(args.iid), time_mark])
        log_dir = experiment_output_dir(args.log_folder_name, args.model_name, args.dataset)
        log_dir.mkdir(parents=True, exist_ok=True)
        #fn=log_dir+'/'+file_name+'.txt'
        fn = log_dir / (file_name + '.log')

        print("training log saved in:",fn)

        lr_0=self.lr
        ul_state_dicts={}
        print('UL_clients:',self.ul_clients)
        for i in self.ul_clients:
            # print(i)
            ul_state_dicts[i]=copy.deepcopy(self.model_ul.state_dict())

        if 'amnesiac_ul' in self.ul_mode:
            update_list=[]
            update_epochs={} # 形状和模型参数字典相同
            for param_tensor in self.model.state_dict():
                if "weight" in param_tensor or "bias" in param_tensor:
                    update_epochs[param_tensor] = torch.zeros_like(self.model.state_dict()[param_tensor]).to(self.device)
            # update_list用于保存各epoch的private update
            update_template=update_epochs
            # 初始化private update的累计量为0
            update_sum=copy.deepcopy(update_template)

        for epoch in range(self.epochs):

            if 'amnesiac_ul' in self.ul_mode:
                global_update_epoch=copy.deepcopy(update_template)

            global_state_dict=copy.deepcopy(self.model.state_dict())

            if self.sampling_type == 'uniform':
                self.m = max(int(self.frac * self.num_users), 1)
                idxs_users = np.random.choice(range(self.num_users), self.m, replace=False)
                idxs_users=list(idxs_users)
                print(idxs_users)

            local_ws, local_losses,= [], []

            start = time.time()
            training_time_normal = 0.0
            training_time_ul = 0.0
            '''
            Start training. 
            For each client in each round, first determine whether it is ul_client, 
            and then determine whether ul_mode is retrain, 
            so as to select the training method.
            '''
            for idx in tqdm(idxs_users, desc='Epoch:%d, lr:%f' % (self.epochs, self.lr)):
 
                if (idx in self.ul_clients) ==False:
                    
                    # print(idx,"True1000000")
                    self.model.load_state_dict(global_state_dict) # 还原 global model
                    start_normal = time.time()
                    local_w, local_loss= self.trainer._local_update(local_train_ldrs[idx], self.local_ep, self.lr, self.optim) 
                    end_normal=time.time()
                    training_time_normal += end_normal - start_normal
                    local_ws.append(copy.deepcopy(local_w))
                    local_losses.append(local_loss)
                    
                else:
                    # start_ul=time.time()
                    # print(idx,"False1000000")
                    if self.ul_mode.startswith('ul_'):
                        
                        # print("ul-idx:",idx)
                        self.model_ul.load_state_dict(ul_state_dicts[idx])
                        # ul_model除W2外替换为global model的参数
                        self.model_ul.load_state_dict(global_state_dict,strict=False)
                        # ul_client时， W2基于W1训练：
                        if self.ul_mode=='ul_samples_backdoor' and self.dataset =='cifar100':
                            # print('Learn based on W1...')
                            gamma=args.ul_client_gamma
                            temp_state_dict=copy.deepcopy(self.model_ul.state_dict())
                            temp_state_dict['classifier_ul.weight']= (1-gamma) * temp_state_dict['classifier.weight'] + gamma * temp_state_dict['classifier_ul.weight']
                            temp_state_dict['classifier_ul.bias']= (1-gamma) * temp_state_dict['classifier.bias'] + gamma * temp_state_dict['classifier_ul.bias']
                            self.model_ul.load_state_dict(temp_state_dict)
                        # 参数替换完毕，开始训练
                        start_ul=time.time()
                        local_w_ul, local_loss, classify_loss, normalize_loss= self.trainer_ul._local_update_ul(local_train_ldrs[idx], self.local_ep, self.lr, self.optim,self.ul_class_id) 
                        end_ul=time.time()
                        training_time_ul += end_ul - start_ul
                        # 本次ul_model结果保存（用于下轮更新W2）
                        ul_state_dicts[idx]=copy.deepcopy(local_w_ul)
                        # 提取W1 (全局模型加载W1，保存到待avg列表中)
                        if self.args.model_variant == 'independent':
                            # FedUL-MG-PRO-CNE: keep auxiliary unlearning updates out of the global branch.
                            local_ws.append(copy.deepcopy(global_state_dict))
                        else:
                            self.model.load_state_dict(local_w_ul,strict=False)
                            local_ws.append(copy.deepcopy(self.model.state_dict()))

                        # class_loss,class_acc=self.trainer.test(ul_ldr)
                        # print('**** local class loss: {:.4f}  local class acc: {:.4f}****'.format(class_loss,class_acc))
                        

                    elif 'retrain' in self.ul_mode:   # retrain scheme
                        print('retrain')
                        self.model.load_state_dict(global_state_dict)
                        # retrain训练，除sample数量减少外（训练过程对ul sample剔除），过程与正常客户端相同
                        start_ul = time.time()
                        local_w, local_loss= self.trainer._local_update(local_train_ldrs[idx], self.local_ep, self.lr, self.optim,self.ul_mode, self.ul_class_id ) 
                        end_ul = time.time()
                        training_time_ul += end_ul - start_ul
                        local_ws.append(copy.deepcopy(local_w))
                        local_losses.append(local_loss)
                    elif 'amnesiac' in self.ul_mode:
                        self.model.load_state_dict(global_state_dict)
                        # 根据敏感batch 计算对应update之和
                        # print('amnesiac learning')
                        start_ul = time.time()
                        local_w, local_loss, local_update_epoch= self.trainer._local_update(local_train_ldrs[idx], self.local_ep, self.lr, self.optim,self.ul_mode, self.ul_class_id ) 
                        end_ul = time.time()
                        training_time_ul += end_ul - start_ul
                        local_ws.append(copy.deepcopy(local_w))
                        local_losses.append(local_loss)

                        for key in local_update_epoch:
                            global_update_epoch[key]+=local_update_epoch[key] * 1/self.num_users  

            if 'amnesiac_ul' in self.ul_mode:
                update_list.append(copy.deepcopy(global_update_epoch))

            if self.optim=="sgd":
                if self.args.lr_up=='common':
                    if self.epochs>101:
                        self.lr = self.lr * 0.99
                    else:
                        self.lr = self.lr * 0.98
                elif self.args.lr_up =='milestone':
                    if args.epochs==500:
                        milestones=[275,400]
                    elif args.epochs==300:
                        milestones=[150,225]
                    else:
                        milestones=[int(args.epochs/3), int(args.epochs/3)*2]
                    if epoch in milestones:
                        self.lr *= 0.1
                else:
                    self.lr=lr_0 * (1 + math.cos(math.pi * epoch/ self.args.epochs)) / 2 
            else:
                pass

            client_weights = []

            for i in range(self.num_users):
                if args.iid:
                    client_weights.append(1/self.num_users)
                    if i in self.ul_clients:
                        client_weights[i]=0.1
                    
                else:
                    client_weight = len(DatasetSplit(self.train_set, self.dict_users[i]))/len(self.train_set)
                    client_weights.append(client_weight)
                    # print('client {}, avg_weight {}'.format(i,client_weight))
            client_weights = [client_weights[i] for i in idxs_users]
            sum_w=sum(client_weights)
            if sum_w == 0:
                client_weights = [1 / len(client_weights) for _ in client_weights]
            else:
                for i in range(len(client_weights)):
                    client_weights[i]/=sum_w
                
            
            # print('len_client:',len(local_ws))
            self.fed_avg(local_ws, client_weights, 1)
            self.model.load_state_dict(self.w_t)# 经过avg之后的model作为下一轮的global model
            
            end = time.time()

            interval_time = end - start
            total_time+=interval_time
            sum_learn_time+=training_time_normal
            sum_unlearn_time+=training_time_ul
            '''
            Test the effects of global model and ul_model
            '''
            if (epoch + 1) == self.epochs or (epoch + 1) % 1 == 0:
                loss_train_mean, acc_train_mean = self.trainer.test(train_ldr)
                loss_val_mean, acc_val_mean = self.trainer.test(val_ldr)
                loss_class_mean, acc_before_mean = self.trainer.test(ul_ldr)  #测试ul之前, global model对该类别样本的识别效果

                loss_test_mean, acc_test_mean = loss_val_mean, acc_val_mean
                loss_val_ul__mean, acc_ul_val_mean = 0, 0
                loss_ul_mean, acc_ul_mean = 0, 0
                recovery_log = {}
                negative_calibration_log = {}

                """
                Need to test self.model_ul: 
                test ul_acc, val_acc after (W1+W2)/2
                """
                if self.ul_mode != 'none':
                    if self.ul_mode.startswith('ul_samples'):
                        if self.args.model_variant == 'independent':
                            self._load_independent_unlearn_model(ul_state_dicts, mode='samples')
                            if (epoch + 1) == self.epochs:
                                negative_calibration_log = self._calibrate_independent_negative_expert(
                                    negative_forget_ldr,
                                    negative_retain_ldr,
                                    mode='samples',
                                )
                        else:
                            ul_state_dict=copy.deepcopy(self.w_t)
                        
                            # print("W1:",self.model.state_dict()['classifier.weight'])
                            with torch.no_grad():

                                alpha=args.ul_samples_alpha
                                print("aplha:",alpha)
                                # multi clients
                                weight_ul=(1-alpha)*ul_state_dict['classifier.weight']
                                bias_ul=(1-alpha)*ul_state_dict['classifier.bias']
                                for idx in self.ul_clients:
                                    weight_ul += alpha / int(len(self.ul_clients)) * ul_state_dicts[idx]['classifier_ul.weight']
                                    bias_ul += alpha / int(len(self.ul_clients)) * ul_state_dicts[idx]['classifier_ul.bias']
                                
                                ul_state_dict['classifier.weight']=copy.deepcopy(weight_ul)
                                ul_state_dict['classifier.bias']=copy.deepcopy(bias_ul)

                                # #single client:
                                # weight_ul=((1-alpha)*self.model_ul.state_dict()['classifier.weight']+alpha*self.model_ul.state_dict()['classifier_ul.weight'])
                                # ul_state_dict['classifier.weight']=copy.deepcopy(weight_ul)

                                # bias_ul=((1-alpha)*self.model_ul.state_dict()['classifier.bias']+alpha*self.model_ul.state_dict()['classifier_ul.bias'])
                                # ul_state_dict['classifier.bias']=copy.deepcopy(bias_ul)

                            self.model.load_state_dict(ul_state_dict)
                    elif self.ul_mode=='ul_class':
                        # ul_model除W2外替换为global model的参数
                        if self.args.model_variant == 'independent':
                            self._load_independent_unlearn_model(ul_state_dicts, mode='class')
                            if (epoch + 1) == self.epochs:
                                negative_calibration_log = self._calibrate_independent_negative_expert(
                                    negative_forget_ldr,
                                    negative_retain_ldr,
                                    mode='class',
                                )
                        else:
                            combined_state_dict=copy.deepcopy(self.w_t)

                            with torch.no_grad():
                                weight_ul=combined_state_dict['classifier.weight']
                                bias_ul=combined_state_dict['classifier.bias']
                                for idx in self.ul_clients:
                                    weight_ul-= 1/int(len(self.ul_clients)) * ul_state_dicts[idx]['classifier_ul.weight']
                                    bias_ul-= 1/int(len(self.ul_clients)) * ul_state_dicts[idx]['classifier_ul.bias']
                                
                                combined_state_dict['classifier.weight']=copy.deepcopy(weight_ul)
                                combined_state_dict['classifier.bias']=copy.deepcopy(bias_ul)

                                self.model.load_state_dict(combined_state_dict)

                    elif 'amnesiac' in self.ul_mode:
                        start_epoch = self._amnesiac_start_epoch()
                        scale=1.0
                        if epoch >=start_epoch:
                            update_sum = self._sum_amnesiac_updates(update_list, start_epoch, epoch)
                            amnesiac_state_dict=copy.deepcopy(self.w_t)
                            with torch.no_grad():
                                for param_name in update_sum:
                                    amnesiac_state_dict[param_name]-=update_sum[param_name] *scale
                                self.model.load_state_dict(amnesiac_state_dict)      
                       

                    """
                    Test the effect after replacing W1 based on Ul module, including:
                    1. the acc on val set
                    2. the acc on ul_test_set 
                        (when ul_samples, ul_test_set is the set of unlearned samples;
                        when ul_class, ul_test_set is the set of samples whose target=ul_class_id in origin test_set)
                    
                    After testing, reload the global model back to prepare for the next round of training.
                    """
                    loss_val_ul__mean, acc_ul_val_mean = self.trainer.test(val_ldr)
                    
                    loss_ul_mean, acc_ul_mean = self.trainer.ul_test(ul_ldr)

                    if (
                        (epoch + 1) == self.epochs
                        and self._should_run_independent_recovery()
                        and recovery_ldr is not None
                    ):
                        pre_recovery_val_acc = acc_ul_val_mean
                        pre_recovery_ul_effect = acc_ul_mean
                        recovery_result = self._run_independent_recovery(
                            recovery_ldr,
                            val_ldr,
                            ul_ldr,
                            pre_recovery_val_acc,
                            pre_recovery_ul_effect,
                            calibration_forget_ldr=negative_forget_ldr,
                            calibration_retain_ldr=negative_retain_ldr,
                            calibration_mode='class' if self.ul_mode == 'ul_class' else 'samples',
                        )
                        total_time += recovery_result['time']
                        sum_unlearn_time += recovery_result['time']
                        sum_recovery_time += recovery_result['time']
                        loss_val_ul__mean = recovery_result['final_val_loss']
                        acc_ul_val_mean = recovery_result['final_val_acc']
                        loss_ul_mean = recovery_result['final_ul_loss']
                        acc_ul_mean = recovery_result['final_ul_effect']
                        recovery_log = {
                            "recovery": True,
                            "recovery_applied": recovery_result['accepted'],
                            "recovery_epochs": recovery_result['epochs'],
                            "recovery_lr": round(recovery_result['lr'], 6),
                            "recovery_objective": recovery_result['objective'],
                            "recovery_freeze_aux": recovery_result['freeze_aux'],
                            "recovery_blend": round(recovery_result['blend'], 4),
                            "recovery_retain_gate_mean": None if recovery_result['retain_gate_mean'] is None else round(recovery_result['retain_gate_mean'], 4),
                            "recovery_ul_gate_mean": None if recovery_result['ul_gate_mean'] is None else round(recovery_result['ul_gate_mean'], 4),
                            "recovery_loss": round(recovery_result['loss'], 4),
                            "recovery_train_acc": round(recovery_result['train_acc'], 4),
                            "pre_recovery_UL val acc": round(pre_recovery_val_acc, 4),
                            "pre_recovery_UL effect": round(pre_recovery_ul_effect, 4),
                            "candidate_recovery_UL val acc": round(recovery_result['candidate_val_acc'], 4),
                            "candidate_recovery_UL effect": round(recovery_result['candidate_ul_effect'], 4),
                            "selected_recovery_UL val acc": round(recovery_result['final_val_acc'], 4),
                            "selected_recovery_UL effect": round(recovery_result['final_ul_effect'], 4),
                        }
                        negative_calibration_log = recovery_result.get('calibration', negative_calibration_log)
                        print('[FedUL Recovery] accepted: {} -- blend {:.2f} -- pre Rm-Acc {:.4f}, candidate Rm-Acc {:.4f}, final Rm-Acc {:.4f}'.format(
                            recovery_result['accepted'],
                            recovery_result['blend'],
                            pre_recovery_val_acc,
                            recovery_result['candidate_val_acc'],
                            acc_ul_val_mean,
                        ))
                    elif (
                        (epoch + 1) == self.epochs
                        and self._should_run_independent_recovery()
                        and recovery_ldr is None
                    ):
                        recovery_log = {
                            "recovery": True,
                            "recovery_applied": False,
                            "recovery_skip_reason": "empty_retained_set",
                        }

                    if (epoch + 1) == self.epochs and self.args.model_variant == 'independent':
                        final_unlearn_state_dict = copy.deepcopy(self.model.state_dict())

                    if self.args.model_variant == 'independent':
                        self.model = self.learning_model
                        self.trainer.model = self.model
                    self.model.load_state_dict(self.w_t) #重新加载回global model

                self.logs['train_acc'].append(acc_train_mean)
                self.logs['train_loss'].append(loss_train_mean)
                self.logs['val_acc'].append(acc_val_mean)
                self.logs['val_loss'].append(loss_val_mean)
                self.logs['local_loss'].append(np.mean(local_losses))


                if self.logs['best_test_acc'] < acc_val_mean:
                    self.logs['best_test_acc'] = acc_val_mean
                    self.logs['best_test_loss'] = loss_val_mean
                    self.logs['best_model'] = copy.deepcopy(self.model.state_dict())

                print('Epoch {}/{}  --time {:.1f} --learn time {:.2f} --unlearn time {:.2f}'.format(
                    epoch, self.epochs,
                    interval_time, training_time_normal,training_time_ul
                )
                )

                print(
                    "Train Loss {:.4f}  -- Val Loss {:.4f} --Unlearned Val Loss {:.4f}"
                    .format(loss_train_mean, loss_val_mean, loss_val_ul__mean))
                print("Train acc {:.4f} -- Val acc {:.4f} --UL set acc {:.4f} --Best acc {:.4f}".format(acc_train_mean,
                                                                                    acc_val_mean,
                                                                                    acc_before_mean,
                                                                                    self.logs['best_test_acc']
                                                                                                        )
                    )
                print("Unlearned Val acc {:.4f} -- Unlearn effect {:.4f}".format(acc_ul_val_mean,acc_ul_mean)
                    )
                # s = 'epoch:{}, lr:{}, val_acc:{:.4f}, val_loss:{:.4f}, tarin_acc:{:.4f}, train_loss:{:.4f},time:{:.4f}, total_time:{:.4f}'.format(epoch,self.lr,acc_val_mean,loss_val_mean,acc_train_mean,loss_train_mean,interval_time,total_time)
                
                # with open(fn, 'a', encoding = 'utf-8') as f:   
                #     f.write(s)
                #     f.write('\n')
                

                log_payload = {"epoch":epoch,"lr":round(self.lr,4),"train_acc":round(acc_train_mean,4  ),"test_acc":round(acc_val_mean,4),\
                                "UL set acc":round(acc_before_mean,4),"UL val acc":round(acc_ul_val_mean,4),"UL effect":round(acc_ul_mean,4),\
                                "time":round(total_time,2),"learn_time":round(sum_learn_time,2),"unlearntime":round(sum_unlearn_time,2)}
                if self.args.model_variant == 'independent':
                    log_payload["disable_cne"] = bool(getattr(self.args, 'disable_cne', False))
                    log_payload["disable_gra"] = bool(getattr(self.args, 'disable_gra', False))
                if recovery_log or sum_recovery_time > 0:
                    log_payload["recovery_time"] = round(sum_recovery_time,2)
                log_payload.update(recovery_log)
                if negative_calibration_log:
                    log_payload.update({
                        "negative_threshold": round(negative_calibration_log.get('negative_threshold', 0.0), 6),
                        "negative_target_class": negative_calibration_log.get('negative_target_class', -1),
                        "negative_forget_trigger_rate": round(negative_calibration_log.get('negative_forget_trigger_rate', 0.0), 4),
                        "negative_retain_trigger_rate": None
                            if negative_calibration_log.get('negative_retain_trigger_rate') is None
                            else round(negative_calibration_log.get('negative_retain_trigger_rate'), 4),
                    })
                with open(fn,"a") as f:
                    json.dump(log_payload,f)
                    f.write('\n')
            
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs:
                self.model_ul.load_state_dict(self.w_t,strict=False) #更新model_ul，用于保存
                if (epoch + 1) == self.epochs and final_unlearn_state_dict is not None:
                    self.model_ul.load_state_dict(final_unlearn_state_dict,strict=False)
                save_dir = experiment_output_dir(
                    self.args.log_folder_name, self.args.model_name, self.args.dataset
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                pkl_name = "_".join(
                            ['FedUL_model', f's{self.args.seed}', str(args.num_users), str(args.batch_size),str(args.lr), str(args.iid), f'{today.year}_{today.month}_{today.day}'])
                pkl_name = save_dir / pkl_name
                print("pkl_name:",pkl_name)

                save_dict={'model_state_dict':copy.deepcopy(self.model.state_dict()),
                           'model_ul_state_dict':copy.deepcopy(self.model_ul.state_dict()),
                           "private_samples_idxs":self.private_samples_idxs,
                           "final_train_idxs":self.final_train_idxs,
                           "ul_clients":self.ul_clients,
                           "dict_users":self.dict_users
                           }
                if final_unlearn_state_dict is not None:
                    save_dict['final_unlearn_state_dict'] = copy.deepcopy(final_unlearn_state_dict)
                if recovery_log.get('recovery', False):
                    save_dict['fedul_recovery_attempted'] = True
                    save_dict['fedul_recovery_applied'] = recovery_log.get('recovery_applied', False)
                torch.save(save_dict, str(pkl_name) + ".pkl")

                if 'amnesiac' in self.ul_mode:
                    pkl_name = "_".join(
                            ['FedUL_updates_list', f's{self.args.seed}',f'e{epoch}', str(args.num_users), str(args.batch_size),str(args.lr), str(args.iid), f'{today.year}_{today.month}_{today.day}',time_mark])
                    pkl_name = save_dir / pkl_name
                    torch.save(update_list, str(pkl_name) + ".pkl")

        print('------------------------------------------------------------------------')
        print('Test loss: {:.4f} --- Test acc: {:.4f}  '.format(self.logs['best_test_loss'], 
                                                                                       self.logs['best_test_acc']
                                                                                       ))
        if 'fedrecovery' in self.ul_mode:
            pass

        return self.logs, interval_time, self.logs['best_test_acc'], acc_test_mean, 
    
    def fed_avg(self, local_ws, client_weights, lr_outer):

        w_avg = copy.deepcopy(local_ws[0])
        
        # client_weight=1.0/len(local_ws)
        # print('client_weights:',client_weights)
        
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * client_weights[0]

            for i in range(1, len(local_ws)):
                w_avg[k] += local_ws[i][k] * client_weights[i] *lr_outer

            self.w_t[k] = w_avg[k]
            

def main(args):
    logs = {'net_info': None,
            'arguments': {
                'frac': args.frac,
                'local_ep': args.local_ep,
                'local_bs': args.batch_size,
                'lr_outer': args.lr_outer,
                'lr_inner': args.lr,
                'iid': args.iid,
                'wd': args.wd,
                'optim': args.optim,      
                'model_name': args.model_name,
                'dataset': args.dataset,
                'log_interval': args.log_interval,                
                'num_classes': args.num_classes,
                'epochs': args.epochs,
                'num_users': args.num_users
            }
            }
    # args.save_dir=""
    # save_dir = args.save_dir
    save_dir=args.log_folder_name
    fl = FederatedLearning(args)

    logg, time, best_test_acc, test_acc = fl.train()                                         
                                             
    logs['net_info'] = logg  #logg=self.logs,    self.logs['best_model'] = copy.deepcopy(self.model.state_dict())
    logs['test_acc'] = test_acc
    logs['bp_local'] = True if args.bp_interval == 0 else False
    #print(logg['keys'])
    
    pkl_path = experiment_output_dir(save_dir, args.model_name, args.dataset)
    pkl_path.mkdir(parents=True, exist_ok=True)
    torch.save(logs,
               pkl_path / 'Final_s{}_iid{}_epoch_{}_E_{}_batch{}_lr{}_c_{}_{:.1f}_{:.4f}_{:.4f}.pkl'.format(
                   args.seed,args.iid, args.epochs, args.local_ep,args.batch_size, args.lr, args.num_users, args.frac, time, test_acc
               ))
    return

def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
         torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = parser_args()
    print(args)

    setup_seed(args.seed)

    main(args)
    # wandb.finish()
