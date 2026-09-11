import numpy as np
import os
import random

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from dataset import UL_CIFAR10,UL_CIFAR100,UL_MNIST,UL_CelebA
from utils.sampling import *


def get_data(dataset, data_root,proportion, iid, num_users,UL_clients, data_aug, noniid_beta,samples_per_user, ul_mode,ul_class_id, celeba_attr='Smiling', celeba_target_mode='attrs10'):
    ds = dataset 
    supported_datasets = {'mnist', 'cifar10', 'cifar100', 'celeba'}
    if ds not in supported_datasets:
        raise ValueError(
            'Unsupported dataset: {}. Choose one of {}.'.format(
                ds,
                ', '.join(sorted(supported_datasets)),
            )
        )
    # total_sample=samples_per_user*num_users
    # #print("total:",total_sample)
    # test_samples=max(samples_per_user,10000)
    
    #print("total:",total_sample)
    test_samples=10000
    # print(UL_clients)

    if ds == 'celeba':
        if 'client' in ul_mode:
            raise ValueError('Client unlearning is not supported in the open-source CelebA setup.')

        transform_train = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        probe_set = UL_CelebA(
            data_root,
            [],
            ul_class_id,
            proportion,
            train=True,
            transform=None,
            ul_mode='none',
            attr_name=celeba_attr,
            target_mode=celeba_target_mode,
        )
        total_sample = len(probe_set)
        noniid_partition = None
        if not iid:
            noniid_partition = cifar_beta(probe_set, noniid_beta, num_users)

        private_samples_idxs = []
        if 'samples' in ul_mode:
            num_private_samples = int(proportion * total_sample)
            probe_targets = np.array(probe_set.targets).astype(np.int64)
            if iid:
                source_pool = range(total_sample)
            else:
                partition_users = noniid_partition[0]
                source_pool = set()
                for client_id in UL_clients:
                    if client_id not in partition_users:
                        raise ValueError(
                            'Unknown CelebA unlearning client {} for {} clients.'.format(
                                client_id,
                                num_users,
                            )
                        )
                    source_pool.update(partition_users[client_id])

            source_idxs = [
                idx for idx in source_pool
                if int(probe_targets[idx]) != int(ul_class_id)
            ]
            if len(source_idxs) < num_private_samples:
                raise ValueError(
                    'Not enough CelebA source samples for sample-level backdoor unlearning: '
                    'need {}, found {} for {} != {}.'.format(
                        num_private_samples,
                        len(source_idxs),
                        probe_set._target_description(),
                        ul_class_id,
                    )
                )
            private_samples_idxs = random.sample(source_idxs, num_private_samples)

        train_set = UL_CelebA(
            data_root,
            private_samples_idxs,
            ul_class_id,
            proportion,
            train=True,
            transform=transform_train,
            ul_mode=ul_mode,
            attr_name=celeba_attr,
            target_mode=celeba_target_mode,
        )
        private_samples_idxs = train_set.ul_sample_idxs
        final_train_idxs = train_set.final_train_list
        print(len(train_set))

        if 'class' in ul_mode:
            test_set = UL_CelebA(
                data_root,
                [],
                ul_class_id,
                proportion,
                train=False,
                transform=transform_test,
                ul_mode=ul_mode,
                attr_name=celeba_attr,
                target_mode=celeba_target_mode,
            )
            splited_ulclass_idxs = list(set(list(range(0, len(test_set)))).difference(set(test_set.ul_class_idxs)))
            ul_test_set = DatasetSplit(test_set, test_set.ul_class_idxs)
            test_set = DatasetSplit(test_set, splited_ulclass_idxs)
            print('ul_test_set len:', len(ul_test_set))
            print('normal test_set len:', len(test_set))
        elif 'samples' in ul_mode:
            test_set = UL_CelebA(
                data_root,
                [],
                ul_class_id,
                proportion,
                train=False,
                transform=transform_test,
                ul_mode='none',
                attr_name=celeba_attr,
                target_mode=celeba_target_mode,
            )
            ul_test_set = DatasetSplit(train_set, private_samples_idxs)
        else:
            test_set = UL_CelebA(
                data_root,
                [],
                ul_class_id,
                proportion,
                train=False,
                transform=transform_test,
                ul_mode='none',
                attr_name=celeba_attr,
                target_mode=celeba_target_mode,
            )
            ul_test_set = {}

        if iid:
            dict_users, train_idxs, val_idxs = cifar_iid_ul(
                train_set,
                num_users,
                UL_clients,
                ul_mode,
            )
        else:
            if len(probe_set) != len(train_set):
                raise ValueError(
                    'CelebA Non-IID partition indices no longer match the training dataset.'
                )
            dict_users, train_idxs, val_idxs = noniid_partition

            if 'class' in ul_mode:
                target_idxs = set(private_samples_idxs)
                for client_id in UL_clients:
                    target_count = len(dict_users[client_id].intersection(target_idxs))
                    if target_count == 0:
                        raise ValueError(
                            'CelebA Non-IID unlearning client {} has no samples from target class {}. '
                            'Use another seed or a larger --beta.'.format(
                                client_id,
                                ul_class_id,
                            )
                        )
                    print(
                        'client {}, ul_class samples {}'.format(
                            client_id,
                            target_count,
                        )
                    )

            if 'retrain' in ul_mode:
                private_idx_set = set(private_samples_idxs)
                for client_id in UL_clients:
                    dict_users[client_id] = dict_users[client_id].difference(
                        private_idx_set
                    )

        return train_set, test_set, ul_test_set, dict_users, train_idxs, val_idxs, private_samples_idxs, final_train_idxs
    
    if ds == 'cifar10':
        total_sample=50000
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])  
        transform_test = transforms.Compose([transforms.CenterCrop(32),
                                             transforms.ToTensor(),
                                             normalize,
                                             ])

        
        train_idxs=np.arange(0, total_sample)

        if iid:
            """
            先确定ul样本, 
            1) ul samples-->随机指定
            2) ul class --> 根据class id在数据集中查找
            """
        
            private_samples_idxs=[]
            # if ul_mode=='ul_samples' or ul_mode =='ul_samples_backdoor'or ul_mode == 'retrain_samples':
            if 'samples' in ul_mode:
                num_private_samples=int(proportion*total_sample)
                private_samples_idxs=random.sample([i for i in range(total_sample)],num_private_samples)
            elif 'class' in ul_mode:
                ul_class_id=ul_class_id
                

            # if ul_mode=='retrain':
            #     retrain_idxs=list(set(np.arange(0, 50000))-set(private_samples_idxs))
            #     train_idxs=retrain_idxs
            train_set = UL_CIFAR10(data_root, 
                                    private_samples_idxs,
                                    ul_class_id,
                                    proportion,
                                    train=True,
                                    transform=transform_train,
                                    ul_mode=ul_mode
                                    )
            # ul sample idxs
            private_samples_idxs=train_set.ul_sample_idxs
            # the idxs of ul samples + common remaining samples
            final_train_idxs=train_set.final_train_list
            print(len(train_set))
            num_private_samples=len(private_samples_idxs)


            # train_set = DatasetSplit(train_set, train_idxs) 

            # if ul_mode =='ul_class' or ul_mode == 'retrain_class':
            if 'class' in ul_mode:
                test_set = UL_CIFAR10(data_root, 
                                    [],
                                    ul_class_id,
                                    proportion,
                                    train=False,
                                    transform=transform_test,
                                    ul_mode=ul_mode
                                    )
                # 筛选出0-8 class的样本索引,并由此划分最终的test_set
                splited_ulclass_idxs=list(set(list(range(0, len(test_set)))).difference(set(test_set.ul_class_idxs)))
                
                
            else:
                test_set = torchvision.datasets.CIFAR10(data_root,
                                                        train=False,
                                                        download=False,
                                                        transform=transform_test
                                                        )
                
                # test_set = DatasetSplit(test_set, np.arange(0, test_samples))
            
            # bulid ul_test_set for evaluating unlearn effect
            # if ul_mode == 'ul_samples' or ul_mode == 'ul_samples_backdoor' or ul_mode == 'retrain_samples':
            if 'samples' in ul_mode:
                ul_test_set=DatasetSplit(train_set, private_samples_idxs) 

            # elif ul_mode == 'ul_class' or ul_mode == 'retrain_class':
            elif 'class' in ul_mode:
                ul_test_set=DatasetSplit(test_set, test_set.ul_class_idxs)
                test_set=DatasetSplit(test_set, splited_ulclass_idxs)
                print('ul_test_set len:',len(ul_test_set))
                print('normal test_set len:',len(test_set))
            else:
                ul_test_set={}
            ## iid数据划分
            dict_users, train_idxs, val_idxs = cifar_iid_ul(train_set, num_users, UL_clients, ul_mode)

        else:
            private_samples_idxs=[]
            train_set = torchvision.datasets.CIFAR10(data_root,
                                               train=True,
                                               download=False,
                                               transform=transform_train
                                               )
            dict_users, train_idxs, val_idxs = cifar_beta(train_set, noniid_beta, num_users)
            # 统计每个Ul client的sample idxs
            ul_clients_sample_idxs=[]
            for i in UL_clients:
                ul_clients_sample_idxs.extend(train_idxs[i])
            # 从中选取Ul idxs
            if 'class' not in ul_mode:
                num_private_samples=int(proportion*total_sample)
                private_samples_idxs=random.sample(ul_clients_sample_idxs,num_private_samples)
            # if 'class' in ul_mode:
            #     ul_class_id=ul_class_id
            # 重构数据集
            ul_class_id=ul_class_id
            train_set = UL_CIFAR10(data_root, 
                                    private_samples_idxs,
                                    ul_class_id,
                                    proportion, #无用
                                    train=True,
                                    transform=transform_train,
                                    ul_mode=ul_mode
                                    )
            private_samples_idxs=train_set.ul_sample_idxs #如果不是ul class无作用，反之返回对应class的idxs
            if 'class' in ul_mode:
                for i in range(num_users):
                    print("client {}, ul_class samples {} ".format(i,len(dict_users[i].intersection(set(private_samples_idxs)))))
            final_train_idxs=train_set.final_train_list
            print(len(train_set))
            num_private_samples=len(private_samples_idxs)


            if 'class' in ul_mode:
                test_set = UL_CIFAR10(data_root, 
                                    [],
                                    ul_class_id,
                                    proportion,
                                    train=False,
                                    transform=transform_test,
                                    ul_mode=ul_mode
                                    )
                # 筛选出0-8 class的样本索引,并由此划分最终的test_set
                splited_ulclass_idxs=list(set(list(range(0, len(test_set)))).difference(set(test_set.ul_class_idxs)))
                ul_test_set=DatasetSplit(test_set, test_set.ul_class_idxs)
                test_set=DatasetSplit(test_set, splited_ulclass_idxs)
                print('ul_test_set len:',len(ul_test_set))
                print('normal test_set len:',len(test_set))
            elif 'samples' in ul_mode:
                test_set = torchvision.datasets.CIFAR10(data_root,
                                                        train=False,
                                                        download=False,
                                                        transform=transform_test
                                                        )
            
                ul_test_set=DatasetSplit(train_set, private_samples_idxs) 
                # retrain samples下，去除pvt样本，只留正常样本 
                # ul client下，去除正常样本，只留pvt样本 
                # (retrain samples client时，dict_users[i]=[]，但训练会skip)
            if 'retrain' in ul_mode:
                for i in UL_clients:
                    print('ul_client {}, len_origin_dataset {}'.format(i,len(dict_users[i])))
                    dict_users[i] =dict_users[i].difference(set(private_samples_idxs)) # 求差集，只留common samples  
            for i in range(num_users):
                if i not in UL_clients:
                    print('client {}, len_dataset {}'.format(i,len(dict_users[i])))
                else:
                    print('ul_client {}, len_dataset {}'.format(i,len(dict_users[i])))


    
    if ds=='mnist':
        # train_data = torchvision.datasets.MNIST(root=data_root,
        #                     train=True,
        #                     download=True)

        # mean = train_data.data.float().mean() / 255
        # std = train_data.data.float().std() / 255
        mean=0.13066
        std=0.30811
        print('-------MNIST mean:{}  std:{}-------'.format(mean,std))

        transform_train = transforms.Compose([
                            transforms.RandomRotation(5, fill=(0,)),
                            transforms.RandomCrop(28, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[mean], std=[std])
                                      ])

        transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[mean], std=[std])
                                            ])
        

        # train_data = torchvision.datasets.MNIST(root=data_root,
        #                     train=True,
        #                     download=True,
        #                     transform=train_transforms)

        # test_set = torchvision.datasets.MNIST(root=data_root,
        #                         train=False,
        #                         download=True,
        #                         transform=test_transforms)
        total_sample=60000
        train_idxs=np.arange(0, total_sample)
        if iid:
            """
            先确定ul样本, 
            1) ul samples-->随机指定
            2) ul class --> 根据class id在数据集中查找
            """
            
            private_samples_idxs=[]
            # if ul_mode=='ul_samples' or ul_mode =='ul_samples_backdoor'or ul_mode == 'retrain_samples':
            if 'samples' in ul_mode:
                num_private_samples=int(proportion*total_sample)
                private_samples_idxs=random.sample([i for i in range(total_sample)],num_private_samples)
            elif 'class' in ul_mode:
                ul_class_id=ul_class_id
                

            # if ul_mode=='retrain':
            #     retrain_idxs=list(set(np.arange(0, 50000))-set(private_samples_idxs))
            #     train_idxs=retrain_idxs
            train_set = UL_MNIST(data_root, 
                                    private_samples_idxs,
                                    ul_class_id,
                                    proportion,
                                    train=True,
                                    transform=transform_train,
                                    ul_mode=ul_mode
                                    )
            # ul sample idxs
            private_samples_idxs=train_set.ul_sample_idxs
            # the idxs of ul samples + common remaining samples
            final_train_idxs=train_set.final_train_list
            print(len(train_set))
            num_private_samples=len(private_samples_idxs)


            # train_set = DatasetSplit(train_set, train_idxs) 

            # if ul_mode =='ul_class' or ul_mode == 'retrain_class':
            if 'class' in ul_mode:
                test_set = UL_MNIST(data_root, 
                                    [],
                                    ul_class_id,
                                    proportion,
                                    train=False,
                                    transform=transform_test,
                                    ul_mode=ul_mode
                                    )
                # 筛选出0-8 class的样本索引,并由此划分最终的test_set
                splited_ulclass_idxs=list(set(list(range(0, len(test_set)))).difference(set(test_set.ul_class_idxs)))
                
                
            else:
                test_set = torchvision.datasets.MNIST(data_root,
                                                        train=False,
                                                        download=False,
                                                        transform=transform_test
                                                        )
                
                # test_set = DatasetSplit(test_set, np.arange(0, test_samples))
            
            # bulid ul_test_set for evaluating unlearn effect
            # if ul_mode == 'ul_samples' or ul_mode == 'ul_samples_backdoor' or ul_mode == 'retrain_samples':
            if 'samples' in ul_mode:
                ul_test_set=DatasetSplit(train_set, private_samples_idxs) 

            # elif ul_mode == 'ul_class' or ul_mode == 'retrain_class':
            elif 'class' in ul_mode:
                ul_test_set=DatasetSplit(test_set, test_set.ul_class_idxs)
                test_set=DatasetSplit(test_set, splited_ulclass_idxs)
                print('ul_test_set len:',len(ul_test_set))
                print('normal test_set len:',len(test_set))
            else:
                ul_test_set={}
            dict_users, train_idxs, val_idxs = cifar_iid_ul(train_set, num_users, UL_clients, ul_mode)    
        else:
            private_samples_idxs=[]
            train_set = torchvision.datasets.MNIST(data_root,
                                               train=True,
                                               download=False,
                                               transform=transform_train
                                               )
            dict_users, train_idxs, val_idxs = cifar_beta(train_set, noniid_beta, num_users)
            # 统计每个Ul client的sample idxs
            ul_clients_sample_idxs=[]
            for i in UL_clients:
                ul_clients_sample_idxs.extend(train_idxs[i])
            # 从中选取Ul idxs
            if 'class' not in ul_mode:
                num_private_samples=int(proportion*total_sample)
                private_samples_idxs=random.sample(ul_clients_sample_idxs,num_private_samples)
            # if 'class' in ul_mode:
            #     ul_class_id=ul_class_id
            # 重构数据集
            ul_class_id=ul_class_id
            train_set = UL_MNIST(data_root, 
                                    private_samples_idxs,
                                    ul_class_id,
                                    proportion, #无用
                                    train=True,
                                    transform=transform_train,
                                    ul_mode=ul_mode
                                    )
            private_samples_idxs=train_set.ul_sample_idxs
            if 'class' in ul_mode:
                for i in range(num_users):
                    print("client {}, ul_class samples {} ".format(i,len(dict_users[i].intersection(set(private_samples_idxs)))))
            
            final_train_idxs=train_set.final_train_list
            print(len(train_set))
            num_private_samples=len(private_samples_idxs)


            if 'class' in ul_mode:
                test_set = UL_MNIST(data_root, 
                                    [],
                                    ul_class_id,
                                    proportion,
                                    train=False,
                                    transform=transform_test,
                                    ul_mode=ul_mode
                                    )
                # 筛选出0-8 class的样本索引,并由此划分最终的test_set
                splited_ulclass_idxs=list(set(list(range(0, len(test_set)))).difference(set(test_set.ul_class_idxs)))
                ul_test_set=DatasetSplit(test_set, test_set.ul_class_idxs)
                test_set=DatasetSplit(test_set, splited_ulclass_idxs)
                print('ul_test_set len:',len(ul_test_set))
                print('normal test_set len:',len(test_set))
            elif 'samples' in ul_mode:
                test_set = torchvision.datasets.MNIST(data_root,
                                                        train=False,
                                                        download=False,
                                                        transform=transform_test
                                                        )
            
                ul_test_set=DatasetSplit(train_set, private_samples_idxs) 
                
                # retrain samples下，去除pvt样本，只留正常样本 
                # ul client下，去除正常样本，只留pvt样本 
                # (retrain samples client时，dict_users[i]=[]，但训练会skip)
                if 'retrain' in ul_mode:
                    for i in UL_clients:
                        print('ul_client {}, len_origin_dataset {}'.format(i,len(dict_users[i])))
                        dict_users[i] =dict_users[i].difference(set(private_samples_idxs)) # 求差集，只留common samples  
                for i in range(num_users):
                    if i not in UL_clients:
                        print('client {}, len_dataset {}'.format(i,len(dict_users[i])))
                    else:
                        print('ul_client {}, len_ul_dataset {}'.format(i,len(dict_users[i])))

    if ds == 'cifar100':
        total_sample=50000
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),#transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                            transforms.ToTensor(),
                                            normalize,
                                            ])  
        transform_test = transforms.Compose([transforms.CenterCrop(32),
                                            transforms.ToTensor(),
                                            normalize,
                                            ])
       
      
        # train_set = torchvision.datasets.CIFAR100(data_root,
        #                                        train=True,
        #                                        download=True,
        #                                        transform=transform_train
        #                                        )

        # train_set = DatasetSplit(train_set, np.arange(0, total_sample))

        # train_set = DatasetSplit(train_set, np.random.permutation(total_sample))

        # test_set = torchvision.datasets.CIFAR100(data_root,
        #                                         train=False,
        #                                         download=False,
        #                                         transform=transform_test
        #                                         )
        # test_set = DatasetSplit(test_set, np.arange(0, samples_per_user))

        """
        先确定ul样本, 
        1) ul samples-->随机指定
        2) ul class --> 根据class id在数据集中查找
        """
    
        private_samples_idxs=[]
        # if ul_mode=='ul_samples' or ul_mode =='ul_samples_backdoor'or ul_mode == 'retrain_samples':
        if 'samples' in ul_mode:
            num_private_samples=int(proportion*total_sample)
            private_samples_idxs=random.sample([i for i in range(total_sample)],num_private_samples)
        elif 'class' in ul_mode:
            ul_class_id=ul_class_id

        # if ul_mode=='retrain':
        #     retrain_idxs=list(set(np.arange(0, 50000))-set(private_samples_idxs))
        #     train_idxs=retrain_idxs
        train_set = UL_CIFAR100(data_root, 
                                private_samples_idxs,
                                ul_class_id,
                                proportion,
                                train=True,
                                transform=transform_train,
                                ul_mode=ul_mode
                                )
        # ul sample idxs (ul class时需要新赋值该参数)
        private_samples_idxs=train_set.ul_sample_idxs
        # final_train_idxs=the idxs of ul samples + common remaining samples, 用于
        final_train_idxs=train_set.final_train_list
        print(len(train_set))
        num_private_samples=len(private_samples_idxs)


        # train_set = DatasetSplit(train_set, train_idxs) 

        # if ul_mode =='ul_class' or ul_mode == 'retrain_class':
        if 'class' in ul_mode:
            test_set = UL_CIFAR100(data_root, 
                                [],
                                ul_class_id,
                                proportion,
                                train=False,
                                transform=transform_test,
                                ul_mode=ul_mode
                                )
            # 筛选出0-8 class的样本索引,并由此划分最终的test_set
            splited_ulclass_idxs=list(set(list(range(0, len(test_set)))).difference(set(test_set.ul_class_idxs)))     
            
        else:
            test_set = torchvision.datasets.CIFAR100(data_root,
                                                    train=False,
                                                    download=False,
                                                    transform=transform_test
                                                    )
            
            # test_set = DatasetSplit(test_set, np.arange(0, test_samples))
        
        # bulid ul_test_set for evaluating unlearn effect
        # if ul_mode == 'ul_samples' or ul_mode == 'ul_samples_backdoor' or ul_mode == 'retrain_samples':
        if 'samples' in ul_mode:
            ul_test_set=DatasetSplit(train_set, private_samples_idxs) 

        # elif ul_mode == 'ul_class' or ul_mode == 'retrain_class':
        elif 'class' in ul_mode:
            ul_test_set=DatasetSplit(test_set, test_set.ul_class_idxs)
            test_set=DatasetSplit(test_set, splited_ulclass_idxs)
            print('ul_test_set len:',len(ul_test_set))
            print('normal test_set len:',len(test_set))
        else:
            ul_test_set={}
        ## iid数据划分
        dict_users, train_idxs, val_idxs = cifar_iid_ul(train_set, num_users, UL_clients, ul_mode)

    return train_set, test_set, ul_test_set, dict_users, train_idxs, val_idxs, private_samples_idxs,final_train_idxs

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.ul_sample_idxs=[]
        try:
            self.ul_sample_idxs=dataset.ul_sample_idxs
        except AttributeError as e:
            self.ul_sample_idxs=[]


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        if isinstance(item, list):
            return self.dataset[[self.idxs[i] for i in item]]

        # print("item:",item)
        # print("idx:",self.idxs[item])
        # print("len_idx:",len(self.idxs))
        image, label = self.dataset[self.idxs[item]]
        # print("idx2:",self.idxs[item])
        
        return image, label
