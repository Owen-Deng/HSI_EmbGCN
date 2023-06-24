import logging
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch import nn
from torch.cuda.amp import GradScaler, autocast

import settings
import utils as ut

def preprocess_data(dataset_name,num_pca,num_emp,global_setting,trainings,load_origin=False):
    cache_data = Path(DATASET_FOLDER,dataset_name,f'{num_pca}_{num_emp}.npz')
    if not load_origin and cache_data.exists():
        hsi_pca, label_list, gt = np.load(str(cache_data)).values()
        logging.info(f'load cached data: {hsi_pca.shape}, {label_list.shape}, {gt.shape}')
    else:
        hsi, gt = ut.load_dataset(dataset_name,DATASET_FOLDER)
        hsi_len = hsi.shape[0] * hsi.shape[1]
        label_list = gt.reshape(hsi_len).astype(np.int64)
        
        label_list = np.delete(label_list,np.where(label_list== 0)) #删除没有标记的样本
        label_list = label_list - 1  #标签要从0开始
        
        hsi_norm = ut.normalization(hsi)
        hsi_pca = ut.apply_pca(hsi_norm,num_pca)
        if num_emp > 0:
            import emp
            hsi_pca = emp.build_emp(base_image=hsi_pca, num_openings_closings=num_emp)
        logging.info(f'load original data: {hsi_pca.shape}, {label_list.shape}, {gt.shape}')
        np.savez(str(cache_data),hsi_pca,label_list,gt)
        
    patch_size = 0
    for key in trainings:
        if trainings[key]['patch_size'] > patch_size:
            patch_size = trainings[key]['patch_size']
    hsi_patches = ut.generate_patches(hsi_pca,gt,patch_size)
    return hsi_patches, label_list, gt


def train_target(net,train_loader,test_loader,train_x, train_y,round_i,args):
    num_epochs = args.get('num_epoch',1000)
    lr = args['started_lr']
    amp = args.get('amp',False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, lr*0.01)
    fixed_lr = False
    pbar = tqdm.tqdm(range(num_epochs))
    test = args.get('test_during_training',0)
    patch_size = args['patch_size']
    try:
        data_patch_size = train_loader.dataset.tensors[0].shape[-2]
    except:
        data_patch_size = train_loader.dataset.samples.shape[-2]
    cur_offset = (data_patch_size - patch_size ) // 2
    test_accs = []
    if amp:
        scaler = GradScaler()
    for epoch in pbar:
        for X,y in train_loader:
            optimizer.zero_grad()
            if cur_offset != 0:
                X = X[:,:,cur_offset:patch_size+cur_offset,cur_offset:patch_size+cur_offset,:]
            if amp:
                with autocast():
                    y_hat = net(X)
                    l = criterion(y_hat, y)
                scaler.scale(l).backward()
                scaler.step(optimizer)
            else:
                y_hat = net(X)
                l = criterion(y_hat, y)
                l.backward() 
                optimizer.step()
        
        if amp:
            scale = scaler.get_scale()
            scaler.update()
            if not fixed_lr:
                skip_lr_sched = (scale > scaler.get_scale())
                if not skip_lr_sched:
                    scheduler.step()
        else:
            if not fixed_lr:
                scheduler.step()
        
        
        if test > 0 and (epoch+1) % test == 0 and epoch != (num_epochs - 1):
            test_acc = ut.test_accuracy(net,test_loader,patch_size,True)
            test_accs.append(test_acc)

            pbar.set_description(f'{test_acc:.5f}')


    pbar.clear()
    return net, test_accs


def augment_tradition(x,y):
    aug_x = np.empty((0,*x.shape[1:]),dtype=np.float32)
    aug_y = np.empty(0,dtype=np.int64)
    
    for idx in range(x.shape[0]):
        x_tmp = np.copy(x[idx])
        x_tmp = np.reshape(x_tmp,(1,*x_tmp.shape))
        x_data_t_flr = np.flip(x_tmp,axis=1)#左右翻转
        x_data_t_fud = np.flip(x_tmp,axis=2)#上下翻转
        x_data_t_rot90 = np.rot90(x_tmp,1,(1,2))#旋转
        x_data_t_rot180 = np.rot90(x_tmp,2,(1,2))
        x_data_t_rot270 = np.rot90(x_tmp,3,(1,2))
        x_data_t_trans = np.transpose(x_tmp,(0,2,1,3))#主对角线翻转
        x_data_t_trans_2 = np.flip(x_data_t_rot90,2)#副对角线翻转
        aug_x = np.concatenate((aug_x,x_data_t_flr,x_data_t_fud,x_data_t_trans,
                                x_data_t_trans_2,x_data_t_rot90 ,x_data_t_rot180,x_data_t_rot270),
                                axis=0)
        tmp_y = y[idx].reshape(1)
        aug_y = np.concatenate((aug_y,tmp_y,tmp_y,tmp_y,
                                tmp_y,tmp_y,tmp_y,tmp_y),axis=0)

    train_x_aug = np.concatenate([x,aug_x],axis=0)
    train_y_aug = np.concatenate([y,aug_y],axis=0)
    return train_x_aug,train_y_aug
    

def train(dataset_name,global_setting,trainings,i_seed):
    train_round = global_setting['train_round']
    ret_dict={}
    hsi_patches, label_list, gt = preprocess_data(dataset_name,global_setting['num_pca'],global_setting['num_emp'],global_setting,trainings)
    for train_idx , train_i in enumerate(range(0,train_round)):
        round_seed = i_seed + train_i*10
        train_x,train_y,test_x,test_y = ut.split_data(
            hsi_patches, label_list,global_setting['n_train_samples'],global_setting['n_test_samples'],seed = round_seed)
        
        test_batch_size = global_setting['test_batch_size']
        n_workers= 0 if WIN else 1
        
        start_round_time = time.time()
        for train_name,args in trainings.items():
            if train_idx >= args.get('test_round',1):
                args['test_during_training'] = 0
            
            args['num_class'] = np.unique(train_y).shape[0]
            patch_size = args['patch_size']
            offset = (train_x.shape[1] - patch_size )// 2
            new_train_x = train_x[:,offset:offset+patch_size,offset:offset+patch_size,:]
            
            train_loader = ut.get_dataloader(new_train_x,train_y,args['batch_size'],n_workers,'cuda:0')
            num_band = train_loader.dataset.tensors[0].shape[-1]
        
            net = get_model(args['model'],num_band,args['num_class'],args,round_seed)

            new_test_x = test_x[:,offset:offset+patch_size,offset:offset+patch_size,:]
            test_loader = ut.get_dataloader(new_test_x,test_y,test_batch_size,n_workers)
            train_time = time.time()
            net, test_accs = train_target(net,train_loader,test_loader,train_x,train_y,train_i,args)
            
            train_time = time.time() - train_time
            test_time = time.time()
            OA,Kappa,AA = ut.test_accuracy(net,test_loader,args['patch_size'])
            test_time = time.time() - test_time
            if train_idx < args.get('test_round',1):
                test_accs.append(OA)
            if train_name in ret_dict:
                ret_dict[train_name][0].append(OA)
                ret_dict[train_name][1].append(Kappa)
                ret_dict[train_name][2].append(AA)
                ret_dict[train_name][3].append(test_accs)
            else:
                ret_dict[train_name] = [[OA],[Kappa],[AA],[test_accs]]
                
            c_map = global_setting.get('c_map',False)
            if c_map:
                new_hsi_patches = hsi_patches[:,offset:offset+patch_size,offset:offset+patch_size,:]
                ut.save_classification_map(net,new_hsi_patches,gt,dataset_name,f'{train_name}_{OA*100:.2f}')
        
            logging.info(f'{train_name} OA: {OA:.5f}, Kappa: {Kappa:.5f}, AA: {np.mean(AA):.5f}, train time: {train_time:.2f}, test time: {test_time:.2f}, num_class: {args["num_class"]}'.center(80,"-"))
        used_round_time = time.time() - start_round_time
        logging.info(f'Train round {train_i} ended, used {used_round_time:.1f} sec '.center(80,"="))
    output_statistics(ret_dict, global_setting)

def get_model(name :str,num_band,num_class,args,seed):
    from models import AugNet
    model = AugNet(name, num_band,args)

    model = model.cuda()
    return model

def output_statistics(ret_dict:dict, global_setting):
    ret_dict = ljust_dict_key(ret_dict)
    for key,value in ret_dict.items():
        mean_OA = np.mean(value[0])
        std_OA = np.std(value[0])
        mean_Kappa = np.mean(value[1])
        std_Kappa = np.std(value[1])
        AA = np.mean(value[2],axis=0)
        mean_AA = np.mean(AA)
        std_mean_AA = np.std(AA)
        test_accs = np.array(value[3])
        test_accs = np.mean(test_accs,axis=0)
        logging.info(f'{key}: OA {mean_OA*100:.2f}±{std_OA*100:.2f}, Kappa {mean_Kappa*100:.2f}±{std_Kappa:.2f}, Mean_AA {mean_AA*100:.2f}±{std_mean_AA:.2f}, AA {str(AA)}, accs {test_accs}')


def ljust_dict_key(indict):
    max_len = 0
    for key in indict:
        t_len = len(key)
        if t_len> max_len:
            max_len = t_len
    new_trainings = {}
    for key in indict:
        new_key = key.ljust(max_len)
        new_trainings[new_key] = indict[key]
    return new_trainings


if __name__ == '__main__':
    if platform.system().lower() == 'windows':
        DATASET_FOLDER = 'D:/Datasets/'
        WIN = True
    else:
        DATASET_FOLDER = '../Datasets/'
        WIN = False
    whole_start = time.time()
    ut.init()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

    global_settings = {
        'test': settings.global_setting(num_pca=7,num_emp=3,train_round=1,n_train_samples=4,c_map=True)
    }
    
    tar_datasets = ['PU','KSC','BW','IP','PC']
    seed = 32668
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 

    trainings = settings.compare

    len_settings = len(list(global_settings.keys()))
    for global_setting_key in global_settings:
        if len_settings != 1:
            logging.info(f' Global setting {global_setting_key} Started '.center(80,"#"))
            
        for dataset_name in tar_datasets:
            logging.info(f' Dataset {dataset_name} Started '.center(80,"#"))
            start_time = time.time()
            train(dataset_name,global_settings[global_setting_key],trainings,seed)
            used_time = time.time() - start_time
            logging.info(f' Dataset {dataset_name} Ended, Used {used_time:.1f} seconds '.center(80,"#"))
        
        if len_settings != 1:
            logging.info(f' Global setting {global_setting_key} Ended '.center(80,"#"))    
    
    whole_end = time.time() - whole_start
    logging.info(f'Took {whole_end:.0f} seconds in total.')