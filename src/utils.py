import logging
import math
import os
import time
from multiprocessing import Process, Queue
from pathlib import Path

import hdf5storage
import numpy as np
import scipy.io as sio
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from torch.utils import data

def test_accuracy_fewshot(net,supports,support_labels, test_iter, OnlyOA =False):
    if isinstance(supports,np.ndarray):
        supports = torch.from_numpy(supports).view((-1,1,*supports.shape[1:])).cuda()
        support_labels = torch.from_numpy(support_labels).cuda()
    supports = supports.cuda()
    support_labels = support_labels.cuda()
    total_hit, total_num = 0,0
    total_preds = []
    total_trues = []
    n_way = torch.unique(support_labels).shape[0]
    net.eval()
    with torch.no_grad():
        z_support = net(supports)
        z_proto = torch.cat([
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)]) 
        
        for idx, (querys, query_labels) in enumerate(test_iter):
            z_query = net(querys.cuda())
            dists = torch.cdist(z_query, z_proto)
            scores = -dists
            result_labels = torch.max(scores.detach().data,1)[1]
            total_preds.extend(result_labels.tolist())
            total_trues.extend(query_labels.tolist())
    
    if OnlyOA:
        return accuracy_score(total_trues,total_preds)
    
    AAs = []
    c_m = confusion_matrix(total_trues,total_preds)
    for i in range(c_m.shape[0]):
        AAs.append(c_m[i,i]/np.sum(c_m[i,:]))
    
    return accuracy_score(total_trues,total_preds),cohen_kappa_score(total_trues,total_preds),AAs

def get_patch_size_from_dataloader(loader):
    try:
        data_patch_size = loader.dataset.tensors[0].shape[-2]
    except:
        data_patch_size = loader.dataset.samples.shape[-2]
    
    return data_patch_size


def test_accuracy(net, test_iter, patch_size, OnlyOA = False):
    device = torch.device('cuda:0')
    
    data_patch_size = get_patch_size_from_dataloader(test_iter)
    cur_offset = (data_patch_size - patch_size ) // 2

    total_preds = []
    total_trues = []
    net.eval()
    with torch.no_grad():
        for X,y in test_iter:
            if cur_offset != 0:
                X = X[:,:,cur_offset:patch_size+cur_offset,cur_offset:patch_size+cur_offset,:]
            X = X.to(device,non_blocking = False)
            preds = net(X).argmax(axis=1)

            total_preds.append(preds)
            total_trues.append(y)
            
    
    #total_preds = total_preds.detach().cpu().numpy()
    #total_trues = total_trues.numpy()
    
    total_preds = torch.cat(total_preds).cpu().numpy()
    total_trues = torch.cat(total_trues).numpy()
    net.train()
    if OnlyOA:
        OA = float(accuracy_score(total_trues,total_preds).item())
        return OA
    
    
    AAs = []
    c_m = confusion_matrix(total_trues,total_preds)
    for i in range(c_m.shape[0]):
        total_count = np.sum(c_m[i,:])
        AAs.append(0 if total_count == 0 else c_m[i,i]/total_count)

    return float(accuracy_score(total_trues,total_preds).item()),cohen_kappa_score(total_trues,total_preds),AAs
    

def shuffle_hsi(hsi_n,label_d,seed=0):
    rd_seed = 0
    if seed == 0:
        rd_seed = np.random.randint(65535)
    else:
        rd_seed = seed
    hsi_ns = np.copy(hsi_n)
    label_ds = np.copy(label_d)

    np.random.seed(rd_seed)
    np.random.shuffle(hsi_ns)
    np.random.seed(rd_seed)
    np.random.shuffle(label_ds)
    return hsi_ns,label_ds


def generate_patches(hsi,gt,patch_size):
    hsi_bands = hsi.shape[-1]
    pad_size = math.floor(patch_size / 2)
    hsi_pad = np.pad(hsi, ((pad_size, pad_size),(pad_size, pad_size), (0, 0)), 'reflect')

    nonezero_indexes = np.where(gt != 0)
    hsi_patches = np.empty((len(nonezero_indexes[0]),patch_size, patch_size,hsi_bands), dtype='float32')
    for x, y, idx in zip(nonezero_indexes[0], nonezero_indexes[1], range(len(nonezero_indexes[0]))):
        hsi_patches[idx] = hsi_pad[x:x+patch_size, y:y+patch_size, :]

    return hsi_patches


def split_data(hsi_ns,label_ds,num_train_data,num_test_data=None,shuffle = True, seed = 0):
    if shuffle:
        hsi_ns, label_ds = shuffle_hsi(hsi_ns, label_ds,seed)

    shape_hsi = hsi_ns.shape
    num_class = len(set(label_ds))
    train_x = np.zeros(0,dtype='float32').reshape(0,shape_hsi[1],shape_hsi[2],shape_hsi[3])
    train_y = np.zeros(0,dtype='int64')
    test_x = np.zeros(0,dtype='float32').reshape(0,shape_hsi[1],shape_hsi[2],shape_hsi[3])
    test_y = np.zeros(0,dtype='int64')
    all_train_indexes = []
    for i in range(num_class):
        indexes = np.where(label_ds == i)
        
        partidxes_train = indexes[0][:num_train_data]
        if num_test_data is None:
            partidxes_test = indexes[0][num_train_data:]
        else:
            partidxes_test = indexes[0][num_train_data:num_train_data+num_test_data]
        
        train_x = np.append(train_x,hsi_ns.take(partidxes_train,axis=0),axis=0)
        train_y = np.append(train_y,label_ds.take(partidxes_train))
        all_train_indexes.extend(partidxes_train)
        test_x = np.append(test_x,hsi_ns.take(partidxes_test,axis=0),axis=0)
        test_y = np.append(test_y,label_ds.take(partidxes_test))
        
    logging.info(f'train indexes: {all_train_indexes}')
    return train_x,train_y,test_x,test_y


def load_mat(mat_path):
    if 'Chikusei' in mat_path:
        hdfile = hdf5storage.loadmat(mat_path)
        mat = hdfile
        if 'GT' in mat:
            mat['GT'] = mat['GT'][0][0][0]
    else:
         mat = sio.loadmat(mat_path)
    
    return mat

def load_dataset(dataset_name,root_path,path_hsi = "",name_hsi = "",path_gt = "",name_gt = ""):
    if dataset_name !="":
        folder_path = Path(root_path,dataset_name)
        if not folder_path.exists():
            raise FileNotFoundError(folder_path)

        file_list = []
        for mat_file in folder_path.glob('*.mat'):
            file_list.append([str(mat_file),os.path.getsize(mat_file)])
            
        file_list.sort(key = lambda x: x[1])
        hsi = load_mat(file_list[-1][0])
        
        for vals in hsi.values():
            if isinstance(vals,np.ndarray):
                hsi = vals
                break
        
        gt = load_mat(file_list[0][0])
        for vals in gt.values():
            if isinstance(vals,np.ndarray):
                gt = vals
                break
    else:
        hsi = load_mat(path_hsi).get(name_hsi)
        gt = load_mat(path_gt).get(name_gt)
    return hsi,gt


def normalization(data):
    voxel = data.reshape(-1,data.shape[2])
    scaler = StandardScaler()
    voxel = scaler.fit_transform(voxel)
    return voxel.reshape(data.shape)

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu') 

def init():
    #matplotlib.use('TkAgg')
    set_logger()
    np.set_printoptions(linewidth=1000,precision=5)
    

def set_logger():
    log_path = 'output/logs'
    os.makedirs(log_path, exist_ok=True)
    file_path = time.strftime(f"{log_path}/%y-%m-%d.log")
    os.makedirs(Path(file_path).parent,exist_ok=True)
    f_h = logging.FileHandler(file_path,mode='a',encoding='utf-8')
    s_h = logging. StreamHandler()
    formater = logging.Formatter('%(asctime)s %(levelname)s %(module)s %(lineno)d %(funcName)s\t%(message)s','%m%d %H:%M:%S')
    f_h.setFormatter(formater)
    s_h.setFormatter(formater)
    
    logg = logging.getLogger()
    logg.addHandler(f_h)
    logg.addHandler(s_h)
    
    logg.setLevel(logging.INFO)
    return logg
    
    
def _apply_pca_producer(q: Queue,X, numComponents):
    newX = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[-1]))
    pca = PCA(n_components=numComponents,whiten=True)
    q.put(0)
    ret_X = pca.fit_transform(newX)
    ret_X = np.reshape(ret_X, (*X.shape[:2],-1))
    q.put(ret_X)

def apply_pca(X, numComponents):
    while True:
        q=Queue()
        p1=Process(target=_apply_pca_producer,args=(q,X,numComponents))
        p1.start()
        q.get()
        retry = 120
        while retry > 0 and q.empty():
            time.sleep(0.1)
            retry -= 0.1
        
        if q.empty():
            if p1.is_alive():
                p1.kill()
        else:
            ret_X = q.get()
            return ret_X

    
def get_dataloader(x,y,batch_size,n_workers,device='cpu'):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    x = x.view(-1,1,x.shape[1],x.shape[2],x.shape[3])
    x = x.to(device,non_blocking=True)
    y = y.to(device,non_blocking=True)
    data_set = data.TensorDataset(x, y)
    return data.DataLoader(data_set, batch_size, shuffle=True,pin_memory=True if device == 'cpu' else False,num_workers = n_workers)


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
    
def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = y_hat == y
    return float(torch.sum(cmp.type(y.dtype)))



def generate_dataset_preview(dataset_name, out_dir , dataset_folder = 'D:/Datasets/'):
    hsi, gt = load_dataset(dataset_name, dataset_folder)
    import spectral
    debug = False
    import matplotlib
    from matplotlib import pyplot as plt
    img_width = (6.1 - 0.1) / 2
    from PIL import Image
    if debug:
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
    hsi = apply_pca(hsi,3)
    rgb_img = spectral.get_rgb(hsi).astype(np.float32)
    if dataset_name == 'BW':
        rgb_img = np.rot90(rgb_img,1,(0,1))
    rgb_img = Image.fromarray((rgb_img*255).astype(np.uint8))
    img_width, img_height = rgb_img.size
    ratio = img_height / img_width
    if dataset_name == 'BW':
        rgb_img = rgb_img.resize((610, round(ratio * 610)))
        figure_height = (rgb_img.height*2 + 20*2+50)/100
    else:
        rgb_img = rgb_img.resize((300, round(ratio * 300)))
        figure_height = (rgb_img.height + 20)/100
        
    
    plt.rcParams['figure.figsize'] = (6.1,figure_height)
    #plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams["font.sans-serif"]=["SimHei", "Arial"] #设置黑体和Arial字体
    plt.rcParams["axes.unicode_minus"]=False #正常显示负号
    if dataset_name == 'BW':
        plt.subplot(2,1,1)
        
    else:
        plt.subplot(1,2,1)
    plt.imshow(rgb_img)
    plt.axis('off')
    
    plt.xticks([])
    plt.yticks([]) 
    plt.title('(a) 伪彩色图',fontdict={'fontsize':8}, y =0, pad = -15)
    
    if dataset_name == 'BW':
        ax = plt.subplot(2,1,2)
        gt = np.rot90(gt,1,(0,1))
    else:
        ax = plt.subplot(1,2,2)
    ax.matshow(gt,cmap=plt.cm.nipy_spectral)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([]) 
    
    plt.title('(b) 真值图',fontdict={'fontsize':8},y=0, pad= -15)
    plt.margins(1,1,tight=True)
    plt.subplots_adjust(0,0,1,1,hspace=0.1,wspace=0.1)
    if debug:
        plt.show(block=True)
    else:
        plt.savefig(Path(out_dir, f"{dataset_name}.jpg"),dpi=600,bbox_inches='tight',pad_inches=0)
        plt.close()


def save_classification_map(model,hsi_n,g_gt,dataset_name,fname):
    import matplotlib
    from matplotlib import pyplot as plt
    x_data = torch.from_numpy(hsi_n)
    x_data = x_data.view(-1,1,*x_data.shape[1:])
    x_data = x_data.to(torch.float32)
    
    data_set = data.TensorDataset(x_data)
    batch_size = 1024
    test_set = data.DataLoader(data_set, batch_size, shuffle=False,pin_memory=True)
    total_preds = np.zeros(0,dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for (X,) in test_set:
            X = X.cuda()
            preds = model(X).argmax(axis=1)
            total_preds = np.append(total_preds,preds.cpu().detach().numpy())
    model.train()
    data_len = g_gt.shape[0]*g_gt.shape[1]
    nonezero_idxes = np.nonzero(g_gt.reshape(data_len))
    predict_map = np.zeros(data_len)
    for i,j in zip(nonezero_idxes[0],range(len(total_preds))):
        predict_map[i] = total_preds[j]+1

    predict_map = predict_map.reshape(g_gt.shape[0],g_gt.shape[1])

    gt_path = Path('./output/figures/',dataset_name,'ground_truth.jpg')
    os.makedirs(gt_path.parent,exist_ok=True)
    gt_path = str(gt_path)
    matplotlib.use('Agg')
    if not os.path.exists(gt_path):
        plt.matshow(g_gt,cmap=plt.cm.nipy_spectral)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(fname=gt_path,bbox_inches='tight',dpi=600,pad_inches=0)
        plt.close()

    map_path = Path('./output/figures/',dataset_name,fname+'.jpg')
    os.makedirs(map_path.parent,exist_ok=True)
    map_path = str(map_path)
    plt.matshow(predict_map,cmap=plt.cm.nipy_spectral)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fname=map_path,bbox_inches='tight',dpi=600,pad_inches=0)
    plt.close()
    
    
def draw_cm(img_path_list,title_list,output_path,title_offset,hspace,wspace,num_class,debug=False):
    import matplotlib
    from matplotlib import pyplot as plt
    if debug:
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
    BW = 'BW' in output_path and num_class == 14
    title_fontsize = 6.5 if BW else 8
    subplot_name_sep = '\n' if BW else ' '
    plt.rcParams['figure.figsize'] = (5.91,5.91)
    plt.rcParams['font.sans-serif'] = "Arial"
    for idx,(img_path, img_title) in enumerate(zip(img_path_list,title_list)):
        if BW:
            plt.subplot(1,8,idx + 1)
        else:
            plt.subplot(2,4,idx + 1)
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([]) 
        plt.title(f'({chr(97+idx)}){subplot_name_sep}{img_title}',y=title_offset,fontdict={'fontsize':title_fontsize})
    
    if BW:
        plt.subplot(1,8,idx + 2)
    else:
        plt.subplot(2,4,idx + 2)
    img = np.ones_like(img)*255
    plt.imshow(img)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([]) 
    import matplotlib.patches as mpatches
    patches = []
    cmap = plt.cm.get_cmap('nipy_spectral',num_class+1)
    for idx in range(1,num_class+1):
        patches.append( mpatches.Patch(color=cmap(idx), label=str(idx)))
    
    
    legend_ncol = 1 if BW or num_class <= 10 else 2
    bbox_to_anchor = (0,1) if BW else (0,1.02)
    plt.legend(handles=patches,bbox_to_anchor=bbox_to_anchor,loc='upper left',ncol=legend_ncol,fontsize=7)
    
    plt.margins(0,0,tight=True)
    plt.subplots_adjust(0,0,1,1,hspace=hspace,wspace=wspace)
    if debug:
        plt.show(block=True)
    else:
        plt.savefig(output_path,dpi=600,bbox_inches='tight',pad_inches=0)
        plt.close()
    
def draw_classification_maps():
    for dataset_name, oa_str, map_args, num_class in [
        #['PU','80.98±3.75	81.09±4.22	87.81±3.03	83.60±4.78	87.41±2.34	91.80±3.28',(-0.11,-0.1,0.1),9],
        #['IP','65.66±3.58	64.99±3.69	73.78±4.91	73.33±1.80	73.99±5.04	78.70±2.57',(-0.2,-0.6,0.1),16],
        ['BW','94.39±2.70	95.26±1.94	91.94±2.93	94.93±2.66	92.66±2.97	95.69±2.45',(-0.08,-0.7,0.05),14],
        #['PC','96.11±1.00	96.51±0.76	96.76±0.68	96.88±0.30	97.36±0.26	97.39±0.37',(-0.11,-0.32,0.1),9],
        #['KSC','87.93±2.21	86.58±2.39	95.84±1.81	90.39±3.17	95.96±1.18	96.29±1.60',(-0.25,-0.67,0.1),13],
        ]:
        #names = ['Ground-truth','3-D CNN','HSI-CNN','DCFSL','HFSL','RandAugment','ADA','RAP','Extended-RAP']
        names = ['Ground-truth','3-D CNN+Extended-RAP','HSI-CNN+Extended-RAP','HSI-CR+Extended-RAP','EmbGCN', 'EmbGCN+RAP', 'EmbGCN+Extended-RAP']
        img_path_list = [f'./output/figures/{dataset_name}/ground_truth.jpg']
        for idx, oa_i in enumerate(oa_str.split('\t')):
            oa_i = float(oa_i[:5])
            min_diff= 100
            suiteble_file = ''
            name = names[idx+1]#.split(' ')[0]
            for file in Path(f'./output/figures/{dataset_name}').glob(f'{name}_*.jpg'):
                acc = float(file.stem[-5:].replace('_',''))
                if abs(acc - oa_i) < min_diff:
                    min_diff = abs(acc - oa_i)
                    suiteble_file = str(file)
    
            img_path_list.append(str(suiteble_file))
    
        names = ['Ground-truth','3-D CNN+E-RAP','HSI-CNN+E-RAP','HSI-CR+E-RAP','EmbGCN', 'EmbGCN+RAP', 'EmbGCN+E-RAP']
        draw_cm(img_path_list,names,f'./output/classification_map/classification_map_{dataset_name}.tif',*map_args,num_class,debug=False)
        

if __name__ == '__main__':
    #draw_classification_maps()
    # generate_dataset_preview('BW',r'D:\OneDrive\Document\Graduation Thesis\Figures\GCN\dataset')
    # generate_dataset_preview('PU',r'D:\OneDrive\Document\Graduation Thesis\Figures\GCN\dataset')
    # generate_dataset_preview('KSC',r'D:\OneDrive\Document\Graduation Thesis\Figures\GCN\dataset')
    # generate_dataset_preview('PC',r'D:\OneDrive\Document\Graduation Thesis\Figures\GCN\dataset')
    # generate_dataset_preview('IP',r'D:\OneDrive\Document\Graduation Thesis\Figures\GCN\dataset')
    pass