import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from rap import RandomAugmentPipe
from gcn import EmbGCN


class AugNet(nn.Module):
    def __init__(self, model_name:str, num_band, args) -> None:
        super().__init__()
        model_name = model_name.lower()
        if model_name.startswith('embgcn'):
            self.model = EmbGCN(args['num_class'],args['num_emb'], args['knn'],args['select'],args['dropout'],True)
        elif model_name.startswith('hsi-cr'):
            self.model = EmbGCN(args['num_class'],args['num_emb'], args['knn'],args['select'],args['dropout'],True)
        elif model_name.startswith('3-d cnn'):
             self.model = _3D_CNN(num_band, args['num_class'], args['patch_size'])
        elif model_name.startswith('hsi-cnn'):
             self.model = HSI_CNN(num_band, args['num_class'], args['patch_size'])
        else:
            raise NotImplementedError()
            
        self.enable_rap = args['enable_rap']
        if self.enable_rap:
            augs = args.get('augs')
            augs = list(augs.keys())
            aug_args = args.get('aug_args')
            
            self.rap = RandomAugmentPipe(num_band,args['patch_size'],torch.device('cuda:0'),augs,aug_args)

    def forward(self, x):
        if self.enable_rap and self.training:
            x = self.rap(x)

        x_gcn = self.model(x)
        
        return x_gcn
    
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv3d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(num_channels)
        self.bn2 = nn.BatchNorm3d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class HSI_CR(nn.Module):
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv3d:
            nn.init.xavier_uniform_(m.weight)
    
    def __init__(self, num_band,num_class,dropout=0.5):
        super(HSI_CR, self).__init__()
        self.num_class = num_class
        self.dropout = dropout

        self.part1 = nn.Sequential(
                nn.Conv3d(1,24,(1,1,5),(1,1,2)),
                nn.BatchNorm3d(24),
                nn.ReLU(),
                nn.Conv3d(24,12,(1,1,7),(1,1,1),padding=(0,0,1)),
                nn.BatchNorm3d(12),
                nn.ReLU(),
                nn.Conv3d(12,12,(1,1,7),(1,1,1),padding=(0,0,1)),
                nn.BatchNorm3d(12),
                nn.ReLU(),
                nn.Conv3d(12,12,(1,1,7),(1,1,1),padding=(0,0,1)),
                nn.BatchNorm3d(12),
                nn.ReLU())
        
        self.part2 = nn.Sequential(nn.BatchNorm3d(50),
                nn.ReLU())
        
        self.part3 = nn.Sequential(Residual(1,12),
                Residual(12,12),
                nn.AdaptiveAvgPool3d((1,1,50)),
                nn.ReLU(), 
                nn.Flatten(),
                
                nn.Linear(600, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_class)
                )
        self.part4 = nn.Conv3d(12,50,(1,1,int(math.ceil(num_band/2-14))),(1,1,1))
        self.apply(self.init_weights)
        
        
    def forward(self, input):
        res1 = self.part1(input)
        res2 = self.part4(res1)
        res3 = self.part2(res2)
        res4 = torch.transpose(res3,1,4)
        res5 = self.part3(res4)
        return res5
    
class _3D_CNN(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """
    # 6 layer network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(_3D_CNN, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1
            )
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0
            )
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )

        # self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x : torch.Tensor):
        x = x.permute(0,1,4,2,3)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x

class HSI_CNN(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=3, n_planes=90):
        super(HSI_CNN, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Conv3d(1, 90, (24, 3, 3), padding=0, stride=(9, 1, 1))
        self.conv2 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            b = x.size(0)
            x = x.view(b, 1, -1, self.n_planes)
            x = self.conv2(x)
            _, c, w, h = x.size()
        return c * w * h

    def forward(self, x):
        x = x.permute(0,1,4,2,3)
        x = F.relu(self.conv1(x))
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
