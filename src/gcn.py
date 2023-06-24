import numpy as np
import torch
import torch_geometric.nn as tgnn
from timm.models.layers import trunc_normal_
from torch import nn
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected


class HSIPatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear,nn.Conv2d,nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    
    def __init__(self,embed_dim=256, select= 'max', dropout = 0.5):
        super().__init__()
        self.avg_size = (4,4,4)
        self.mid_size = 96
        self.proj1 = nn.Sequential(
            nn.Conv3d(1,embed_dim,(3,3,12),stride=(2,2,12)),
                                   nn.BatchNorm3d(embed_dim),
                                   nn.ReLU())
        self.proj2 = nn.Sequential(nn.Conv3d(1,self.mid_size,kernel_size=(3,3,13),padding=0,stride=3),
                                   nn.BatchNorm3d(self.mid_size),
                                   nn.ReLU())

        self.proj3 = nn.Sequential(nn.Conv3d(self.mid_size,embed_dim,kernel_size=(3,3,13),padding=0,stride=2,groups=self.mid_size),
                                   nn.BatchNorm3d(embed_dim),
                                   nn.ReLU())
        
        self.aap = nn.AdaptiveAvgPool3d(self.avg_size)
        self.feature_factor = nn.Parameter(torch.ones(2,device='cuda:0'), requires_grad=True)
        
        self.fc = nn.Sequential(nn.Linear(embed_dim*np.prod(self.avg_size), embed_dim),
                                nn.Dropout(dropout))
        
        self.apply(self._init_weights)
        if select:
            self.select = torch.max
        elif select == 'min':
            self.select = torch.min
        elif select == 'mean':
            self.select = torch.mean
        else:
            raise NotImplementedError()
    
    
    def get_patch(self, x,patch_size):
        cur_offset = (x.shape[-2] - patch_size ) // 2
        X = x[:,:,cur_offset:patch_size+cur_offset,cur_offset:patch_size+cur_offset,:]
        return X
    
    def forward(self, x):
        x_min = self.get_patch(x,5)
        x_min = self.proj1(x_min)
        x = self.proj2(x)
        x = self.proj3(x)
        x_min = self.aap(x_min)
        x = self.aap(x)
        x_min = x_min.flatten(1)
        x = x.flatten(1)
        
        x = torch.concat([torch.reshape(x_min,(1,*x_min.shape)),torch.reshape(x,(1,*x.shape))])
        x = self.select(x,dim=0)[0] #max min mean
        x = self.fc(x)
        return x


class EmbGCN(torch.nn.Module):
    def _init_weights(self, m):
        if isinstance(m, (tgnn.GCNConv)):
            nn.init.xavier_normal_(m.lin.weight)
            nn.init.constant_(m.bias, 0)

    def __init__(self,num_class, num_emb = 768, knn = 10, select = 'max', dropout = 0.5, enable_head=True):
        super().__init__()
        
        self.patch_emb = HSIPatchEmbed(num_emb, select, dropout)
        self.conv1 = tgnn.Sequential('x, edge_index',[(tgnn.GCNConv(num_emb, 256),'x, edge_index -> x'),
                                                      nn.ReLU(),
                                                      (tgnn.BatchNorm(256), 'x -> x')
                                                      ])
        
        self.conv2 = tgnn.Sequential('x, edge_index',[(tgnn.GCNConv(256, 64),'x, edge_index -> x'),
                                                      nn.ReLU(),
                                                       (tgnn.BatchNorm(64), 'x -> x')
                                                       ])
        if enable_head:
            self.conv3 = tgnn.GCNConv(64, num_class)
        self.enable_head = enable_head
        self.knn = knn
    
    def forward(self,x):
        x = self.patch_emb(x)
        edge_index = knn_graph(x, self.knn)
        edge_index = to_undirected(edge_index)
        edge_index = edge_index.cuda()
    
        x = self.conv1(x,edge_index)
        x = self.conv2(x,edge_index)
        if self.enable_head:
            x = self.conv3(x,edge_index)
        return x