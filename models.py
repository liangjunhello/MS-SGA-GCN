import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F     # 激励函数都在这
import time  # 引入time模块
 
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SGA_GCN_Resnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None):
        superSGA_GCN_Resnetself).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool)
        self.l1=model.layer1   
        self.l2=model.layer2
        self.l3=model.layer3
        self.l4=model.layer4
        
        self.num_classes = num_classes 
      
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(2048)
        )
        
        self.class_trans = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(num_classes)
        )
        
        self.z_relu=nn.ReLU(inplace=True)

        self.gc1 =  GraphConvolution(in_channel, 1024)
        self.gc2 =  GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature , inp):
            
            feature  = self.features(feature)
            feature =self.l1(feature )       
            feature =self.l2(feature )
            feature =self.l3(feature )
            feature = self.l4(feature )

            inp = inp[0]
            adj = gen_adj(self.A).detach()
            x = self.gc1(inp, adj)
            x = self.relu(x)
            x = self.gc2(x, adj)    
            
            featureCST = feature 
            #att 部分
            feature = self.transform(feature)
            feature = feature.reshape((feature.shape[0],feature.shape[1],-1))       
            z = self.att(feature,x)
            z = z.reshape((feature.shape[0],-1,z.shape[1]))
            
            featureCST  =featureCST.reshape((featureCST .shape[0],featureCST .shape[1],-1))#b 2048 196
            featureCST  =self.class_trans(featureCST )
            
            fc=z.mul(featureCST )
            fc=torch.sum(fc, dim=2)
            return fc
   
    def att(self,feature,x):
        z = None 
        li=feature.shape[0]
        lj=x.shape[0]
        for i in range(li):
                cosine_dis = self.z_relu(cos_similarity(x,feature[i].transpose(0, 1)))
                cosine_dis = F.normalize(cosine_dis, p=1, dim=1)
                if(z is not None):
                     z=torch.cat((z,cosine_dis), 0)
                else:
                     z=cosine_dis
        return z
    

def SGAGCNResnet(num_classes, t, pretrained=True, adj_file=None, in_channel=300):
    model_up = models.resnet101(pretrained=False)
    state = torch.load('./pth/resnet101.pth')
    model_up.load_state_dict(state)
    return SGA_GCN_Resnet(model_up,num_classes=num_classes,t=t, adj_file=adj_file, in_channel=in_channel)
