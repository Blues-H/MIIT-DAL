from typing import Any, Optional, Tuple
import torch
import torch.nn as nn


from .utils1 import Conv1d, MaxPool1d
from position_encoding import positional_encoding
from torch.autograd import Function
from GradientReverseFunction import GradientReverseFunction


class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class DeepSleepNetFeature_from_sleepyco(nn.Module):
    def __init__(self,domain_num=None):
        super(DeepSleepNetFeature_from_sleepyco, self).__init__()

        self.chn = 64

        
        self.dropout = nn.Dropout(p=0.5)
        self.path1 = nn.Sequential(Conv1d(2, self.chn, 50, 6, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(8, padding='SAME'),
                                   nn.Dropout(),
                                   Conv1d(self.chn, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn * 2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(4, padding='SAME')
                                   )
        self.path2 = nn.Sequential(Conv1d(2, self.chn, 400, 50, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(4, padding='SAME'),
                                   nn.Dropout(),
                                   Conv1d(self.chn, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn * 2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(2, padding='SAME'))

        self.compress = nn.Conv1d(self.chn*4, 128, 1, 1, 0)
        
        self.conv_c5 = nn.Conv1d(128, 128, 1, 1, 0)
        self.fc_class_classifier=nn.Sequential(nn.Linear(128 * 16, 128),nn.ReLU(),nn.Linear(128,5))
        self.fc_domain_classifier=nn.Sequential(nn.Linear(128 * 16,128),nn.ReLU(),nn.Linear(128,domain_num))
        self.grl=GRL()
        c=128
        t_d=16
        t=16
        c_d=128
        n_heads1=4
        n_heads2=4
        
        dropout=0.1
        self.W1_pos = positional_encoding('sincos', False, c, t_d)
        self.W2_pos = positional_encoding('sincos', False, t, c_d)
        spatial_encoder_layer = nn.TransformerEncoderLayer(d_model=t_d, nhead=n_heads1, dim_feedforward=2*t_d, dropout=dropout,batch_first=True)
        temporal_encoder_layer=nn.TransformerEncoderLayer(d_model=c_d, nhead=n_heads2, dim_feedforward=2*c_d, dropout=dropout,batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(spatial_encoder_layer, num_layers=1)
        self.temporal_transformer = nn.TransformerEncoder(temporal_encoder_layer, num_layers=1)
        
        
        
        
        
    def forward(self, x):
        out = []
        x1 = self.path1(x)
        x2 = self.path2(x)
        
        x2 = torch.nn.functional.interpolate(x2, x1.size(2))
        
        c5 = self.compress(torch.cat([x1, x2], dim=1))
        out = self.conv_c5(c5)
        
        out = self.spatial_transformer(out+self.W1_pos)
        out = self.temporal_transformer(out.permute(0,2,1)+self.W2_pos)
        out=out.permute(0,2,1)
        out=out.reshape(out.shape[0], -1)
        out1=self.fc_class_classifier(out)
        out2=self.fc_domain_classifier(self.grl(out))


        return out1,out2
    
