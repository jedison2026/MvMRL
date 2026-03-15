import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
from torch_geometric.nn import global_max_pool
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from new_pubchemfp import GetPubChemFPs  ## source.


###################################### 添加fingerprint部分 ####################################################
class FPN(nn.Module):
    def __init__(self, fp_2_dim=512, dropout_fpn=0.5, fp_dim = 1489, hidden_dim_fpn=96, fp_changebit=0):
        super(FPN, self).__init__()
        self.fp_2_dim=fp_2_dim  ## 512，The dim of the second layer in fpn.
        self.dropout_fpn = dropout_fpn  ## 0.0，The dropout of gnn.
        ## self.cuda = cuda  ## ？？ cpu也可以跑
        ## self.fp_type = fp_type
        self.hidden_dim_fpn = hidden_dim_fpn ## 300，The dim of hidden layers in model.
        self.fp_dim = fp_dim

        self.fc1=nn.Linear(self.fp_dim, self.fp_2_dim) ## 线性变换
        self.act_func = nn.ReLU()  ## 激活函数
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim_fpn)
        self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, x):  ## def forward(self, data.smile):

        ## fpn_out = self.fc1(fp_list)  ## (1489,512)
        fpn_out = self.fc1(x.cuda())
        fpn_out = self.dropout(fpn_out) ## (0.0)
        fpn_out = self.act_func(fpn_out) ## Relu()
        fpn_out = self.fc2(fpn_out)  ## (512,300)
        return fpn_out  ## 最终得到一个处理好的300维的分子指纹特征矩阵fp_list，一行为一个分子的指纹特征(x,300)

################################# 添加smile序列 ##############################################
## SE模块
class SE_Block(nn.Module):                         # Squeeze-and-Excitation block
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.sigmoid(x)
        return out


class Smi_CNN(torch.nn.Module):
    def __init__(self, n_output=1,n_filters=32, embed_dim=128, output_dim=96, dropout=0.2):  ## output_dim=128改为96
        super(Smi_CNN, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        ## smile序列部分
        # 1D convolution on smile sequence
        self.embedding_xt_smile = nn.Embedding(100, embed_dim)
        #         self.fnn1 = nn.Linear(embed_dim, embed_dim)
        ## kernel_size = 2
        self.conv_xt2 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=2)
        self.fc_xt2 = nn.Linear(32 * 127, output_dim)
        ## kernel_size = 4
        self.conv_xt4 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=4)
        ## self.SE1 = SE_Block(n_filters)
        self.fc_xt4 = nn.Linear(32 * 125, output_dim)
        ## kernel_size = 8
        self.conv_xt8 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=8)
        self.SE1 = SE_Block(n_filters)
        self.fc_xt8 = nn.Linear(32 * 121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(384, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(288, output_dim)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

    def forward(self, data):

        ## smile序列
        smil2vec = data.smil2vec  ## [64,100]
        embedded_xt1 = self.embedding_xt_smile(smil2vec)  ## [64, 100, 128]，将smil2vec转换成long()类型
        # 跳连，将嵌入层加入到FNN中
        #         embedded_xt_fnn = self.fnn1(embedded_xt)
        #         embedded_xt = embedded_xt + embedded_xt_fnn
        ## kernel_size = 2
        conv_xt2 = self.conv_xt2(embedded_xt1)
        conv_xt2 = self.relu(conv_xt2)
        SE_smi2 = self.SE1(conv_xt2)
        conv_xt2 = conv_xt2 * SE_smi2  ## [64,32,127]
        ## kernel_size = 4
        conv_xt4 = self.conv_xt4(embedded_xt1)  ##
        conv_xt4 = self.relu(conv_xt4)  ##
        SE_smi4 = self.SE1(conv_xt4)  ##
        conv_xt4 = conv_xt4 * SE_smi4  ## [64,32,125]
        ## kernel_size = 8
        conv_xt8 = self.conv_xt8(embedded_xt1)  ## [64,32,121]
        conv_xt8 = self.relu(conv_xt8)  ## [64,32,121]
        SE_smi8 = self.SE1(conv_xt8)  ## [64,32,1]
        conv_xt8 = conv_xt8 * SE_smi8  ## [64,32,121]
        # flatten
        ## kernel_size = 2
        x2 = conv_xt2.view(-1, 32 * 127)
        x2 = self.fc_xt2(x2)
        ## kernel_size = 4
        x4 = conv_xt4.view(-1, 32 * 125)
        x4 = self.fc_xt4(x4)
        ## kernel_size = 8
        x8 = conv_xt8.view(-1, 32 * 121)  ## [64,96]
        x8 = self.fc_xt8(x8)  ## [64,96]

        x = torch.cat([x2, x4, x8], dim=1)  ## [64, 288]
        x = self.fc3(x)
        ##x = global_max_pool(xd, data.batch)
        return x
####################### 添加self-attention和cross-attention部分 ##################################
class AttentionBlock(nn.Module):
    """ A class for attention mechanisn with QKV attention """

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()  ## 删除.cuda()

    def forward(self, query, key, value, mask=None):
        """
        :Query : A projection function
        :Key : A projection function
        :Value : A projection function
        Cross-Att: Query and Value should always come from the same source (Aiming to forcus on), Key comes from the other source  ## 这里写错了，应该是Query来自其它源
        Self-Att : Both three Query, Key, Value come form the same source (For refining purpose)
        """

        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)

        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2, 3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix


class CrossAttentionBlock(nn.Module):

    def __init__(self,filter_num=32):   ## 删除 , args: TrainArgs
        super(CrossAttentionBlock, self).__init__()
        self.fp_encoder = FPN()  ## fp_2_dim=512,hidden_dim=96,dropout_fp=0.5, fp_dim=1489

        self.smi_encoder = Smi_CNN()

        self.ligand_encoder = GraphDenseNet(num_input_features=40, out_dim=filter_num * 3, block_config=[8, 8, 8],
                                            bn_sizes=[2, 2, 2])  ## num_input_features由87改为7，再又7改为87，再又87改为46,又改成40


        self.att = AttentionBlock(hid_dim=96, n_heads=1, dropout=0.1)  ## hid_dim=args.hidden_size改为300改成96，dropout=args.dropout改为 0.1

    def forward(self, data):
        """
            :graph_feature : A batch of 1D tensor for representing the Graph information from compound
            :morgan_feature: A batch of 1D tensor for representing the ECFP information from compound
            :sequence_feature: A batch of 1D tensor for representing the information from protein sequence
            :fp_x: A batch of 1D tensor for representing the information from fingerprint
            :ligand_x: A batch of 1D tensor for representing the infromation from graph
            :smi_x: A batch of 1D tensor for represneting the information from smiles sequences
        """
        fp = data.fp
        ## smiles = data.smiles  ## fp部分改
        ## fp_x = self.fp_encoder(smiles)  ## fp部分改
        fp_x = self.fp_encoder(fp)  ## fp部分改
        ligand_x = self.ligand_encoder(data)  ## [512,96]  消融
        smi_x = self.smi_encoder(data)  ## 消融

        # cross attention for compound information enrichment
        ## 消融
        smi_x1 = smi_x + self.att(fp_x, smi_x, smi_x)  ## 这里实际是两部分，一部分仍然是graph_feature，另一部分是与morgan_feature进行cross attention之后的特征
        fp_x1 = fp_x + self.att(smi_x, fp_x, fp_x)
        # self-attention
        ## smi_x = self.att(smi_x, smi_x, smi_x)

        # cross-attention for interaction
        ligand_x1 = ligand_x + self.att(fp_x, ligand_x, ligand_x)  ## 消融
        fp_x2 = fp_x + self.att(ligand_x, fp_x, fp_x)  ## 消融

        ## self-attention
        ## ligand_x = self.att(ligand_x, ligand_x, ligand_x)

        return ligand_x1, fp_x2, smi_x1, fp_x1  ## output

#################################################################################################
## 化合物部分
class NodeLevelBatchNorm(_BatchNorm):
    r"""
    Applies Batch Normalization over a batch of graph data.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    batch_nodes_dim: all nodes of a batch graph
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  ## x.shape:[9294,7],torch.int64  edge_index.shape:[2,19240],torch.int64 batch.shape:[9294],torch.int64
        ## x = x.to(torch.float32) ## 格式变换
        data.x = F.relu(self.norm(self.conv(x, edge_index)))

        return data


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features

        data = self.conv1(data)

        return data

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]

        data = self.bn_function(data)
        data = self.conv2(data)

        return data


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)

    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data


class GraphDenseNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config=(3, 3, 3, 3), bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i + 1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i + 1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        data = self.features(data)
        x = gnn.global_mean_pool(data.x, data.batch)
        x = self.classifer(x)

        return x


class MvMRL(nn.Module):
    def __init__(self, out_dim, filter_num=32):
        super().__init__()

        self.att_encoder = CrossAttentionBlock()
        self.out_dim = out_dim

        ## 分类器
        self.classifier = nn.Sequential(
            nn.Linear(filter_num*2*2* 3, 1024),  ## 原nn.Linear(filter_num*2*3, 1024)，还是把*2给去掉
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, data):

        ligand_x1, fp_x2, smi_x1, fp_x1= self.att_encoder(data)  ## smi_x1, ligand_x1, fp_x1, fp_x2

        x = torch.cat([ligand_x1, fp_x2, smi_x1, fp_x1], dim=1)  ##smi_x1, ligand_x1, fp_x1, fp_x2 dim = -1
        x = self.classifier(x)

        return x

