import logging
import sys
import math
import inspect
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from bert import make_model as bert_model
from typing import Callable, Optional
import numpy as np

from PatchTST_backbone import PatchTST_backbone
from PatchTST_layers import series_decomp


def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    x = Variable(x)

    return x

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """

    #nn.init.xavier_uniform_(layer.weight)

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 3:
        (n_out, n_in, width) = layer.weight.size()
        n = n_in * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


class DilatedResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, kernel_size, dilation, bias):
        super(DilatedResidualBlock, self).__init__()
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.dilated_conv = nn.Conv1d(residual_channels, 2 * dilation_channels, kernel_size=kernel_size, dilation=dilation, padding=dilation, bias=bias)
        self.mixing_conv = nn.Conv1d(dilation_channels, residual_channels + skip_channels, kernel_size=1, bias=False)
        self.init_weights()

    def init_weights(self):
        init_layer(self.dilated_conv)
        init_layer(self.mixing_conv)

    def forward(self, data_in):

        out = self.dilated_conv(data_in)
        out1 = out.narrow(-2, 0, self.dilation_channels)
        out2 = out.narrow(-2, self.dilation_channels, self.dilation_channels)
        tanh_out = torch.tanh(out1)
        sigm_out = torch.sigmoid(out2)
        data = torch.mul(tanh_out, sigm_out)
        data = self.mixing_conv(data)
        res = data.narrow(-2, 0, self.residual_channels)
        skip = data.narrow(-2, self.residual_channels, self.skip_channels)
        res = res + data_in
        return res, skip


class WaveNet(nn.Module):

    def __init__(self, layers=6, kernel_size=3, residual_channels=32, dilation_channels=32, skip_channels=128, to_binary=False):
        super(WaveNet, self).__init__()
        assert kernel_size % 2 == 1, f'kernel_size ({kernel_size}) must be odd'
        self.kernel_size = kernel_size # has to be odd integer, since even integer may break dilated conv output size
        self.to_binary = to_binary
        self.seq_len = (2 ** layers - 1) * (kernel_size - 1) + 1

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels

        self.causal_conv = nn.Conv1d(1, residual_channels, kernel_size=1, bias=True)
        self.blocks = [DilatedResidualBlock(residual_channels, dilation_channels, skip_channels, kernel_size, 2**i, True)
                       for i in range(layers)]
        for i, block in enumerate(self.blocks):
            self.add_module(f"dilatedConv{i}", block)
        self.penultimate_conv = nn.Conv1d(skip_channels, skip_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True)
        self.final_conv = nn.Conv1d(skip_channels, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.causal_conv)
        init_layer(self.penultimate_conv)
        init_layer(self.final_conv)

    def forward(self, data_in):
        data_in = data_in.view(data_in.shape[0], 1, data_in.shape[1])
        data_out = self.causal_conv(data_in)
        skip_connections = []
        for block in self.blocks:
            data_out, skip_out = block(data_out)
            skip_connections.append(skip_out)
        skip_out = skip_connections[0]
        for skip_other in skip_connections[1:]:
            skip_out = skip_out + skip_other
        data_out = F.relu(skip_out)
        data_out = self.penultimate_conv(data_out)
        #data_out = F.relu(data_out)
        data_out = self.final_conv(data_out)
        data_out = data_out.narrow(-1, self.seq_len//2, data_out.size()[-1]-self.seq_len+1)
        data_out = data_out.view(data_out.shape[0], data_out.shape[2])
        if self.to_binary:
            return torch.sigmoid(data_out)
        return data_out


class BGRU(nn.Module):

    def __init__(self, seq_len=511, to_binary=False):

        super(BGRU, self).__init__()

        self.seq_len = seq_len

        self.to_binary = to_binary

        self.bgru = nn.GRU(input_size=1, hidden_size=64, num_layers=3, bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc_final = nn.Linear(128, 1)

        self.init_weights()

    def _init_param(self, param):

        if param.ndimension() == 1:
            param.data.fill_(0.)

        elif param.ndimension() == 2:
            n = param.size(-1)
            std = math.sqrt(2. / n)
            scale = std * math.sqrt(3.)
            param.data.uniform_(-scale, scale)

    def init_weights(self):

        for param in self.bgru.parameters():
            self._init_param(param)

        init_layer(self.fc_final)

    def forward(self, input):

        x = input
        x = x.view(x.shape[0], x.shape[1], 1)
        '''(batch_size, time_steps, 1)'''

        (x, h) = self.bgru(x)
        '''x: (batch_size, time_steps, feature_maps)'''

        x = self.fc_final(x)
        '''(batch_size, time_steps, 1)'''

        x = x.view(x.shape[0 : 2])
        '''(batch_size, time_steps)'''

        seq_len = self.seq_len
        width = x.shape[1] - seq_len + 1
        output = x[:, seq_len // 2 : seq_len // 2 + width]
        '''(batch_size, width)'''

        if self.to_binary:
            return torch.sigmoid(output)

        return output


class Transformer(nn.Module):
    def __init__(self, configs, seq_len, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs['enc_in']
        context_window = seq_len
        target_window = configs['pred_len']
        
        n_layers = configs['e_layers']
        n_heads = configs['n_heads']
        d_model = configs['d_model']
        d_ff = configs['d_ff']
        dropout = configs['dropout']
        fc_dropout = configs['fc_dropout']
        head_dropout = configs['head_dropout']
        
        individual = configs['individual']
    
        patch_len = configs['patch_len']
        stride = configs['stride']
        padding_patch = configs['padding_patch']
        
        revin = configs['revin']
        affine = configs['affine']
        subtract_last = configs['subtract_last']
        
        decomposition = configs['decomposition']
        kernel_size = configs['kernel_size']
        self.seq_len = seq_len
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1)           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        x = torch.squeeze(x)
        return x
    

MODELS = {cname: (cls, inspect.getfullargspec(cls.__init__).args[1:])
          for cname, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
          if issubclass(cls, nn.Module)}
