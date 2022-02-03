import torch
from torch import nn
from torch.nn import functional as F

class SimpleFunc(nn.Module):
    
    def __init__(self, dim_model, dim_hid, num_heads, dropout=0.1, activation=F.relu, batch_first=True, device=None) -> None:
        assert dim_model % num_heads == 0
        factory_kwargs = {'device': device}
        super(SimpleFunc, self).__init__()
        self.attention = nn.MultiheadAttention(dim_model, num_heads, batch_first=batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(dim_model, dim_hid, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_hid, dim_model, **factory_kwargs)
        self.activation = activation

        self.lin_k = nn.Linear(dim_model, dim_model, **factory_kwargs)
        self.lin_q = nn.Linear(dim_model, dim_model, **factory_kwargs)
        self.lin_v = nn.Linear(dim_model, dim_model, **factory_kwargs)
        #self.code = nn.parameter.Parameter(F.normalize(torch.randn(dim_model), dim=0))
        self.type_vec = nn.parameter.Parameter(F.normalize(torch.randn(dim_model), dim=0), requires_grad=False).to(device)
    
    def get_mask(self, src, eps = 1e-6):
        """
        src: tensor in shape [batch, src_len, dim_model]
        """
        threshold = 1 / src.shape[1] + eps
        normalized_x = F.normalize(src, dim=2)
        scores = (normalized_x @ self.type_vec)
        scores = F.softmax(scores, 1)
        mask = 1 - ((scores > threshold).type(torch.uint8))
        return mask


    def forward(self, src, func_mask, key_padding_mask=None, attn_mask=None):
        """
        src: tensor in shape [batch, src_len, dim_model]
        src_mask: tensor in shape [batch, src_len]
        """
        temp = src 
        mask = func_mask
        if key_padding_mask is not None:
          mask = (mask.logical_or(key_padding_mask)).type(torch.uint8)
        if attn_mask is not None:
          mask = (mask.logical_or(attn_mask)).type(torch.uint8)
        query = self.lin_q(temp)
        key = self.lin_k(temp)
        value = self.lin_k(temp)
        attn_output, attn_output_weights = self.attention(temp, temp, temp, key_padding_mask=mask)#, attn_mask=mask)
        temp = self.linear1(attn_output)
        temp = self.dropout(self.activation(temp))
        temp = self.activation(self.linear2(temp))
        output_mask = mask.unsqueeze(-1).repeat(1,1,src.shape[-1])
        temp = temp.masked_fill(output_mask == 1 , 0)
        temp = src + temp
        return temp


class SimpleFuncsRow(nn.Module):

    def __init__(self, dim_model, dim_hid, num_funcs = 8, num_heads=8, dropout=0.1, activation=F.relu, device=None) -> None:
        factory_kwargs = {'dim_model': dim_model, 'dim_hid':dim_hid, 'device': device, 'dropout': dropout, 'activation': activation,\
                          'num_heads':num_heads}
        super(SimpleFuncsRow, self).__init__()
        self.funcs = nn.ModuleList([SimpleFunc(**factory_kwargs) for i in range(num_funcs)])
        self.type_inference = nn.Sequential(nn.Linear(dim_model, dim_hid, device=device), nn.ReLU(), nn.Linear(dim_hid, dim_model, device=device)).to(device)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        """
        src: tensor in shape [batch, src_len, dim_model]
        """
        factory_kwargs = {'key_padding_mask': key_padding_mask, 'attn_mask':attn_mask}
        temp = src
        for func in self.funcs:
          src_types = F.normalize(self.type_inference(src), dim=2)
          src_mask = func.get_mask(src_types)
          temp = temp + func(src, src_mask, **factory_kwargs)
        return temp
  
class SimpleRowIter(nn.Module):
    def __init__(self, dim_model, dim_hid, num_iter = 4 , num_funcs = 8, num_heads=8, dropout=0.1, activation=F.relu, device=None):
        factory_kwargs = {'dim_model': dim_model, 'dim_hid':dim_hid, 'num_funcs':num_funcs,\
                          'device': device, 'dropout': dropout, 'activation': activation, 'num_heads':num_heads}
        super(SimpleRowIter, self).__init__()
        self.funcs_row =  SimpleFuncsRow(**factory_kwargs)
        self.num_iter = num_iter

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        factory_kwargs = {'key_padding_mask': key_padding_mask, 'attn_mask':attn_mask}
        temp = src 
        for iter_num in range(self.num_iter):
          temp = temp + self.funcs_row(temp, **factory_kwargs)
        return temp

class SimpleEncoder(nn.Module):
    def __init__(self, dim_model, dim_hid, num_layer = 6, num_iter = 4 , num_funcs = 8, num_heads=8, dropout=0.1, activation=F.relu, device=None):
        factory_kwargs = {'dim_model': dim_model, 'dim_hid':dim_hid, 'num_iter':num_iter, 'num_funcs':num_funcs,\
                          'device': device, 'dropout': dropout, 'activation': activation, 'num_heads':num_heads}
        super(SimpleEncoder, self).__init__()
        self.layers = nn.ModuleList([SimpleRowIter(**factory_kwargs) for i in range(num_layer)])
    
    def forward(self, src, key_padding_mask=None, attn_mask=None):
      factory_kwargs = {'key_padding_mask': key_padding_mask, 'attn_mask':attn_mask}
      temp = src
      i = 0
      for layer in self.layers:
        temp = temp + layer(temp, **factory_kwargs)
      return temp
