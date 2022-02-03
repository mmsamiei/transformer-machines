import torch
from torch import nn
from torch.nn import functional as F

class DummyFunc(nn.Module):
    
    def __init__(self, dim_model, dim_hid, dropout=0.1, activation=F.relu, device=None) -> None:
        factory_kwargs = {'device': device}
        super(DummyFunc, self).__init__()
        self.linear1 = nn.Linear(dim_model, dim_hid, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_hid, dim_model, **factory_kwargs)
        self.activation = activation

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
        mask = (scores > threshold).type(torch.uint8)
        return mask


    def forward(self, src, src_mask=None):
        """
        src: tensor in shape [batch, src_len, dim_model]
        src_mask: tensor in shape [batch, src_len]
        """
        temp = src 
        temp = self.linear1(temp)
        temp = self.dropout(self.activation(temp))
        temp = self.activation(self.linear2(temp))
        mask = src_mask.unsqueeze(-1).repeat(1,1,src.shape[-1])
        temp = temp.masked_fill(mask == 0 , 0)
        temp = src + temp
        return temp


class DummyFuncsRow(nn.Module):

    def __init__(self, dim_model, dim_hid, num_funcs = 8 , dropout=0.1, activation=F.relu, device=None) -> None:
        factory_kwargs = {'dim_model': dim_model, 'dim_hid':dim_hid, 'device': device, 'dropout': dropout, 'activation': activation}
        super(DummyFuncsRow, self).__init__()
        self.funcs = nn.ModuleList([DummyFunc(**factory_kwargs) for i in range(num_funcs)])
        self.type_inference = nn.Sequential(nn.Linear(dim_model, dim_hid), nn.ReLU(), nn.Linear(dim_hid, dim_model)).to(device)

    def forward(self, src):
        """
        src: tensor in shape [batch, src_len, dim_model]
        """
        temp = src
        for func in self.funcs:
          src_types = F.normalize(self.type_inference(src), dim=2)
          src_mask = func.get_mask(src_types)
          temp = temp + func(src, src_mask)
        return temp
  
class DummyRowIter(nn.Module):
    def __init__(self, dim_model, dim_hid, num_iter = 4 , num_funcs = 8 , dropout=0.1, activation=F.relu, device=None):
        factory_kwargs = {'dim_model': dim_model, 'dim_hid':dim_hid, 'num_funcs':num_funcs,\
                          'device': device, 'dropout': dropout, 'activation': activation}
        super(DummyRowIter, self).__init__()
        self.funcs_row =  DummyFuncsRow(**factory_kwargs)
        self.num_iter = num_iter

    def forward(self, src):
        temp = src 
        for iter_num in range(self.num_iter):
          temp = temp + self.funcs_row(temp)
        return temp

class DummyEncoder(nn.Module):
    def __init__(self, dim_model, dim_hid, num_layer = 6, num_iter = 4 , num_funcs = 8 , dropout=0.1, activation=F.relu, device=None):
        factory_kwargs = {'dim_model': dim_model, 'dim_hid':dim_hid, 'num_iter':num_iter, 'num_funcs':num_funcs,\
                          'device': device, 'dropout': dropout, 'activation': activation}
        super(DummyEncoder, self).__init__()
        self.layers = nn.ModuleList([DummyRowIter(**factory_kwargs) for i in range(num_layer)])
    
    def forward(self, src):
      temp = src
      i = 0
      for layer in self.layers:
        temp = temp + layer(temp)
      return temp
