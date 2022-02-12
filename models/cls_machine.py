import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class ClsMachine(nn.Module):

    def __init__(self, backbone_machine, num_cls, input_dim, max_len=128):
        
        super(ClsMachine, self).__init__()
        self.backbone_machine = backbone_machine
        self.num_cls = num_cls
        self.max_len = max_len
        self.cls_embeddings = \
            nn.Parameter(torch.randn(self.num_cls,self.backbone_machine.dim_model))
        
        self.token_embedding = nn.Linear(input_dim, self.backbone_machine.dim_model)

        self.pos_embeddings = \
            nn.Parameter(torch.randn(self.max_len,self.backbone_machine.dim_model))
    
    def add_cls(self, num_new_cls):
        new_clss = torch.randn(num_new_cls, self.backbone_machine.dim_model)
        all_clss = torch.cat([self.cls_embeddings.data, new_clss])
        self.cls_embeddings = nn.Parameter(all_clss)
        self.num_cls = self.num_cls + num_new_cls
    
    def forward(self, x, cls_indices: list = None):
        """
        Args:
            x: tensor with shape [batch, len, input_dim]

        Returns:
            This is a description of what is returned.
        """    
        if cls_indices is None :
            cls_indices = list(range(self.num_cls))
        batch_size, len_inputs, _ = x.shape
        cls_batched  = self.cls_embeddings[cls_indices].unsqueeze(0).repeat(batch_size, 1, 1)
        pos_batched = self.pos_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        temp = x
        
        temp = self.token_embedding(temp)
        temp = temp + pos_batched[:, :len_inputs, :]
        temp = torch.cat((cls_batched, temp), dim=1)
        temp = self.backbone_machine(temp)
        return temp[:, :len(cls_indices), :]