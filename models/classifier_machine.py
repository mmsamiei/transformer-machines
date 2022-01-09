import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class ClassifierMachine(nn.Module):

    def __init__(self, backbone_machine, num_cls, input_dim, max_len=128, device=None):
        factory_kwargs = {'device': device}
        super(ClassifierMachine, self).__init__()
        self.backbone_machine = backbone_machine
        self.num_cls = num_cls
        self.max_len = max_len
        self.cls_embeddings = \
            nn.Parameter(torch.randn(self.num_cls,self.backbone_machine.dim_model))
        
        self.token_embedding = nn.Linear(input_dim, self.backbone_machine.dim_model)

        self.pos_embeddings = \
            nn.Parameter(torch.randn(self.max_len,self.backbone_machine.dim_model))
    
    def forward(self, x, key_padding_mask=None, attn_mask=None, device=None):
        """
        Args:
            x: tensor with shape [batch, len, input_dim]
            param2: This is a second param.

        Returns:
            This is a description of what is returned.
        """    
        batch_size, len_inputs, _ = x.shape
        cls_batched  = self.cls_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        pos_batched = self.pos_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        temp = x
        temp = self.token_embedding(temp)
        print(temp.shape)
        temp = temp + pos_batched[:, :len_inputs, :]
        temp = torch.cat((cls_batched, temp), dim=1)
        temp = self.backbone_machine(temp)
        return temp[:, :self.num_cls, :]