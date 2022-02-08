import sys
sys.path.append('/home/mmsamiei/transformer-machines/')


from datasets import fuzzy_boolean_dataset
import torch.nn as nn
from models import simple_machine, cls_machine, dini_machine
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import r2_score
from utils import utils
import math

batch_size = 128

dataset = fuzzy_boolean_dataset.FuzzyBooleanDataset('data/fbd.npy')
train_len = int( 0.8 * len(dataset) ) 
val_len = len(dataset) - train_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, \
    shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, \
    shuffle=True, num_workers=2)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #self.backbone = simple_machine.SimpleEncoder(32, 64, 2, 2, 4, 8)
        self.backbone = dini_machine.DiniEncoder(128, 4, 2, 2, 4, 2, 2, 0.1, 1, 1, 0.1)
        self.cls_net = cls_machine.ClsMachine(self.backbone, 30, 1)
        self.reg_head = nn.Linear(128, 1)
    
    def forward(self, x):
        temp = x 
        temp = self.cls_net(temp)
        temp = self.reg_head(temp)
        return temp

device = torch.device('cuda')
model = Model().to(device)

print(utils.count_parameters(model))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


for epoch in range(20):
    running_loss = []
    pbar = tqdm(trainloader)
    for X, Y in pbar:
        X = X.to(device).unsqueeze(2).float()
        Y = Y.to(device).float()[:, :20]
        optimizer.zero_grad()
        pred = model(X).squeeze(2)[:, :20]
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch}")
        running_loss.append(math.sqrt(loss.item()))
        pbar.set_postfix(loss = sum(running_loss) / len(running_loss))
    
    with torch.no_grad():
        pbar = tqdm(valloader)
        scores = []
        for X, Y in pbar:
            X = X.to(device).unsqueeze(2).float().detach()
            Y = Y.to(device).float()[:, :20]
            pred = model(X).squeeze(2)[:, :20]
            score = r2_score(pred.detach().cpu(), Y.detach().cpu())
            scores.append(score)
            pbar.set_postfix(score = sum(scores) / len(scores))
