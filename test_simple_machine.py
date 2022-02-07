from vit_pytorch import ViT
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from models import simple_machine
from tqdm.auto import tqdm
from torch.optim import lr_scheduler

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class SiT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, dim_hid,\
                 num_layer, num_iter, num_funcs, num_heads,  device, pool='cls', channels = 3, dropout = 0., emb_dropout=0.):
        super(SiT, self).__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = simple_machine.SimpleEncoder(dim, dim_hid, num_layer, num_iter, num_funcs, num_heads, dropout, device=device)
        # self.transformer = Transformer(dim,  num_layer, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

batch_size = 128
device = torch.device('cuda')

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

model = SiT(image_size=(32,32), patch_size=(8,8), num_classes=10, dim=256, dim_hid=512,\
                 num_layer=6, num_iter=2, num_funcs=4, num_heads=8,  device=device).to(device)


print(count_parameters(model))
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
for epoch in range(200):  # loop over the dataset multiple times
    if epoch < 1:
        lr = 5e-4
    elif epoch < 5 :
        lr = 4e-5
    elif epoch < 25:
        lr = 2e-5
    else:
        lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    # if epoch < 1:
    #     for g in optimizer.param_groups:
    #         g['lr'] = 5e-4
    # elif epoch < 2:
    #     for g in optimizer.param_groups:
    #         g['lr'] = 1e-5
    # elif epoch < 10:
    #     for g in optimizer.param_groups:
    #         g['lr'] = 1e-5
    # else:
    #     for g in optimizer.param_groups:
    #         g['lr'] = 1e-6
    
    running_loss = []
    pbar = tqdm(trainloader)
    for data in pbar:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        pbar.set_description(f"Epoch {epoch}")
        running_loss.append(loss.item())
        pbar.set_postfix(loss = sum(running_loss) / len(running_loss))
        optimizer.step()
    correct = 0
    total = 0
    print(f"Loss: ", sum(running_loss) / len(running_loss))
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


print('Finished Training')
