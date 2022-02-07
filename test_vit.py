from vit_pytorch import ViT
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm.auto import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

batch_size = 512
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


model = ViT(
    image_size = 32,
    patch_size = 8,
    num_classes = 10,
    dim = 128,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)

print(count_parameters(model))
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
for epoch in range(200):  # loop over the dataset multiple times
    if epoch < 1:
        lr = 1e-4
    elif epoch < 5 :
        lr = 3e-5
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
