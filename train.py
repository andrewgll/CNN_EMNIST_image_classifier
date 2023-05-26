from CNN import Net
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn as nn
from PIL import Image
import numpy as np

batch_size = 64
lr = 0.0001
num_warmup_iters = 500
epochs = 30
weight_decay = 0.0001
wandb_log = False #if need to save each run stats  
wandb_project = ""
wandb_run_name = ""
resume = False
compile = True #Gives better results but need more time for startup. Set to True is recommended
out_file = "model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPU much faster

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


    

# I used training set where images are transposed so need to transform them back
emnist_transform = transforms.Compose([
    transforms.Lambda(lambda x:Image.fromarray(np.transpose(x))),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

letter_target__transform = transforms.Compose([    
    transforms.Lambda(lambda y: y + 9)
])
#default dataset from pytorch
emnist_trainset = datasets.EMNIST(root='./data', split='digits', train=True, download=True, transform=emnist_transform)
emnist_testset =  datasets.EMNIST(root='./data', split='digits', train=False, download=True, transform=emnist_transform)

emnist_letters_trainset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=emnist_transform, target_transform=letter_target__transform)
emnist_letters_testset = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=emnist_transform, target_transform=letter_target__transform)

trainset = torch.utils.data.ConcatDataset([emnist_trainset, emnist_letters_trainset])
testset = torch.utils.data.ConcatDataset([emnist_testset, emnist_letters_testset])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=True)

model = Net().to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
    
test_loss = 0

#Warmup stage
warmup_lr = 0.00001
warmup_optimizer = optim.AdamW(model.parameters(), lr=warmup_lr, weight_decay=weight_decay)
for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        warmup_optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        warmup_optimizer.step()
        if batch_idx > num_warmup_iters:
            break
            
        if batch_idx % 100 == 0:
            print('Warmup: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
for epoch in range(1, epochs + 1):
    # Train the model
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        if wandb_log:
            wandb.log({
                "Epoch": epoch,
                "train/loss": loss.item()
            })
    # Test the model
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target,reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    mean_loss = test_loss / len(test_loader.dataset)
    if wandb_log:
        wandb.log({
            "val/loss": mean_loss
        })
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        mean_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_loss = 0   

# Save the trained weights
torch.save(model.state_dict(), 'model.pt')