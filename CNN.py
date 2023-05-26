
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self, ).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.28),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.28),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64, 4 * 4 * 64),
            nn.ReLU(),
            nn.BatchNorm1d(4 * 4 * 64),
            nn.Linear(4 * 4 * 64, 4 * 4 * 64),
            nn.ReLU(),
            nn.BatchNorm1d(4 * 4 * 64),
            nn.Dropout(0.28),
        )
        self.fc2 = nn.Linear(4 * 4 * 64, 36)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
    
    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1 )
        out = self.fc1(out)
        out = torch.flatten(out, 1)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)