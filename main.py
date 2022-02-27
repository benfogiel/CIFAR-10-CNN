import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import csv

# checking if GPU is available or not
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# torchvision has CIFAR10 dataset built in
train_ds = datasets.CIFAR10("./data", train=True, download=True)
valid_ds = datasets.CIFAR10("./data", train=False)

class TestData(datasets.VisionDataset):
    
    filepath = "./test.npy"
    
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.data = np.load(self.filepath)
    
    def __getitem__(self, index: int):
        
        img = self.data[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img
        
    def __len__(self):
        return len(self.data)

test_ds = TestData("./data")

# first transform the images to tensor format, 
# then normalize the pixel values
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_ds.transform = transform
valid_ds.transform = transform
test_ds.transform = transform

train_ds[0][0].shape

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=64, shuffle=True
)
valid_loader = torch.utils.data.DataLoader(
    valid_ds, batch_size=1000
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=1000
)

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

convnet = ConvNet().to(device)
print(convnet)
    
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(convnet.parameters(), lr=0.01, momentum=0.9)

def train(epoch, model, trainloader, criterion, optimizer):
    model.train()
    running_loss, total, correct = 0.0, 0, 0
    for i, data in tqdm(enumerate(trainloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # compare predictions to ground truth
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # gather statistics
        running_loss += loss.item()
        
    running_loss /= len(trainloader)
    
    print('Training | Epoch: {}| Loss: {:.3f} | Accuracy on 50000 train images: {:.1f}'.format \
          (epoch+1, running_loss, 100 * correct / total))

def validate(epoch, model, valloader, criterion):
    model.eval()
    running_loss, total, correct = 0.0, 0, 0
    for i, data in tqdm(enumerate(valloader, 0)):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # compare predictions to ground truth
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # gather statistics
        running_loss += loss.item()
        
    running_loss /= len(valloader)
    
    print('Validation | Epoch: {}| Loss: {:.3f} | Accuracy on 10000 val images: {:.1f}'.format \
          (epoch+1, running_loss, 100 * correct / total))

for epoch in range(50):
    train(epoch, convnet, train_loader, criterion, optimizer)

# validating
validate(epoch, convnet, valid_loader, criterion)

def predict(model,testloader):
    
    model.eval()
    preds = []
    with torch.no_grad():
        # labels are not available for the actual test set
        for feature in tqdm(testloader):
            # calculate outputs by running images through the network
            outputs = model(feature.to(device))
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.tolist())
    
    return preds

predictions = predict(convnet, test_loader)
with open("new_weights.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "label"])
    for i, label in enumerate(predictions):
        writer.writerow([i, label])