# Basic imports
import glob
import os
from timeit import default_timer as timer

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Dataset class and neural network class
from custom_dataset import MyDataset

import model
import matplotlib.pyplot as plt


# Learning parameters
batch_size = 16
lr = 1e-3
max_epoch = 20
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataset variables
data_path = "/home/mehmet/Desktop/SirenDetection/3class"

# Output files
netname = "net"

# Initialize the dataset and dataloader
traindataset = MyDataset(data_path = data_path, train = True)
trainloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)

testdataset = MyDataset(data_path = data_path, train = False)
testloader = DataLoader(testdataset, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 0)

# Initialize the NN model
net = model.Net()
net = net.to(device)

# Optimizer and Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = lr)

start = timer()  # start the timer
# Training
for epoch in range(max_epoch + 1):
    batchiter = 0

    for batch in trainloader:
        
        batchiter += 1
        spec = batch[0].unsqueeze(1).to(device) 
        label = batch[1].to(device)
        y_pred = net(spec)   
        optimizer.zero_grad()    
        loss = criterion(y_pred, label)
        loss.backward()
        optimizer.step()
            
        print("TRAIN","Epoch:",epoch+1, "Data-Num:",batchiter, "Loss:",loss.item(), " label: ", label.tolist())

    if epoch % 1 == 0:
        torch.save(net.state_dict(), "./saved_models_3class/" + netname + "_epoch_%d"%(epoch) + ".pth")
        

end = timer()  # end the timer
elapsed_time = (end - start)/60  # elapsed time is calculated

print('Elapsed time for training: {:.3f} minutes!'.format(float(elapsed_time)))

# Test part
directory = "./saved_models_3class"
files = list(filter(os.path.isfile, glob.glob(directory + "/*.pth")))
files.sort(key=lambda x: os.path.getmtime(x))
acc_list = []
epoch_list = []
for filename in files:
    net.load_state_dict(torch.load(filename))
    correct = 0
    total = 0
    print(filename)
    with torch.no_grad():
        for data in testloader:
            spec = data[0].unsqueeze(1).to(device)
            label = data[1].to(device)
            output = net(spec)

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    acc_list.append(100 * (correct/total))
    epoch_list.append(int(filename.split("_")[-1].split(".")[0]))

fig = plt.figure()
plt.plot(epoch_list, acc_list)
plt.title("Elapsed Time ({:.3f} minutes), Test Accuracy ({:.3f} %)".format(float(elapsed_time), float(acc_list[-1])))
plt.grid()
plt.xticks(range(0, 21, 2))
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
fig.savefig("Test3class.jpg")
plt.show()
