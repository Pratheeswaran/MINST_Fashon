import torch
from  model import Net
import pandas as pd
from  torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torch.nn as nn

# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.01

net = Net().cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

class MINSTFashon(Dataset):
    """Fashon MINST dataset."""
    def __init__(self, csv_file):
        """
        Args:csv_file (string): Path to the csv file with dats.
        """
        self.images = pd.read_csv(csv_file)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image=self.images.iloc[idx][1:].as_matrix().reshape(1,28,28)
        b = torch.from_numpy(image).type(torch.FloatTensor).cuda()
        target = torch.zeros(num_classes).cuda() 
        target[self.images.iloc[idx][0]]=1.
        sample = [ b,  target]
        return sample

train_dataset =MINSTFashon('\\path to\\fashion-mnist_train.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
total_step = len(train_dataset)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad() 
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step() 
        total = labels.size(0)
        _, predicted = torch.max(output.data, 1)
        _, labels = torch.max(labels.data, 1)
        correct = (predicted == labels).sum().item()
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, (i+1)*batch_size, total_step, loss.item(),
                          (correct / total) * 100))

torch.save(net.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')
