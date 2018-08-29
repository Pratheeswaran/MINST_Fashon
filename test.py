import torch
from  torch.utils.data import Dataset,DataLoader
import pandas as pd
from  model import Net

net = Net()
net.eval() 
MODEL_STORE_PATH = 'path to stored weights'
net.load_state_dict(torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt'))

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
        b = torch.from_numpy(image).type(torch.FloatTensor)
        target = torch.zeros(10)
        target[self.images.iloc[idx][0]]=1.
        sample = [b,target]
        return sample

test_dataset =MINSTFashon('\\path to \\fashion-mnist_test.csv')
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(labels.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))