#Seed 3

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

class Seed(nn.Module):
    # features: int = 40
    hidden_size: int = 100
    def __init__(self):
          super(Seed, self).__init__()
          # Initialize model parameters here
          self.flatten = nn.Flatten()
          self.fc1 = nn.Linear(in_features=40, out_features=self.hidden_size)
          self.relu1 = nn.ReLU()
          self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
          # self.pool1 = nn.AvgPool2d(kernel_size=(2,1), padding=(1,0))
          self.fc3 = nn.Linear(in_features=self.hidden_size, out_features=10)
          
         
         


    def __call__(self, x):

        x = x.reshape(10, 1, 40) 
        # data_reshaped = data.reshape(10, 1, 40, 1)
        # x = x[..., None]  # Add a channel dimension for 2D convolution

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = x + self.relu1(self.fc2(x))
        x = self.fc3(x)

        return x


def main():

    with open("mnist1d_data.pkl", "rb") as f:
          data = pickle.load(f)

        # print(data)
    x = data['x']
    y = data['y']

    # print(np.array(x).shape)
    
    # Define the split ratio (e.g., 80% train, 20% validation)
    train_size = 0.8

    # Split the data and labels into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=1-train_size, random_state=42)

    train_dataset = MyDataset(x_train.astype(np.float32), y_train.astype(np.float32))

    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    
    val_dataset = MyDataset(x_val.astype(np.float32), y_val.astype(np.float32))
    test_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)
   
    # Create model instance
    seed_model = Seed()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(seed_model.parameters())

    # Train the model
    for epoch in range(10):  # Train for 10 epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            
            optimizer.zero_grad()
            output = seed_model(data)
            target = target.long()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Evaluate on test set after each epoch
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = seed_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%")

    # Calculate model size
    num_params = sum(p.numel() for p in seed_model.parameters())

    return accuracy, num_params


class MyDataset(Dataset):
    def __init__(self, x, y):
        
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == "__main__":
    accuracy, model_size = main()
    print(f"Final Accuracy: {accuracy:.2f}%, Model Size: {model_size}")