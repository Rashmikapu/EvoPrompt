#Seed 2

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

class Seed(nn.Module):
    features: int = 25
    # nlayer: int = 3
    def __init__(self):
          super(Seed, self).__init__()
          # Initialize model parameters here
          self.conv1 = nn.Conv2d(in_channels = 1, out_channels = self.features, kernel_size=(5,1), stride=(2,1), padding=(1,))
          self.relu1 = nn.ReLU()


          self.conv_layers = nn.ModuleList()
          for _ in range(2):
              self.conv_layers.append(nn.Conv2d(in_channels=self.features, out_channels = self.features, kernel_size=(3,1), stride =(2,1), padding = (1,)))
              self.conv_layers.append(nn.ReLU())

          self.flatten = nn.Flatten()
        
         # Calculate output size after convolutions dynamically
          self.conv_out_size = self._get_conv_out_size()
          self.fc1 = nn.Linear(in_features=self.conv_out_size[1], out_features=10)


    def _get_conv_out_size(self):
        """
        Calculates the output size after convolutions
        """
        dummy_tensor = torch.randn(1,1,40)
        dummy_tensor = dummy_tensor[..., None]
        output = self.conv1(dummy_tensor)
        for conv, relu in zip(self.conv_layers[::2], self.conv_layers[1::2]):
            output = conv(output)
        output = self.flatten(output)
        output_size = output.size() # Get output size
        return output_size

    def __call__(self, x):

        x = x.reshape(10, 1, 40)
        # data_reshaped = data.reshape(10, 1, 40, 1)
        x = x[..., None]  # Add a channel dimension for 2D convolution
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.pool1(x)

        for conv, relu in zip(self.conv_layers[::2], self.conv_layers[1::2]):
            x = conv(x)
            x = relu(x)

        x = self.flatten(x)
        # print(x.shape)
        x = self.fc1(x)

        return x


def main():

    with open("mnist1d_data.pkl", "rb") as f:
          data = pickle.load(f)

        # print(data)
    x = data['x']
    y = data['y']

    # Define the split ratio (e.g., 80% train, 20% validation)
    train_size = 0.8

    # Split the data and labels into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=1-train_size, random_state=42)

    # Create training and validation data in the desired format
 
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