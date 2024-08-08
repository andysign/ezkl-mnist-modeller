import platform
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Convolutional encoder
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel

        # Fully connected layers / Dense block
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)         # 120 inputs, 84 outputs
        self.fc3 = nn.Linear(84, 10)          # 84 inputs, 10 outputs (number of classes)

    def forward(self, x):
        # Convolutional block
        x = F.avg_pool2d(F.sigmoid(self.conv1(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool
        x = F.avg_pool2d(F.sigmoid(self.conv2(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool

        # Flattening
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)  # No activation function here, will use CrossEntropyLoss later
        return x


import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import Adam  # Import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def normalize_img(image, label):
  return torch.round(image), label

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256

is_needs_downloading = True
if (os.path.exists('mnist-train-n-test.tar.gz')):
    print('Found local zip. Unzipping MNIST train and test...')
    os.system('tar -xzf mnist-train-n-test.tar.gz')
    is_needs_downloading = False

train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=is_needs_downloading)
test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download=is_needs_downloading)

train_dataset_all = train_dataset # torch.utils.data.Subset(train_dataset, list(range(0, len(train_dataset), 1)))
train_dataset_subset = torch.utils.data.Subset(train_dataset, list(range(1, len(train_dataset), 2)))
test_dataset_all = test_dataset # torch.utils.data.Subset(train_dataset, list(range(0, len(train_dataset), 1)))
test_dataset_subset = torch.utils.data.Subset(train_dataset, list(range(1, len(train_dataset), 2)))

train_loader = DataLoader(train_dataset_all, batch_size=batch_size) # train_dataset_subset
test_loader = DataLoader(test_dataset_all, batch_size=batch_size) # test_dataset_subset
model = LeNet().to(device)
adam = Adam(model.parameters())  # Using Adam with a learning rate of 1e-3
loss_fn = CrossEntropyLoss()
all_epoch = 25
prev_acc = 0

if (os.path.exists('models-pkl.tar.gz')):
    print('Found local zip. Unzipping MNIST model(s)...')
    os.system('tar -xzf models-pkl.tar.gz')
    all_epoch = 0

for current_epoch in range(all_epoch):
    model.train()
    for idx, (train_x, train_label) in enumerate(train_loader):
        train_x = train_x.to(device)
        # normalize the image to 0 or 1 to reflect the inputs from the drawing board
        train_x = train_x.round()
        train_label = train_label.to(device)
        adam.zero_grad()  # Use adam optimizer
        predict_y = model(train_x.float())
        loss = loss_fn(predict_y, train_label.long())
        loss.backward()
        adam.step()  # Use adam optimizer
    all_correct_num = 0
    all_sample_num = 0
    model.eval()

    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x = test_x.to(device)
         # normalize the image to 0 or 1 to reflect the inputs from the drawing board
        test_x = test_x.round()
        test_label = test_label.to(device)
        predict_y = model(test_x.float()).detach()
        predict_y = torch.argmax(predict_y, dim=-1)
        current_correct_num = predict_y == test_label
        all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
        all_sample_num += current_correct_num.shape[0]
    acc = all_correct_num / all_sample_num
    print('test accuracy: {:.3f}'.format(acc), flush=True)
    if not os.path.isdir("models_good"):
        os.mkdir("models_good")
    torch.save(model, 'models_good/mnist_{:.3f}.pkl'.format(acc))
    prev_acc = acc

if (os.path.exists('models-pkl.tar.gz')):
    print('Loading MNIST model from disk...')
    model = torch.load(os.path.join('models_good', os.listdir('./models_good/')[-1]))


import os

model_path = os.path.join('network_good.onnx')
data_path = os.path.join('input.json')


import torch
import json

model.eval()  # Set the model to evaluation mode

# # Fetch a single data point from the train_dataset
# # Ensure train_dataset is already loaded and accessible
train_data_point, _ = next(iter(train_dataset))
train_data_point = train_data_point.unsqueeze(0)  # Add a batch dimension

# Verify the device (CPU or CUDA) and transfer the data point to the same device as the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data_point = train_data_point.to(device)

# # Export the model to ONNX format
torch.onnx.export(model, train_data_point, model_path, export_params=True, opset_version=12, do_constant_folding=True, input_names=['input_0'], output_names=['output'])

# Convert the tensor to numpy array and reshape it for JSON serialization
x = train_data_point.cpu().detach().numpy().reshape([-1]).tolist()
data = {'input_data': [x]}
with open('input.json', 'w') as f:
    json.dump(data, f)

print(f"Model exported to {model_path} and input data saved to input.json")


# Capture set of data points
num_data_points = 15

# Fetch 30 data points from the train_dataset
data_points = []
for i, (data_point, _) in enumerate(train_dataset):
    if i >= num_data_points:
        break
    data_points.append(data_point)

# Stack the data points to create a batch
train_data_batch = torch.stack(data_points)

# Add a batch dimension if not already present
if train_data_batch.dim() == 3:
    train_data_batch = train_data_batch.unsqueeze(0)

x = train_data_batch.cpu().detach().numpy().reshape([-1]).tolist()

data = dict(input_data = [x])

cal_path = os.path.join('cal_data.json')

# Serialize data into file:
json.dump( data, open(cal_path, 'w' ))


# Train the bad model (network_bad)
import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import Adam  # Import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def normalize_img(image, label):
  return torch.round(image), label

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256

is_needs_downloading = True
if (os.path.exists('mnist-train-n-test.tar.gz')):
    print('Found local zip. Unzipping MNIST train and test...')
    os.system('tar -xzf mnist-train-n-test.tar.gz')
    is_needs_downloading = False

train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=is_needs_downloading)
test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download=is_needs_downloading)

train_dataset_all = train_dataset # torch.utils.data.Subset(train_dataset, list(range(0, len(train_dataset), 1)))
train_dataset_subset = torch.utils.data.Subset(train_dataset, list(range(1, len(train_dataset), 2)))
test_dataset_all = test_dataset # torch.utils.data.Subset(train_dataset, list(range(0, len(train_dataset), 1)))
test_dataset_subset = torch.utils.data.Subset(train_dataset, list(range(1, len(train_dataset), 2)))

train_loader = DataLoader(train_dataset_all, batch_size=batch_size) # train_dataset_subset
test_loader = DataLoader(test_dataset_all, batch_size=batch_size) # test_dataset_subset
model = LeNet().to(device)
adam = Adam(model.parameters())  # Using Adam with a learning rate of 1e-3
loss_fn = CrossEntropyLoss()
all_epoch = 1 # Only one round of training for the bad model
prev_acc = 0

if (os.path.exists('models-pkl.tar.gz')):
    os.system('tar -xzf models-pkl.tar.gz')
    print('Found local zip. Unzipping MNIST model(s)...')
    all_epoch = 0

for current_epoch in range(all_epoch):
    model.train()
    for idx, (train_x, train_label) in enumerate(train_loader):
        train_x = train_x.to(device)
        # normalize the image to 0 or 1 to reflect the inputs from the drawing board
        train_x = train_x.round()
        train_label = train_label.to(device)
        adam.zero_grad()  # Use adam optimizer
        predict_y = model(train_x.float())
        loss = loss_fn(predict_y, train_label.long())
        loss.backward()
        adam.step()  # Use adam optimizer
    all_correct_num = 0
    all_sample_num = 0
    model.eval()

    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x = test_x.to(device)
         # normalize the image to 0 or 1 to reflect the inputs from the drawing board
        test_x = test_x.round()
        test_label = test_label.to(device)
        predict_y = model(test_x.float()).detach()
        predict_y = torch.argmax(predict_y, dim=-1)
        current_correct_num = predict_y == test_label
        all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
        all_sample_num += current_correct_num.shape[0]
    acc = all_correct_num / all_sample_num
    print('test accuracy: {:.3f}'.format(acc), flush=True)
    if not os.path.isdir("models_bad"):
        os.mkdir("models_bad")
    torch.save(model, 'models_bad/mnist_{:.3f}.pkl'.format(acc))
    prev_acc = acc

if (os.path.exists('models-pkl.tar.gz')):
    print('Loading MNIST model from disk...')
    model = torch.load(os.path.join('models_bad', os.listdir('./models_bad/')[-1]))


import os

model_path = os.path.join('network_bad.onnx')


import torch
import json

model.eval()  # Set the model to evaluation mode

# # Fetch a single data point from the train_dataset
# # Ensure train_dataset is already loaded and accessible
train_data_point, _ = next(iter(train_dataset))
train_data_point = train_data_point.unsqueeze(0)  # Add a batch dimension
train_data_point = train_data_point.to(device)

# # Export the model to ONNX format
torch.onnx.export(model, train_data_point, model_path, export_params=True, opset_version=12, do_constant_folding=True, input_names=['input_0'], output_names=['output'])

print(f"Model exported to {model_path}")