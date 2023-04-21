import time

import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import datasets, transforms

# Define data augmentation and normalization transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the MNIST training dataset
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # The input layer has 784 neurons (28x28)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Flatten the input tensor from (batch_size, 1, 28, 28) to (batch_size, 784)
        x = x.view(-1, 784)

        # Pass the tensor through the first fully connected layer
        x = torch.relu(self.fc1(x))

        # Pass the tensor through the second fully connected layer
        x = self.fc2(x)

        return x

start_time = time.time()
# Create an instance of the model
model = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(10):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # Print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

end_time = time.time()
print('Finished Training', end_time - start_time)
print('Finished Training')

torch.save(model.state_dict(), './mnist_model.pth')
print('Model saved as mnist_model.pth')