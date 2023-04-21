import torch.nn as nn
import torch
from torchvision import datasets, transforms

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

# Define data augmentation and normalization transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the MNIST test dataset
testset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Load the saved model state dictionary into a new instance of the model
model = Net()
model.load_state_dict(torch.load('./mnist_model.pth'))

# Set the model to evaluation mode (turns off dropout)
model.eval()

# Iterate over the test set and make predictions with the trained model
correct = 0
total = 0
with torch.no_grad():
    count = 0
    for data in testloader:
        if count > 1:
            break
        count += 1
        # Get the inputs; data is a list of [inputs, labels]
        images, labels = data
        print(labels)

        # Forward pass through the network to get predicted outputs
        outputs = model(images)

        # Choose the class with the highest output (probability)
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)

        # Keep track of the number of correctly classified images
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(correct, total)

# Print the accuracy on the test set
print('Accuracy on test set: %d %%' % (100 * correct / total))