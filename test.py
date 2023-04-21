import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image

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
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the saved model state dictionary into a new instance of the model
model = Net()
model.load_state_dict(torch.load('./mnist_model.pth'))

# Set the model to evaluation mode (turns off dropout)
model.eval()

# Load the image and apply the same transformations used during training
for i in range(10):
    name = 'test/image_000' + str(i) + '.png'
    image = Image.open(name)
    input_tensor = transform(image).unsqueeze(0)  # unsqueeze to add batch dimension

    # Make a prediction with the trained model
    try:
        with torch.no_grad():
            output = model(input_tensor)
    except:
        print("error")


    # Choose the class with the highest output (probability)
    _, predicted = torch.max(output.data, 1)
    print(predicted)

    # Print the predicted label
    print('Predicted digit:', predicted.item())