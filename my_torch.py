import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Define a simple feedforward neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input: 28x28 images
        self.fc2 = nn.Linear(128, 10)  # Output: 10 classes (digits 0-9)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Initialize model, loss, and optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (one epoch for brevity)
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

print("Training complete! (This is a simplified example.)")

class_names = train_data.classes
rows, cols = 4, 4
fig = plt.figure(figsize=(7, 7))

for i in range(1, rows * cols + 1):
  img, label = train_data[i]
  fig.add_subplot(rows, cols, i)
  plt.imshow(img.squeeze(), cmap="jet")
  plt.title(class_names[label])
  plt.axis(False)

plt.show()
