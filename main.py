from matplotlib import animation
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

import loadData

BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 4000
EPOCH_DIVISOR = 100
DATASET_NAME = "clusters"
LOSS_FUNCTION = nn.MSELoss()
CMAP = plt.cm.get_cmap("coolwarm")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 inputs: x, y
        # 2 outputs: red, blue
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 12)
        self.layer3 = nn.Linear(12, 12)
        self.layer4 = nn.Linear(12, 8)
        self.layer5 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x)) + 0.01 * self.layer4(x)
        x = self.layer5(x)
        return x


class Data(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Load data
data, labels = loadData.loadFrom(DATASET_NAME)

# Split data into train and test
index = int(0.8 * len(data))
trainData, testData = np.split(data, [index])
trainLabels, testLabels = np.split(labels, [index])

minX = min(trainData[:, 0])
maxX = max(trainData[:, 0])
minY = min(trainData[:, 1])
maxY = max(trainData[:, 1])

# Create the dataset
trainDataset = Data(trainData, trainLabels)
testDataset = Data(testData, testLabels)

# Create the dataloader
trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)

# Create the model
model = Model()


def getDecisionBoundaryData(minX, maxX, minY, maxY):
    # Plot the decision boundary
    x = np.linspace(minX, maxX, 300)
    y = np.linspace(minY, maxY, 300)
    # Meshgrid will create a grid of points from the x and y arrays
    xx, yy = np.meshgrid(x, y)

    # Create the data
    data = np.array([xx.ravel(), yy.ravel()]).T
    data = torch.from_numpy(data).float()
    
    # Get the predictions
    predictions = model(data)
    predictions = torch.argmin(predictions, dim=1)
    predictions = predictions.numpy()

    return xx, yy, predictions.reshape(xx.shape)


def plotDecisionBoundary(xx, yy, predictions):
    return dataAx.contourf(xx, yy, predictions, cmap=CMAP, alpha=0.4)


# Create the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Train the model
lossHistory = []
decisionBoundaryHistory = []

for epoch in range(EPOCHS):
    for data, labels in trainLoader:
        # Forward pass
        predictions = model(data.float())
        loss = LOSS_FUNCTION(predictions, labels.float())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the loss
    currentLoss = loss.item()
    lossHistory.append(currentLoss)

    numEpochs = " " * (len(str(EPOCHS)) - len(str(epoch + 1))) + str(epoch + 1)
    
    if (epoch + 1) % EPOCH_DIVISOR == 0 :
        print(f"Epoch {numEpochs}/{EPOCHS} - Loss: {currentLoss}")
        # Save the decision boundary
        xx, yy, predictions = getDecisionBoundaryData(minX, maxX, minY, maxY)
        decisionBoundaryHistory.append((xx, yy, predictions))


# Get accuracy
def getAccuracy(loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in loader:
            predictions = model(data.float())
            predictions = torch.argmax(predictions, dim=1)
            labels = torch.argmax(labels, dim=1)

            correct += (predictions == labels).sum().item()
            total += len(data)

    return correct / total

# Get loss and accuracy
trainLoss = LOSS_FUNCTION(model(torch.from_numpy(trainData).float()), torch.from_numpy(trainLabels).float()).item()
testLoss = LOSS_FUNCTION(model(torch.from_numpy(testData).float()), torch.from_numpy(testLabels).float()).item()
trainAccuracy = getAccuracy(trainLoader)
testAccuracy = getAccuracy(testLoader)

print(f"Train loss: {trainLoss} - Train accuracy: {trainAccuracy}")
print(f"Test loss: {testLoss} - Test accuracy: {testAccuracy}")

# Plot data and loss history
fig, (dataAx, lossAx) = plt.subplots(1, 2, figsize=(10, 5))

# Plot data
xx, yy, predictions = getDecisionBoundaryData(minX, maxX, minY, maxY)
boundary = plotDecisionBoundary(xx, yy, predictions)

def animateBoundary(i):
    xx, yy, predictions = decisionBoundaryHistory[i]

    # Remove the old boundary
    dataAx.cla()
    dataAx.scatter(trainData[:, 0], trainData[:, 1], c=trainLabels[:, 0], cmap=CMAP, label="Training data", s=20)
    dataAx.scatter(testData[:, 0], testData[:, 1], c=testLabels[:, 0], cmap=CMAP, label="Testing data", marker="x")

    dataAx.set_xlim(minX, maxX)
    dataAx.set_ylim(minY, maxY)

    return plotDecisionBoundary(xx, yy, predictions)

animBoundary = animation.FuncAnimation(fig, animateBoundary, frames=len(decisionBoundaryHistory), interval=1)

dataAx.set_title("Data")
dataAx.set_xlabel("x")
dataAx.set_ylabel("y")
dataAx.scatter(trainData[:, 0], trainData[:, 1], c=trainLabels[:, 0], cmap=CMAP, label="Training data", s=20)
dataAx.scatter(testData[:, 0], testData[:, 1], c=testLabels[:, 0], cmap=CMAP, label="Testing data", marker="x")

dataAx.set_xlim(minX, maxX)
dataAx.set_ylim(minY, maxY)
dataAx.legend()

# Plot loss history
lossAx.set_title("Loss History")
lossAx.set_xlabel("Epoch")
lossAx.set_ylabel("Loss")

lossLine = lossAx.plot(lossHistory, color="red")[0]

def animateLoss(i):
    i *= EPOCH_DIVISOR
    lossLine.set_data(range(i + 1), lossHistory[:i + 1])
    lossAx.set_xlim(0, i + 5)

    # for the last 500 epochs, set the y limits
    if i > len(lossHistory) - 500:
        lossAx.set_ylim(min(lossHistory[i - 500:i + 1]), max(lossHistory[i - 500:i + 1]))

    return lossLine,

animCost = animation.FuncAnimation(fig, animateLoss, frames=int(len(lossHistory) / EPOCH_DIVISOR), interval=1)

plt.suptitle(
    f"Dataset: {DATASET_NAME} | Epochs: {EPOCHS} | Batch size: {BATCH_SIZE} | Learning rate: {LEARNING_RATE} | Loss function: {LOSS_FUNCTION}"
)

# subtitle
plt.figtext(0.5, 0.01, f"Train loss: {trainLoss} - Train accuracy: {trainAccuracy * 100:.2f}% | Test loss: {testLoss} - Test accuracy: {testAccuracy * 100:.2f}%", ha="center", fontsize=8)

plt.tight_layout()
plt.show()
