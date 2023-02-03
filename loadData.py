import numpy as np

def loadFrom(filename):
    # Get blueX, blueY, redX, redY
    data = np.load(f"datasets/{filename}.npz")
    blueX = data["blueX"]
    blueY = data["blueY"]
    redX = data["redX"]
    redY = data["redY"]

    # Dataset will be 2 arrays:
    # Array 1: (x, y)
    # Array 2: [red, blue]

    # Create the dataset
    blue = np.array([blueX, blueY]).T
    red = np.array([redX, redY]).T

    # Create the labels
    blueLabels = np.array([[0, 1] for _ in range(len(blue))])
    redLabels = np.array([[1, 0] for _ in range(len(red))])

    # Combine the data and labels
    points = np.concatenate((blue, red))
    labels = np.concatenate((blueLabels, redLabels))

    # Shuffle the data
    indices = np.arange(points.shape[0])
    np.random.shuffle(indices)

    points = points[indices]
    labels = labels[indices]

    return points, labels