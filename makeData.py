import matplotlib.pyplot as plt
import numpy as np

# Create your own dataset
# Left click = Red
# Right click = Blue

datasetName = input("Enter the name of the dataset: ")

plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title("Click to create data points")
plt.xlabel("X")
plt.ylabel("Y")

# Create a list to store the data
blueX = np.array([])
blueY = np.array([])
redX = np.array([])
redY = np.array([])

# Function to handle mouse clicks
def onclick(event):
    global blueX, blueY, redX, redY
    if event.button == 1:
        redX = np.append(redX, event.xdata)
        redY = np.append(redY, event.ydata)
        plt.plot(redX, redY, 'ro')
    elif event.button == 3:
        blueX = np.append(blueX, event.xdata)
        blueY = np.append(blueY, event.ydata)
        plt.plot(blueX, blueY, 'bo')
    plt.show()

# Connect the function to the mouse click event
cid = plt.connect('button_press_event', onclick)

plt.show()

print("blueX", blueX)
print("blueY", blueY)
print("redX", redX)
print("redY", redY)

# Save the data to 1 file
np.savez(f"datasets/{datasetName}", blueX=blueX, blueY=blueY, redX=redX, redY=redY)