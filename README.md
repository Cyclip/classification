# Classification
![](https://github.com/Cyclip/classification/raw/main/repo/fitting.gif)

A simple PyTorch neural network model used to fit onto a dataset. It consists of 2 colours (red and blue) all placed in various places with an obvious dividing line (to a human, atleast). These are contained in multiple datasets in `./datasets` stored as `numpy` files.  

## Prerequisites
You will need Python, along with these modules:
- Matplotlib
- PyTorch
- Numpy
Install them via `python -m pip install matplotlib pytorch numpy`.

## Making your own dataset
The `makeData.py` module can be ran directly. Upon running, it will ask for the name of the dataset you are creating. After entering, left clicking will create a **red** dot at the given position, whereas right clicking will create a **blue** dot. Once you are done, close the **matplotlib window** -- not the python program itself -- and the dataset should appear in `./datasets/your_dataset_name.npz`.

## Training your model
### Constants
There are some constants which you can adjust:
- `BATCH_SIZE` (default 128): The number of samples to be used in each batch.
- `LEARNING_RATE` (default 0.01): The learning rate of the model.
- `EPOCHS` (default 4000): The number of times the model will be trained on the entire dataset.
- `EPOCH_DIVISOR` (default 100): The number of epochs to wait before printing the loss.
- `DATASET_NAME` (default "clusters"): The name of the dataset to be used. You may make your own.
- `LOSS_FUNCTION` (default `nn.MSELoss()`): The loss function to be used.
- `CMAP` (default `pl.cm.get_cmap("coolwarm")`): The colour map to be used for the plot when displaying decision boundaries

### Training
To train the model after adjusting the constants, run `python main.py`. The model will be trained and the loss will be printed every `EPOCH_DIVISOR` epochs. Once the model is trained, the decision boundary (along with loss history) will be displayed in an animation showing the models progress.