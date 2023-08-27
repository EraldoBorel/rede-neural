import numpy as np
import pathlib


def get_mnist():
    path_data = f"{pathlib.Path(__file__).parent.absolute()}\..\data\mnist.npz"
    with np.load(path_data) as f:
        images, labels = f["x_train"], f["y_train"]

    #convert pixel range 0:255 to floag values... 
    # Ex: pixel = 255 -> pixel/255 -> pixel = 1.0
    # Ex: pixel = 18 -> pixel/255 -> pixel = 0.07058824
    images = images.astype("float32") / 255


    # images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    
    
    labels = np.eye(10)[labels]
    
    return images, labels