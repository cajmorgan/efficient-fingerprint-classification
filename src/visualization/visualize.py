import matplotlib.pyplot as plt
import torchvision
import numpy as np

def plot_batch(images):
    img_grid = torchvision.utils.make_grid(images)
    npimg = img_grid.numpy()

    plt.figure(figsize=(12,4))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
