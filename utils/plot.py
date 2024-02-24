import matplotlib.pyplot as plt
import os
import torch
from torchvision.utils import save_image
from math import sqrt

def saveScatteredImage(z, id, savePath, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=id, marker='o', edgecolor='none', cmap='jet')
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-4, 4])
    axes.set_ylim([-4, 4])
    plt.grid(True)
    plt.savefig(os.path.join(savePath, name))
    plt.close()

def saveImages(images, imgSize, numImages, savePath, name):
    batchSize = images.shape[0]
    images = images.reshape((batchSize, -1, imgSize, imgSize))
    numImages = int(sqrt(numImages))
    saveImage = torch.zeros((1, images.shape[1], images.shape[2] * numImages, images.shape[3] * numImages))
    
    for idx, image in enumerate(images):
        i = int(idx % numImages)
        j = int(idx / numImages)
        saveImage[0, 0, j * imgSize:j * imgSize + imgSize, i * imgSize:i * imgSize + imgSize] = image
        
    save_image(fp=os.path.join(savePath, name), tensor=saveImage)
    