import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from network.layer import AutoEncoder
from network.loss import kullbackLeiblerLoss, crossentropyLoss
from utils.util import createIncrementalPath
import numpy as np
from utils.plot import saveImages, saveScatteredImage
torch.autograd.set_detect_anomaly(True)

resultPath = createIncrementalPath('./result')
imgSize = 28
batchSize = 64

''' In order to plot, the following value should be 2. '''
dimlatentVector = 2

encdimsLayer = [500, 500]
decdimsLayer = [500, 500]
learningRate = 1e-3
numEpoch = 100
plot = True
zSpaceForPlot = torch.Tensor(np.rollaxis(np.mgrid[2:-2:20 * 1j, 2:-2:20 * 1j], 0, 3).reshape([-1, 2]))

os.makedirs("../data/MNIST", exist_ok=True)
transform = transforms.Compose([transforms.Resize(imgSize), transforms.ToTensor()])
trainDataset = datasets.MNIST("./data/MNIST", train=True, download=True, transform=transform)
trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True)
testDataset = datasets.MNIST("./data/MNIST", train=False, download=True, transform=transform)
testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True, drop_last=True)

vae = AutoEncoder(imgSize, encdimsLayer, dimlatentVector, decdimsLayer)
optimizer = torch.optim.Adam(vae.parameters(), lr=learningRate)

for epoch in range(numEpoch):
    vae.train()
    for i, (imgs, _) in enumerate(trainLoader):
        imgs = imgs.view(batchSize, -1)
        mu, sigma, predictImgs = vae(imgs)
        reconstructionError = crossentropyLoss(predictImgs, imgs)
        regulizationError = kullbackLeiblerLoss(mu, sigma).mean()
        elbo = regulizationError + reconstructionError
        
        optimizer.zero_grad()
        elbo.backward()
        optimizer.step()
    
    print(f"epoch: {epoch + 1}, reconstruction loss: {reconstructionError}, regulization loss: {regulizationError}, evidence low bound: {elbo}")
    
    with torch.no_grad():
        if dimlatentVector == 2 and plot:
            plotImages = vae.decoder(zSpaceForPlot)
            saveImages(plotImages, imgSize, 20, resultPath, f"epoch_{epoch}.jpg")
            latentVectors = torch.Tensor()
            ids = torch.Tensor()
            for imgs, labels in testLoader:
                imgs = imgs.view(batchSize, -1)
                latentVectors = torch.cat([latentVectors, vae.encode(imgs)])
                ids = torch.cat([ids, labels])
            saveScatteredImage(latentVectors, ids, resultPath, f"epoch_{epoch}_scatteredImage.jpg")
                
        elif plot:
            imgs, labels = next(iter(testLoader))
            imgs = imgs.view(batchSize, -1)
            plotImages = vae(imgs).reshape((batchSize, -1, imgSize, imgSize))
            saveImages(plotImages, imgSize, 8, resultPath, f"epoch_{epoch}.jpg")