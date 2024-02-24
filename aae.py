import os
import itertools

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, MSELoss
import numpy as np

from network.layer import AdversarialAutoEncoder
from utils.util import createIncrementalPath
from utils.plot import saveImages, saveScatteredImage
from utils.distribution import GaussianDistributor, GaussianMixtureDistributor, SwissRollDistributor
torch.autograd.set_detect_anomaly(True)

imgSize = 28
batchSize = 64
labelNum = 10

''' In order to plot, the following value should be 2. '''
dimLatentVector = 2
plot = True
zSpaceForPlot = torch.Tensor(np.rollaxis(np.mgrid[2:-2:20 * 1j, 2:-2:20 * 1j], 0, 3).reshape([-1, 2]))

encDimsLayer = [500, 500]
decDimsLayer = [500, 500]
discDimsLayer = [500, 500]
learningRate = 1e-3
numEpoch = 100

os.makedirs("../data/MNIST", exist_ok=True)
transform = transforms.Compose([transforms.Resize(imgSize), transforms.ToTensor()])
trainDataset = datasets.MNIST("./data/MNIST", train=True, download=True, transform=transform)
trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True)
testDataset = datasets.MNIST("./data/MNIST", train=False, download=True, transform=transform)
testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True, drop_last=True)

plotSampleNum = 10000
resultPath = createIncrementalPath('./result')
# distributor = GaussianDistributor(plotSampleNum, dimLatentVector, labelNum)
distributor = GaussianMixtureDistributor(plotSampleNum, dimLatentVector, labelNum)
# distributor = SwissRollDistributor(plotSampleNum, dimLatentVector, labelNum)
z, zId = distributor.getSample()
saveScatteredImage(z, zId, resultPath, 'prior_distribution_image.jpg')
saveImages(next(iter(trainLoader))[0], imgSize, batchSize, resultPath, 'trainImages.jpg')

aae = AdversarialAutoEncoder(imgSize, encDimsLayer, discDimsLayer, dimLatentVector, decDimsLayer).cuda()
encOptimizer = torch.optim.Adam(aae.encoder.parameters(), lr=learningRate)
aeOptimizer = torch.optim.Adam(itertools.chain(aae.encoder.parameters(), aae.decoder.parameters()), lr=learningRate)
discOptimizer = torch.optim.Adam(aae.discriminator.parameters(), lr=learningRate / 5)
crossentropyLoss = BCEWithLogitsLoss()
pixelWiseLoss = MSELoss()
distributor.setBatchSize(batchSize)

print("Train Start!")
for epoch in range(numEpoch):
    aae.train()
    for i, (images, labels) in enumerate(trainLoader):
        images = images.view(batchSize, -1).cuda()
        labels = F.one_hot(labels, num_classes=labelNum).cuda()
        sample, sampleId = distributor.getSample()
        sample = torch.tensor(sample).cuda()
        sampleId = F.one_hot(torch.tensor(sampleId, dtype=torch.int64), num_classes=labelNum).cuda()
        predictImages, realLogit, fakeLogit = aae(images, labels, sample, sampleId)
        
        likelihoodLoss = pixelWiseLoss(predictImages, images)
        aeOptimizer.zero_grad()
        likelihoodLoss.backward(retain_graph=True)
        aeOptimizer.step()
        
        discriminatorLoss = crossentropyLoss(fakeLogit, torch.zeros_like(fakeLogit)) + crossentropyLoss(realLogit, torch.ones_like(realLogit))
        discOptimizer.zero_grad()
        discriminatorLoss.backward(retain_graph=True)
        discOptimizer.step()
        
        generatorLoss = crossentropyLoss(fakeLogit, torch.ones_like(fakeLogit))
        encOptimizer.zero_grad()
        generatorLoss.backward()
        encOptimizer.step()
    
    print(f"epoch: {epoch + 1}, Discriminator Loss: {discriminatorLoss.item()}, Generator Loss: {generatorLoss.item()}, Likelihood Loss: {likelihoodLoss.item()}")
    
    with torch.no_grad():
        aae.eval()
        if dimLatentVector == 2 and plot:
            plotImages = aae.decoder(zSpaceForPlot.cuda())
            saveImages(plotImages.cpu(), imgSize, 400, resultPath, f"epoch_{epoch}.jpg")
            
            latentVectors = torch.Tensor().cuda()
            ids = torch.Tensor().cuda()
            for images, labels in testLoader:
                images = images.view(batchSize, -1).cuda()
                latentVectors = torch.cat([latentVectors, aae.encode(images)])
                ids = torch.cat([ids, labels.cuda()])
            saveScatteredImage(latentVectors.cpu(), ids.cpu(), resultPath, f"epoch_{epoch}_scatteredImage.jpg")
                
        elif plot:
            images, labels = next(iter(testLoader))
            images = images.view(batchSize, -1).cuda()
            plotImages = aae.forwardWithoutDiscriminator(images).reshape((batchSize, -1, imgSize, imgSize))
            saveImages(plotImages.cpu(), imgSize, 64, resultPath, f"epoch_{epoch}.jpg")