from torch import nn
import torch
import torch.nn.init as init

def initialize_linear(layer):
    if isinstance(layer, nn.Linear):
        init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            init.zeros_(layer.bias)

class Encoder(nn.Module):
    def __init__(self, imgSize: int, dimsHidden: list[int], dimLatentVector: int):
        super().__init__()
        self.dimHidden = dimsHidden
        self.dimLatentVector = dimLatentVector
        self.hiddenLayer = nn.Sequential()

        self.preDimOfLayer = imgSize * imgSize
        self.hiddenLayer.append(nn.Linear(self.preDimOfLayer, dimsHidden[0]))
        self.hiddenLayer.append(nn.ReLU())
        self.hiddenLayer.append(nn.Dropout(p=0.1))
        self.preDimOfLayer = dimsHidden[0]
        self.hiddenLayer.append(nn.Linear(self.preDimOfLayer, dimsHidden[1]))
        self.hiddenLayer.append(nn.ReLU())
        self.hiddenLayer.append(nn.Dropout(p=0.1))
        self.preDimOfLayer = dimsHidden[1]
        self.outLayer = nn.Linear(self.preDimOfLayer, self.dimLatentVector * 2)
        self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor):
        x = self.hiddenLayer(x)
        x = self.outLayer(x)
        
        mu = x[:, :self.dimLatentVector]
        # Add a small epsilon for numerical stability
        sigma = self.softplus(x[:, self.dimLatentVector:]) + 1e-6
        
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, dimLatentVector: int, dimsHidden: list[int], imgSize: int):
        super().__init__()
        self.dimLatentVector = dimLatentVector
        self.hiddenLayer = nn.Sequential()
        self.dimHidden = dimsHidden
        
        self.preDimOfLayer = dimLatentVector
        self.hiddenLayer.append(nn.Linear(self.preDimOfLayer, dimsHidden[0]))
        self.hiddenLayer.append(nn.ReLU())
        self.hiddenLayer.append(nn.Dropout(p=0.2))
        self.preDimOfLayer = dimsHidden[0]
        self.hiddenLayer.append(nn.Linear(self.preDimOfLayer, dimsHidden[1]))
        self.hiddenLayer.append(nn.ReLU())
        self.hiddenLayer.append(nn.Dropout(p=0.2))
        self.preDimOfLayer = dimsHidden[1]
        
        self.outLayer = nn.Linear(self.preDimOfLayer, imgSize * imgSize)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor):
        return self.sigmoid(self.outLayer(self.hiddenLayer(x)))

class VariationalAutoEncoder(nn.Module):
    def __init__(self, imgSize: int, encDimsHidden: list[int], dimLatentVector: int, decDimsHidden: list[int]):
        super().__init__()
        self.encoder = Encoder(imgSize, encDimsHidden, dimLatentVector)
        self.decoder = Decoder(dimLatentVector, decDimsHidden, imgSize)
        initialize_linear(self)
        
    def forward(self, x: torch.Tensor):
        mu, sigma = self.encoder(x)
        return mu, sigma, self.decode(self.getReparameterization(mu, sigma))
    
    def getReparameterization(self, mu: torch.Tensor, sigma: torch.Tensor):
        return mu + sigma * torch.randn_like(sigma)
    
    def encode(self, x):
        mu, sigma = self.encoder(x)
        return self.getReparameterization(mu, sigma)
    
    def decode(self, z):
        return self.decoder(z)

class AdversarialEncoder(Encoder):
    def __init__(self, imgSize, dimsHidden, dimLatentVector):
        super().__init__(imgSize, dimsHidden, dimLatentVector)
        self.outLayer = nn.Linear(dimsHidden[-1], dimLatentVector)
    
    def forward(self, x):
        return self.outLayer(self.hiddenLayer(x))

class Discriminator(nn.Module):
    def __init__(self, dimLatentVector, disciminatorDimsHidden, dimOutput=1):
        super().__init__()
        self.dimLatentVector = dimLatentVector + 10
        self.hiddenLayer = nn.Sequential()
        self.dimHidden = disciminatorDimsHidden
        
        self.preDimOfLayer = self.dimLatentVector
        self.hiddenLayer.append(nn.Linear(self.preDimOfLayer, disciminatorDimsHidden[0]))
        self.hiddenLayer.append(nn.ReLU())
        self.hiddenLayer.append(nn.Dropout(p=0.2))
        self.preDimOfLayer = disciminatorDimsHidden[0]
        self.hiddenLayer.append(nn.Linear(self.preDimOfLayer, disciminatorDimsHidden[1]))
        self.hiddenLayer.append(nn.ReLU())
        self.hiddenLayer.append(nn.Dropout(p=0.2))
        self.preDimOfLayer = disciminatorDimsHidden[1]
        
        self.outLayer = nn.Linear(self.preDimOfLayer, dimOutput)

    def forward(self, x):
        return self.outLayer(self.hiddenLayer(x))
        
class AdversarialAutoEncoder(nn.Module):
    def __init__(self, imgSize: int, encDimsHidden: list[int], disciminatorDimsHidden: list[int], dimLatentVector: int, decDimsHidden: list[int]) -> None:
        super().__init__()
        self.encoder = AdversarialEncoder(imgSize, encDimsHidden, dimLatentVector)
        self.decoder = Decoder(dimLatentVector, decDimsHidden, imgSize)
        self.discriminator = Discriminator(dimLatentVector, disciminatorDimsHidden)
        initialize_linear(self)
        
    def forward(self, x: torch.Tensor, xId: torch.Tensor, realZ: torch.Tensor, realZId: torch.Tensor):
        fakeZ = self.encode(x)
        y = self.decode(fakeZ)
        
        realLogit = self.discriminator(torch.concat([realZ, realZId], 1))
        fakeLogit = self.discriminator(torch.concat([fakeZ, xId], 1))
        
        return y, realLogit, fakeLogit
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, sample):
        return self.decoder(sample)
    
    def forwardWithoutDiscriminator(self, x):
        return self.decode(self.encode(x))