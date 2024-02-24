import torch

''' From zero-mean gaussain distribution'''
def getKullbackLeiblerLoss(mu, sigma):
    return 0.5 * torch.sum(mu ** 2 + sigma ** 2 - torch.log(1e-8 + sigma ** 2) - 1, 1)

def getCrossEntropyLoss(predict, target):
    return -torch.sum(target * torch.log(predict + 1e-8) + (1 - target) * torch.log(1 - predict + 1e-8), 1).mean()