import torch
from utils.hw2.model import BasicNet
from utils.hw2.train import startTraining

device = torch.device('cuda')

model = BasicNet().to(device)

startTraining(model, device=device)