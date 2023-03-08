from utils.hw1 import *
import torch

BATCH_SIZE = 128
EPOCHS = 200

device = torch.device('cuda')

trainset = Homework1Dataset(device, False)
validationset = Homework1Dataset(device, True)
# model = Predictor(3,3).to(device)
model = BasicNet().to(device)

trainModel(model, trainset, BATCH_SIZE, EPOCHS, validation = True, validationSet=validationset)