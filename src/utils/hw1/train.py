import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def trainModel(model : nn.Module, dataset: Dataset, batch_size: int, epochs: int, validationSet = None, validation=False, ):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    trainLoader = DataLoader(dataset, batch_size)
    validationLoader = None

    if validation:
        validationLoader = DataLoader(validationSet, batch_size)

    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            images, actions, positions = data

            optimizer.zero_grad()

            outputs = model(images, actions)
            loss = criterion(outputs, positions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        if validation:
            valid_loss = 0.0
            model.eval()  
            for i, data in enumerate(validationLoader, 0):
                images, actions, positions = data

                optimizer.zero_grad()

                outputs = model(images, actions)
                loss = criterion(outputs, positions)
                loss.backward()
                optimizer.step()
                valid_loss += loss.item()
            print(f'[Epoch {epoch + 1:2d}] Loss: {running_loss/len(trainLoader):.6f}\tValidation: {valid_loss/len(validationLoader):.6f}')
        else:
            print(f'[Epoch {epoch + 1:2d}] Loss: {running_loss/len(trainLoader):.6f}')


def evaluate(model : nn.Module, dataset: Dataset, batch_size: int):
    criterion = nn.MSELoss()

    testLoader = DataLoader(dataset, batch_size)

    running_loss = 0.0
    for i, data in enumerate(testLoader, 0):
        images, actions, positions = data
        outputs = model(images, actions)
        loss = criterion(outputs, positions)
        running_loss += loss.item()
    
    print(f'[Evaluation] Loss: {running_loss/len(testLoader):.6f}')