import torch
import os

from torch.utils.data import Dataset, DataLoader

class Homework1Dataset(Dataset):
    def __init__(self, device: torch.device, test: bool):
        self.images = torch.empty((0,3,128,128))
        self.actions = torch.empty((0))
        self.positions = torch.empty((0,2))

        for file in os.listdir('./hw1_data/'):
            tensor = torch.load(f'./hw1_data/{file}')
            if(file.startswith('actions')):
                self.actions = torch.cat((self.actions, tensor))
            if(file.startswith('states')):
                self.images = torch.cat((self.images, tensor))
            if(file.startswith('positions')):
                self.positions = torch.cat((self.positions, tensor))

        test_size = int(0.2*self.actions.shape[0])

        self.position_mean = self.positions.mean(dim=0)
        self.positions_std =self.positions.std(dim=0)

        self.positions = self.positions - self.position_mean
        self.positions = self.positions/self.positions_std
        
        if test:
            self.images = self.images[-test_size:]
            self.actions = self.actions[-test_size:]
            self.positions = self.positions[-test_size:]
        else:
            self.images = self.images[:-test_size]
            self.actions = self.actions[:-test_size]
            self.positions = self.positions[:-test_size]


        self.images = self.images.to(device)
        self.actions = self.actions.to(torch.int64).to(device)
        self.positions = self.positions.to(device)

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.actions[idx], self.positions[idx]

