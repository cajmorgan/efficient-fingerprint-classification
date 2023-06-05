import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Trainer():
    def __init__(self, network, trainloader, epochs, logger, lr=0.0001):
        self.network = network
        self.trainloader = trainloader
        self.cost_func = nn.CrossEntropyLoss()
        self.lr = lr
        self.epochs = epochs
        self.logger = logger
        self.optimizer = optim.Adam(self.network.parameters(), self.lr)
        self.losses = []


    def train(self, stop_cost = 0.001):
        self.network.train()
        self.logger.info(f'Initialized training with lr: {self.lr}, epochs: {self.epochs} and a stop cost of: {stop_cost}')

        for epoch in range(self.epochs):
        
            running_loss = 0.0
            
            for i, data in enumerate(self.trainloader):
                images, labels = data
                self.optimizer.zero_grad()
                
                outputs = self.network(images)
                cost = self.cost_func(outputs, labels)
                cost.backward()
                self.optimizer.step()
                
                running_loss += cost.item()
                
                if i % 10 == 9:
                    self.logger.info(f'[{epoch + 1}, {i + 1:5}] loss: {running_loss/10:.3f}')
                    self.losses.append(running_loss/10)
                    running_loss = 0.0
                
            if (running_loss / 10) < stop_cost:
                self.logger.info(f'Stopping as {running_loss / 10} < {stop_cost}')
                break

    def save_parameters(self, path):
        torch.save(self.network.state_dict(), path)