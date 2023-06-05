from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from src.data.CustomTransform import preprocess

class FingerprintDataset():

    def __init__(self, train_path, test_path):
        batch_size = 16
      
        trainset = datasets.ImageFolder(train_path, transform=preprocess)
        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        testset = datasets.ImageFolder(test_path, transform=preprocess)
        self.testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

        

    def get_sample(self, train=False):
        if train == True:
            dataiter = iter(self.trainloader)
        else:
            dataiter = iter(self.testloader)

        return next(dataiter)