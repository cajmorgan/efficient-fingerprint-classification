import torch
import torch.nn as nn
from torchvision import models


class FingerprintModel():
    classes = classes = ('A', 'L', 'R', 'W')

    def __init__(self, weights_path, dataset):

        self.weights = weights_path
        self.dataset = dataset
        self.network = models.get_model(name="googlenet", weights=models.GoogLeNet_Weights.DEFAULT)
        
        # Replace last layer to 5 outputs
        self.network.fc = nn.Linear(1024, 5, bias=True)
        self.network.load_state_dict(torch.load(weights_path))

    def __call__(self, images):
        output = self.network(images)
        return output
    
    def predict(self, img):
        self.network.eval()
        outputs = self.network(img)
        _, predicted = torch.max(outputs, 1)
        return self.classes[predicted]
                
    def batch_predictions(self, loader):
        self.network.eval()

        dataiter = iter(loader)
        images, labels = next(dataiter)
        outputs = self.network(images)
        _, predictions = torch.max(outputs, 1)
        return predictions
    

    def accuracy(self):
        correct = 0
        total = 0
        
        self.network.eval()
        with torch.no_grad():
            for i, data in enumerate(self.dataset.testloader):
                images, labels = data
                
                outputs = self.network(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return f'Accuracy: {round(100*(correct / total), 3)}%'
    
    """
    Simple cosine similarity used on the layer before the classifier - 
    For more accurate similarity a REID model should be trained on appropriate data
    """

    def similarity(self, images):
        self.network.eval()
        with torch.no_grad():
            img = self.network(images)
            img_1 = img[0]
            
            print((img @ img_1) /  (((torch.sum(img**2, axis=1))**(1/2))*(torch.sum(img_1**2)**(1/2))))

    

    
