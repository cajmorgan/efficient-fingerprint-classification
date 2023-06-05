from torchvision import datasets, transforms

class CustomTransform:
    def __init__(self, contrast, sharpness):
        self.contrast = contrast
        self.sharpness = sharpness

    def __call__(self, img):
        img = transforms.functional.adjust_gamma(img, 0.6)
        img = transforms.functional.adjust_sharpness(img, self.sharpness)
        img = transforms.functional.adjust_contrast(img, contrast_factor=self.contrast)
        img = transforms.functional.adjust_brightness(img, 1.2)
        
        return img 
    

preprocess = transforms.Compose([
            transforms.CenterCrop(400),
            CustomTransform(3, 2),
            transforms.Resize(227),
            transforms.ToTensor(),
        ])