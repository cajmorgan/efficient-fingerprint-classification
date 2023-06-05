import os
import torch
import torch.nn as nn
from FingerprintModel import FingerprintModel
from src.data.FingerprintDataset import FingerprintDataset
import matplotlib.pyplot as plt 
import numpy as np
from src.visualization.visualize import plot_batch
import PIL.Image as Image
from src.data.CustomTransform import preprocess
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@click.command()
def main():
    logger = logging.getLogger(__name__)

    logger.info('Checks the feature map cosine similarity between pred-1, pred-2 and other photos')
    
    root = os.path.abspath('.')
    WEIGHTS_PATH = f'{root}/models/googlenet-fingerprints-1.pth'

    model = FingerprintModel(weights_path=WEIGHTS_PATH, dataset=None)
   
    # Remove the classifier layer
    model.network.fc = nn.Identity()

    img_path = f'{root}/data'
    img_1 = preprocess(Image.open(img_path + '/pred-1.png').convert('RGB')).unsqueeze(0)
    img_2 = preprocess(Image.open(img_path + '/pred-2.png').convert('RGB')).unsqueeze(0)
    img_3 = preprocess(Image.open(img_path + '/test/L/s0234_10.png').convert('RGB')).unsqueeze(0)
    img_4 = preprocess(Image.open(img_path + '/test/W/s1304_06.png').convert('RGB')).unsqueeze(0)
    img_5 = preprocess(Image.open(img_path + '/test/A/f0624_03.png').convert('RGB')).unsqueeze(0)
    
    images = torch.cat((img_1, img_2, img_3, img_4, img_5), 0)
    logger.info(model.similarity(images))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()