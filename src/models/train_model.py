import os
from FingerprintModel import FingerprintModel
from Trainer import Trainer
from src.data.FingerprintDataset import FingerprintDataset
import matplotlib.pyplot as plt 
import numpy as np
from src.visualization.visualize import plot_batch
from src.data.CustomTransform import preprocess
import PIL.Image as Image
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv



@click.command()
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    
    root = os.path.abspath('.')
    WEIGHTS_PATH = f'{root}/models/googlenet-fingerprints-1.pth'
    TRAIN_PATH = f'{root}/data/train'
    TEST_PATH = f'{root}/data/test'

    dataset = FingerprintDataset(train_path=TRAIN_PATH, test_path=TEST_PATH)
    model = FingerprintModel(weights_path=WEIGHTS_PATH, dataset=dataset)
    trainer = Trainer(network=model.network, trainloader=dataset.trainloader, epochs=1, lr=0.0001, logger=logger)
    trainer.train()
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()