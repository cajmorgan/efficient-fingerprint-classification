import os
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
@click.argument('relative_path', type=click.Path(exists=True))
def main(relative_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    root = os.path.abspath('.')
    WEIGHTS_PATH = f'{root}/models/googlenet-fingerprints-1.pth'

    img_path = f'{root}/{relative_path}'

    img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0)
    model = FingerprintModel(weights_path=WEIGHTS_PATH, dataset=None)
    prediction = model.predict(img)
    logger.info(f'Predict class for {img_path}: {prediction}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()