import os
import numpy as np
import logging
import shutil
import math
import random
import copy 

class DataSorter:
    def __init__(self, logger):
        self.logger = logger
        self.data_path = ""
        self.path = ""
        self.image_names = []
        self.classified_images = []
        self.image_data = []
        self.model = None
        self.preprocess = None
        self.logger.info('Data Sorter initialized')
        self.test_set = []

        return
    
    def clean(self):
        shutil.rmtree(self.labeled_path)
        shutil.rmtree(self.test_path)
        os.makedirs(self.labeled_path)
        os.makedirs(self.test_path)

        labels = ['A', 'L', 'R', 'W']

        for l in labels:
            os.makedirs(f'{self.labeled_path}/{l}')
            os.makedirs(f'{self.test_path}/{l}')

    def load_images(self):
        self.logger.info('Loading images...')
        
        self.data_path = os.path.abspath('./data/')
        self.labeled_path = self.data_path + '/train/'
        self.test_path = self.data_path + '/test/'
        self.path = os.path.abspath("./data/raw/fingerprints/")

        self.clean()
        
        for d,s, files in os.walk(self.path):
            for i in range(len(files)-1, -1, -1):
                if not files[i].endswith('.png'):
                    del files[i]
                    continue
                self.image_names.append(files[i][:-4])
    
    def classify_images(self):
        self.logger.info('Classifying images from txt files...')

        for img in self.image_names:
            img_dict = {'img_path': '', 'class_letter': ''}
            general_path = f'{self.path}/{img}'
            img_dict['img_path'] = general_path + '.png'

            text_path = general_path + '.txt'
            with open(text_path) as f:
                class_letter = f.readlines()[1][7]
                if class_letter == 'T': class_letter = 'A'

                img_dict['class_letter'] = class_letter
            
            self.classified_images.append(img_dict)

    def split_set(self):
        percent = 0.03
        for i in range(math.floor(len(self.classified_images)*percent)):
            rand = random.randint(0, len(self.classified_images)-1 -i)
            to_copy = copy.deepcopy(self.classified_images[i])
            self.test_set.append(to_copy)
            del self.classified_images[i]
            

    def copy_images(self):

        def copy_it(set, path):
            for i, img_dict in enumerate(set):
                img_path = img_dict['img_path']
                class_letter = img_dict['class_letter']
                self.logger.info(f'Copying image {i} to {class_letter}')
                
                if class_letter == 'L':
                    shutil.copy(img_path, path + 'L/')
                elif class_letter == 'W':
                    shutil.copy(img_path, path + 'W/')
                elif class_letter == 'R':
                    shutil.copy(img_path, path + 'R/')
                elif class_letter == 'A':
                    shutil.copy(img_path, path + 'A/')

        copy_it(self.classified_images, self.labeled_path)
        copy_it(self.test_set, self.test_path)
        


        
# logger = logging.getLogger(__name__)
# data = DataSorter(logger)
# data.load_images()
# data.classify_images()
# data.split_set()
# data.copy_images()