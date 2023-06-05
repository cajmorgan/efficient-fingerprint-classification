efficient-fingerprint-identification
==============================

A class based fingerprint identification using Convolution Neural Networks

## Dataset
This is a model used for predicting 4 types of fingerprint classes:
- Arch
- Left Loop
- Right Loop
- Whirl 

The dataset is composed of image from the *NIST Special Database 4*, that originally uses 5 classes, where the current Arch is splitted as:
- Tented Arch
- Arch

Unfortunately, this split is pretty ambigious within this dataset as many of the labeled images are of a pretty low quality and are sometimes not labeled correctly. 

## Architecture
Originally experimenting with different arcitechtures, by first using a custom convolution model influenced by a selection of papers, the decision was later made to continue the process with a pretrained model, as it generated higher testing accuracy.


### Results
The proposed architecture uses transfer learning from weights of a pretrained GoogLeNet model (PyTorch), replacing the last layer with an appropiate output, achieving a classification accuracy of 93%. The test dataset uses images from the same source as the train dataset. This accuracy might be lower for other datasets where the source of the images are different, for example, by using different fingerprint scanners.


## Usage
This repo and it's weights can be used as a starting point for software that would need to handle classification of fingerprints and as a start for an efficient fingerprint identification system. In a larger database, this would allow the exclusion of the majority of samples by first classifying the fingerprint before identifying it.  

### Identification
The feature maps that the network outputs before the classification layer can be seen as dimensional reduced data containing important information of the fingerprint images. These feature maps can be used in order to identify specific fingerprints. Unfortunately, with the lack of different fingerprint images from the same person, this couldn't be developed further at the time being.

With some small similarity tests using the naive approach of cosine similiarity, it was shown that the different images of the same fingerprint for one specific person generally has a cosine similarity closer to 1 than features maps of fingerprints from other persons. 

This idea can be further developed by using a ReID model, with the appropiate data and by using a Triplet Loss function, in order to minimize the distance between a person's fingerprint from different images/scanners and maximize the distance between fingerprints of the same class that are **not** from the same person. 



## Commands
The commands are specified in the make file:
- **data**: generates the dataset from the raw data using the structure provided by *NIST Special Database 4*
- **predict**: a test prediction of a specified image
- **train**: train the model from the processed data
- **accuracy**: checks the accuracy for test test dataset from the selected weights in /models
- **similarity**: calculates the cosine similarity between pred-1, pred-2 and a selection of other test images






