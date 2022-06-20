# Fire event detection dataset
This repository contains the instructions on how to download and prepare the fire event detection dataset. There are also instructions on how to train and evaluate a convolutional neural network on this dataset as a baseline for further research on this topic.

The data can be downloaded from here:
- https://www.dropbox.com/s/7tkeh7wwuofe9zl/data.zip

but please follow the instructions below to recreate the train/validation/test split of the paper, and if you want to re-produce the main results of the paper.

## Clone repository and setup environment

    git clone https://github.com/anonymous7483/fire-event-detection-dataset.git
    cd fire-event-detection-dataset
    pip install -r requirements.txt

## Download data
The data can be downloaded from the following Dropbox link:

    wget https://www.dropbox.com/s/7tkeh7wwuofe9zl/data.zip
    unzip data.zip
    
There should now be a folder named "wav" with the audio source files in the working directory.

## Prepare data
This will create a HDF5 file with the prepared dataset, by default using the path "./wav" for the audio source files.

    python prepare_data.py

## Train model
This will train a model, by default the training output is stored in the folder: "experiments/baseline", with the configuration used to find the model presented in the paper.

    python baseline.py "train_model"

## Evaluate model
This will evaluate the trained model on the test dataset.

    python baseline.py "evaluate_model"
