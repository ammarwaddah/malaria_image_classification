# Malaria Image Classification
Using Machine Learning and DeepLearning to detect the class of target ('parasitized', 'uninfected') using significant features given by the most linked features that are taken into consideration when evaluating the target.

## Table of Contents
* [Introduction](#introduction)
* [Dataset General info](#dataset-general-info)
* [Evaluation](#evaluation)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Run Example](#run-example)
* [Sources](#sources)

## Introduction
AI has become one of the basic tools that play a significant role in our daily lives, especially related to the reached accuracy when dealing with medical science using
Computer Vision, Natural Language Processing, and other fields of Artificial Intelligence that are related and intertwined with each other. Based on this point, I present my Computer Vision project with the aim of refining and showing my theoretical and practical skills acquired from my past studies in the field of Computer Vision, refining it gradually within practical projects to show as much detail as possible related to Computer Vision, thus give benefit to those who are new to this field when reviewing the code, so I present my project in solving the problem of Malaria Image Classification by blood cells, and put my suggestions for solving it with the best possible ways and the current capabilities using Machine Learning, and CNN.\
Hoping to improve it gradually in the coming times.

## Dataset General info
**General info about the dataset:**

    * This Dataset is an embeded based dataset, and we can fetch it using tensorflow datasets fetch method:
    tfds.load(name= 'malaria', data_dir='tmp', with_info=True, as_supervised=True, split=tfds.Split.TRAIN)
    * The Malaria dataset contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells from the thin blood smear slide images of segmented cells.
    
## Evaluation
The accuracy score is using to evaluate the performance, because of the balanced dataset, also, the model is evaluated by validation dataset and testing dataset.

## Technologies
* Programming language: Python.
* Libraries: Numpy, Matplotlib, Pandas, Seaborn, tensorflow, PIL, tensorflow_datasets. 
* Application: Jupyter Notebook.

## Setup
To run this project setup the following libraries on your local machine using pip on the terminal after installing Python:\
'''\
pip install numpy\
pip install matplotlib\
pip install pandas\
pip install seaborn\
pip install tensorflow\
pip install Pillow\
pip install tensorflow-datasets\
pip install pip-custom-platform\
'''\
To install these packages with conda run:\
'''\
conda install -c anaconda numpy\
conda install -c conda-forge matplotlib\
conda install -c anaconda pandas\
conda install -c anaconda seaborn\
conda install -c conda-forge tensorflow\
conda install -c anaconda pillow\
conda install -c conda-forge tensorflow-datasets\
conda install -c anaconda-platform anaconda-platform-cli\
'''

## Features
* I present to you my project solving the problem of Malaria Image Classification using CNN algorithm (LeNet, AlexNet, VGG-16, VGG-19) of some effective algorithm and techniques with a good analysis (EDA), and comparing between them using logical thinking, and put my suggestions for solving it in the best possible ways and the current capabilities using Machine Learning, and CNN.

### To Do:
**Briefly about the process of the project work, here are (some) insights that I took care of it:**

* Load the dataset from tensorflow public datasets.
* Perform files check is used to ensure that files are valid (here the perform step is done by using public dataset in tensorflow dataset).
* Doing EDA to ensure the status, correctness and type of images to ensure the balance of the data, view classes, explore what values are used to represent the image, and etc.
* Pre-processing the data (I used split functions that wrote from scratch instead of ImageDataGenerator) by take the average of the images shape to process all data into fixed size, Data Augmentation, Data shuffling and batching.
* Create function that used as DataGenerator for the training images because of the idea of data augmentation was used.
* Data shuffling and batching depends on the data.
* Trained on LeNet-5, AlexNet, VGG-16, VGG-19 (All Algorithms are implemented from scratch) with adam optimizer algorithm, and binary_crossentropy loss function.
* Visualize the results, whether training accuracy or error.

## Run Example
To run and show analysis, insights, correlation, and results between any set of features of the dataset, here is a simple example of it:

* Note: you have to use a jupyter notebook to open this code file.

1. Run the importing stage.

2. Load the dataset.

3. Select which cell you would like to run and show its output.

5. Run the rest of cells to end up in the training process and visualizing results.

4. Run Selection/Line in Python Terminal command (Shift+Enter).

## Sources
This data was taken from TensorFlow public datasets\
(tfds.load(name='malaria', data_dir='tmp', with_info=True, as_supervised=True, split=tfds.Split.TRAIN))
