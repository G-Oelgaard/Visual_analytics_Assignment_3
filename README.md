# Visual_analytics_Assignment_3
## ------ SCRIPT DESCRIPTION ------
This repository contains a script that will create a classifier using VGG16 and the CIFAR10 dataset.

The model will:
- train a classifier
- save plots of loss and accuracy
- save a classification report

## ------ DATA ------
The data is the CIFAR10 dataset from Keras.

## ------ REPO STRUCTURE ------
"src" FOLDER:
- This folder contains the .py script to train the classifier and save the report and plots

"in" FOLDER:
- This is where the data used in the scripts should be placed. In this case no data needs to be provided.

"out" FOLDER:
- This is where the report and plot will be saved

"utils" FOLDER:
- This folder should include all utility scripts used by the main script.

## ------ RESULTS ------
As we can see in the classification report and model plot, the test and training scores are roughly following each other, yet still being a subpar model. This means that it is not suffering from over or underfitting, but that it is still bad at reliably predicting the correct scores. 
