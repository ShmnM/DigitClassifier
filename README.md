# Handwritten Digit Classifier 

The performance of a decision tree with custom feature extraction for 16x16 grayscale images from 
U.S. Postal Service dataset (Le Cun et al., 1990) is compared to a SVM for digit recognition accuracy.

## Overview
The custom feature extraction focused on symmetry, concavity, pixel density which is fed into a decision tree.

The SVM approach is fed raw pixel input

The custom decision tree achieved ~40% accuracy while the SVM reached
~93% accuracy, showing the tradeoff using interpretable handcrafted 
feature extraction instead of powerful machine learning technology.

A full write-up is available in the linked paper [Report](Report.pdf).

## Dataset
- Training set: 7291 images
- Test set: 2007 images
- Each image is 16x16 pixel represented by 256 intensity values between -1 and 1

## Features Engineered
- Symmetry to group numbers that are symmetrical such as 1, 8, 0
  measured by the horizontal pixel overlap of digits
- Concavity to distinguish digits that curve inwards from straighter ones
  measured by horizontal scanlines counting the number of holes in each row
- Pixel density by row and column to distinguish wide digits from narrow digits
  measured by averaging the number of pixels found in each row and column

## Note
StandardScaler normalization before the SVM was added after the project completion
after taking a Machine Learning course. Results in the paper used unscaled inputs.

## How to Run
1. Install dependencies `pip install scikit-learn matplotlib numpy`
2. Put `train-data.txt` and `test-data.txt` in the same directory
3. run `python DigitClassification.py`
4. The first chart is feature extraction, close the window to get the SVM method