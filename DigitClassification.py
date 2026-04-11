from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# in case i want to see what im working with. this displays the image
def seeImages(image, label):
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()

#Step 0 Preprocessing ////////////////////////////////////////////////////////////////////
               
# turns the string of numbers into an image 
def arrayToImage(intensity, size=16):
    image = []
    for i in range(size):
        row = intensity[i*size : (i+1)*size]
        image.append(row)
    return image

# simple thresholding getting rid of noise 
def thresholding(image, threshold):
    return [[1 if p > threshold else 0 for p in row] for row in image]

# extract image and label from txt
def load_data(filename):
    intensities = []
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            data = line.strip().split()
            label = int(float(data[0]))
            intensity = list(map(float, data[1:]))
            intensities.append(intensity)
            labels.append(label)
    return intensities, labels

#Step 1 try to extract the features ///////////////////////////////////////////////////////// 

# find the number of "holes" in a row
def countHolesInRow(row):
        holes = 0
        in_hole = False
        
        for i in range(len(row)- 1):
                if not in_hole and row[i] == 1 and row[i+1] == 0:
                        in_hole = True
                elif in_hole and row[i] == 0 and row[i+1] == 1:
                        holes += 1
                        in_hole = False
                        
        #if in hole at end count it as well
        if in_hole:
                holes += 1
        # each hole is counted twice because of an entry and exit. integer division by 2 fixes this
        return holes//2 

# compare # holes between adjacent rows, if change from > 0 to 0 implies concavity
def countConcavity(image):
        total_holes = 0
        prev_holes = 0

        for row in image:
                current_holes = countHolesInRow(row)
                # if previous row has holes and current row has 0 = intersection 
                if prev_holes > 0 and current_holes == 0:
                        total_holes += 1
                prev_holes = current_holes
                        
        return total_holes
                
#some numbers have a lot of symmetry (1,8,0)
def symmetry(image):
    size = len(image)  
    notMatch = 0
    match = 0
    
    for row in image:
        for i in range(size // 2):
            match += 1
            if row[i] != row[size - 1 - i]:
                notMatch += 1
    
    symmetryScore = 1 - (notMatch / match)
    return symmetryScore
                
# find the shape by iterating through the image and getting pixel density by row and by column
def shapeFeatures(image):
    rowDensities = [sum(row)/len(row) for row in image]

    numRows = len(image)
    numCols = len(image[0]) if numRows > 0 else 0
    colDensities = []
    for col_index in range(numCols):
        sumCols = 0
        for rows in range(numRows):
            sumCols += image[rows][col_index]
        colDensities.append(sumCols / numRows if numRows > 0 else 0)

    X = sum(d > 0.5 for d in rowDensities)
    Y = sum(d > 0.5 for d in colDensities)

    return X, Y
    
 
# Step 2 classification ///////////////////////////////////////////////////////// 
def extractFeatures(intensity):
    image = arrayToImage(intensity)
    thresholded = thresholding(image, 0.3)
    symmetryScore = symmetry(thresholded)
    concavity = countConcavity(thresholded)
    xShape, yShape = shapeFeatures(thresholded)   

    return [symmetryScore, concavity, xShape, yShape]

def trainModel(trainIntensities, labels):
    X = [extractFeatures(i) for i in trainIntensities]
    tree = DecisionTreeClassifier()
    tree.fit(X, labels)
    return tree

def predictModel(tree, testIntensities):
    return tree.predict([extractFeatures(i) for i in testIntensities])

# Main ///////////////////////////////////////////////////////// 

def main():
        # extracting features 
        trainIntensities, trainLabels = load_data("train-data.txt")
        testIntensities, testLabels = load_data("test-data.txt")
        #seeImages(arrayToImage(testIntensities[1]), testLabels[1])

        tree = trainModel(trainIntensities, trainLabels)
        predictions = predictModel(tree, testIntensities)

        accuracy = accuracy_score(testLabels, predictions)
        print("Feature Extraction + Decision Tree Accuracy:", accuracy)
 
        #generate confusion matrix 
        #first chart is Decision Tree results, close the window to view SVM results
        cm = confusion_matrix(testLabels, predictions)
        print("Confusion Matrix:\n", cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Feature Extraction + Decision Tree Confusion Matrix")
        plt.show()

        # using svm model from scikit 
        #normalization was done after taking machine learning class, paper does not reflect this
        scaler = StandardScaler()
        trainNormalized = scaler.fit_transform(trainIntensities)
        testNormalized = scaler.transform(testIntensities)
        
        model = SVC()
        model.fit(trainNormalized, trainLabels)
    
        predictions = model.predict(testNormalized)
        accuracy = accuracy_score(testLabels, predictions)
        print("SVM Accuracy:", accuracy)
        
        #generate confustion matrix 
        cm = confusion_matrix(testLabels, predictions)
        print("Confusion Matrix:\n", cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("SVM Confusion Matrix")
        plt.show()

if __name__ == "__main__":
    main()

