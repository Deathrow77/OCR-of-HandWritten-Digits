from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2

mnist = datasets.load_digits()

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), np.array(mnist.target), test_size=0.25, random_state=42)

(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)
print(testData[:])
print("Training Data Points : {}".format(len(trainLabels)))
print("Validation Data Points : {}".format(len(valLabels)))
print("Testing Data Points: {}".format(len(testLabels)))
kVals = range(3,30,2)
accuracies = []

for k in range(3,30,2):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    score = model.score(valData, valLabels)
    print("k=%d : Accuracy = %.2f%%"%(k, score*100))
    accuracies.append(score)

i = np.argmax(accuracies)
print("k = %d achieved the maximum accuracy of %.2f%% on the validation data"%(kVals[i], accuracies[i]*100))


model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)
predictions = model.predict(testData)

print("RESULTS : \n")
print(classification_report(testLabels, predictions))

for i in np.random.randint(0, high=len(testLabels), size=(5,)):
    image = testData[i]
    prediction = model.predict(image)[0]

    image = image.reshape((8,8)).astype("uint8")
    image = exposure.rescale_intensity(0, out_range=(0,255))
    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)


    print("The digit is : {}".format(prediction))

    cv2.imshow("Image", image)
    cv2.waitKey(0)

