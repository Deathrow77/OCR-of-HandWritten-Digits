from sklearn import preprocessing
from sklearn import datasets
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter


dataset = datasets.fetch_mldata('MNIST Original')

features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')

list_hog_ft = []
for feature in features:
    fd = hog(feature.reshape((28,28)), orientations=9, pixels_per_cell=(14,14), cells_per_block=(1,1), visualize=False )
    list_hog_ft.append(fd)

hog_features = np.array(list_hog_ft, 'float64')
pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)

print("Total number of digits in the dataset : ",Counter(labels))

clf = LinearSVC()
clf.fit(hog_features, labels)
joblib.dump((clf,pp), "digits_cls.pkl", compress=3)

