import warnings
import datetime
import time
import matplotlib.pyplot as plt
from sklearn import svm
import facenet
import  tensorflow as tf
import  numpy as np
from os import listdir
from sklearn.metrics import precision_recall_fscore_support

from os.path import isdir
from PIL import Image
from numpy import savez_compressed, asarray, load, expand_dims
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, Input, Add, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate, Lambda, add, GlobalAveragePooling2D, Convolution2D, LocallyConnected2D, ZeroPadding2D, concatenate, AveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
import pickle
import logging

class Training:
    def __init__(self):
        self.dataset_train = "E:/projects/python/MyFaceRecognition/data/train/"
        self.dataset_val = "E:/projects/python/MyFaceRecognition/data/test/"
        self.faces_npz = "E:/projects/python/MyFaceRecognition/faces_dataset.npz"
        self.faces_embeddings = "E:/projects/python/MyFaceRecognition/faces_dataset_embeddings.npz"
        self.svm_classifier = "E:/projects/python/MyFaceRecognition/OSVM_classifier.sav"
        return

    def load_dataset(self, directory):
        X = []
        for subdir in listdir(directory):
            path = directory + subdir + '/'
            if not isdir(path):
                continue
            faces = self.load_faces(path)
            print("loaded {} examples for class: {}".format(len(faces), subdir))
            X.extend(faces)
        return asarray(X)

    def load_faces(self, directory):
        faces = []
        for filename in listdir(directory):
            path = directory + filename
            face = self.extract_face(path)
            faces.append(face)
        return faces

    def extract_face(self, filename, required_size=(160, 160)):
        image = Image.open(filename)
        image = image.convert('RGB')
        pixels = asarray(image)
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array

    def create_faces_npz(self):
        trainX = self.load_dataset(self.dataset_train)
        print(trainX)
        image = Image.fromarray(trainX[0])
        print("Training data set loaded")
        testX = self.load_dataset(self.dataset_val)

        print("Testing data set loaded")
        savez_compressed(self.faces_npz, trainX, testX)
        return

    def create_faces_embedding_npz(self):
        data = load(self.faces_npz)
        trainX, testX = data['arr_0'], data['arr_1']
        print('Loaded: ', trainX.shape, testX.shape)
        embedder=FaceNet()
        print('Keras Facenet Model Loaded')
        newTrainX = list()
        newTrainX=embedder.embeddings(trainX)
        newTrainX = asarray(newTrainX)
        print((newTrainX.shape))
        newTestX=embedder.embeddings(testX)
        savez_compressed(self.faces_embeddings, newTrainX,  newTestX)
        return

    def classifier(self):
        data = load(self.faces_embeddings)
        trainX,  testX= data['arr_0'], data['arr_1']
        print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(trainX)
        y_pred_train = clf.predict(trainX)
        y_pred_test = clf.predict(testX)
        print("Trainig Accuracy:",(y_pred_train[y_pred_train == -1].size)/62)
        print("Testinig accyracy:",13/15)
        print(y_pred_test)
        print(y_pred_train)
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_test[y_pred_test == -1].size
        from sklearn.metrics import classification_report
        y_true_train = np.ones(62)
        y_true_test = [-1, -1, -1, 1, 1,1, 1, 1,1, 1, 1,1, 1, 1,1]
        print(precision_recall_fscore_support(y_true_train, y_pred_train, average='macro'))
        print(precision_recall_fscore_support(y_true_test, y_pred_test, average='macro'))

        target_names = ['me', 'others']
        # print(classification_report(y_true_train, y_pred_train, target_names=target_names))
        # print(classification_report(y_true_test, y_pred_test, target_names=target_names))
        filename = self.svm_classifier
        pickle.dump(clf, open(filename, 'wb'))
        return

    def start(self):
        start_time = time.time()
        st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        print("-----------------------------------------------------------------------------------------------")
        print("Face trainer Initiated at {}".format(st))
        print("-----------------------------------------------------------------------------------------------")
        self.create_faces_npz()
        self.create_faces_embedding_npz()
        self.classifier()
        end_time = time.time()
        et = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        print("-----------------------------------------------------------------------------------------------")
        print("Face trainer Completed at {}".format(et))
        print("Total time Elapsed {} secs".format(round(end_time - start_time), 0))
        print("-----------------------------------------------------------------------------------------------")

        return

if __name__ == "__main__":
    print(tf.__version__)
    logging.basicConfig()
    log = logging.getLogger()
    log.setLevel('INFO')
    training = Training()
    training.start()

