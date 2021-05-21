import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cv2
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder
import pickle
from keras_facenet import FaceNet

class FaceDetector:

    def __init__(self):
        self.svm_model = pickle.load(open("E:\\projects\python\\MyFaceRecognition\\OSVM_classifier.sav", 'rb'))
        self.data = np.load('E:\\projects\\python\\MyFaceRecognition\\faces_dataset_embeddings.npz')
        self.detector = MTCNN()

    def face_mtcnn_extractor(self, frame):
        result = self.detector.detect_faces(frame)
        return result

    def face_localizer(self, person):
        bounding_box = person['box']
        x1, y1 = abs(bounding_box[0]), abs(bounding_box[1])
        width, height = bounding_box[2], bounding_box[3]
        x2, y2 = x1 + width, y1 + height
        return x1, y1, x2, y2, width, height

    def face_preprocessor(self, frame, x1, y1, x2, y2, required_size=(160, 160)):
        face = frame[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        face_pixels = face_array.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = np.expand_dims(face_pixels, axis=0)
        print(samples)
        embedder = FaceNet()
        embeddeds= embedder.embeddings(samples)
        face_embedded = embeddeds[0]
        in_encoder = Normalizer(norm='l2')
        X = in_encoder.transform(face_embedded.reshape(1, -1))
        return X

    def face_svm_classifier(self, X):
        yhat = self.svm_model.predict(X)
        print(yhat)
        label = yhat[0]
        return label

    def face_detector(self):
        cap = cv2.VideoCapture(0)
        while True:
            __, frame = cap.read()
            result = self.face_mtcnn_extractor(frame)
            if result:
                for person in result:
                    x1, y1, x2, y2, width, height = self.face_localizer(person)
                    X = self.face_preprocessor(frame, x1, y1, x2, y2, required_size=(160, 160))
                    label = self.face_svm_classifier(X)
                    print(" Person : {} ".format(label))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 155, 255), 2)
                    if(label==1):
                        cv2.putText(frame, "Authorized", (x1, height),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                                lineType=cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "Unauthorized", (x1, height),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0),
                                    lineType=cv2.LINE_AA)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    facedetector = FaceDetector()
    facedetector.face_detector()
