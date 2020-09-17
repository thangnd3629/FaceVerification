import cv2
from PIL import Image
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import tensorflow as tf
from tensorflow import keras
import h5py
from src.face_detector import extract_face, extract_face_
from src.feature_extractor import get_embedding

from utils import Dataset
from tensorflow.keras.models import load_model
from utils import Dataset
from sklearn.model_selection import train_test_split
from os import listdir
import keras
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from src.feature_extractor import load_embedded
from model.backbone import InceptionResNetV1
from tensorflow.keras import models
import os

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def stripoff(str_name):
    for i in str_name:
        print(i)
    s = ""
    for i in str_name:
        if i.isalnum():
            s += i
    return s


s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'


def remove_accents(input_str):
    print(input_str)
    s = ''

    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s


class FaceRecognition:
    def __init__(self):
        self.model = SVC(kernel="rbf", probability=True)
        self.embbeding_model = InceptionResNetV1()

    @property
    def embedded_data(self):
        data = {}
        trainX, trainy, testX, testy = load_embedded()
        data["x_train"] = trainX
        data["y_train"] = trainy
        data["x_test"] = testX
        data["y_test"] = testy
        return data

    # prewhiten
    def preprocess_input(self):
        normalizer = Normalizer(norm="l2")
        self.x_train = normalizer.transform(self.embedded_data["x_train"])
        self.x_test = normalizer.transform(self.embedded_data["x_test"])

    def encode_label(self):
        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(self.embedded_data["y_train"])
        self.y_train = self.out_encoder.transform(self.embedded_data["y_train"])
        self.y_test = self.out_encoder.transform(self.embedded_data["y_test"])

    def train(self):
        # preprocess
        self.preprocess_input()
        self.encode_label()
        # train
        self.model.fit(self.x_train, self.y_train)
        yhat_train = self.model.predict(self.x_train)
        yhat_test = self.model.predict(self.x_test)
        score_train = accuracy_score(self.y_train, yhat_train)
        score_test = accuracy_score(self.y_test, yhat_test)

        # summarize
        print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))

    def predict_face(self, test_face_dir):
        embeded_face = []
        confident_level = []

        faces = Dataset.load_img_from_dir(test_face_dir)
        faces_index = []
        # start
        t0 = time.clock()
        extracted_face, bounding_box = extract_face_(faces)
        for index, face in enumerate(extracted_face):
            if face is not None:
                embbeded = get_embedding(self.embedding_model, face)
                embbeded = np.expand_dims(embbeded, axis=0)

                in_encoder = Normalizer(norm='l2')
                embbeded = in_encoder.transform(embbeded)

                embeded_face.append(embbeded[0])
                faces_index.append(index)

        embeded_face = asarray(embeded_face)
        y_hat = self.model.predict(embeded_face)
        precision = self.model.predict_proba(embeded_face)
        # stop
        t1 = time.clock() - t0
        print("Elasped time: {:.2f} second".format(t1))
        for i in precision:
            confident_level.append(np.amax(i))
        result = self.out_encoder.inverse_transform(y_hat)

        # visualization
        result_index = 0
        person_f = []
        for j, i in enumerate(faces):
            # detected face
            if j in faces_index:

                x, y, w, h = bounding_box[j]
                cv2.rectangle(faces[j], (x, y), (x + w, y + h), (36, 255, 12), 1)

                predicted_label = remove_accents(str(result[result_index].astype(np.unicode))) if confident_level[
                                                                                                      result_index] > 0.4 else "Unknown"
                if predicted_label == "Unknown":
                    cv2.putText(faces[j], predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)
                else:
                    cv2.putText(faces[j], predicted_label + ":" + str(
                        round(confident_level[result_index] * 100)) + "%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)

                result_index += 1

            person_f.append(faces[j])
        plt.figure(figsize=(100, 100))  # specifying the overall grid size
        for i in range(len(person_f)):
            plt.subplot(len(person_f) / 5 + 1, 5, i + 1)  # the number of images in the grid is 5*5 (25)
            plt.imshow(person_f[i])

        plt.set_axis("off")
        plt.show()
        return result

    def predict_single_face(self, frame):
        # extracted_face, bounding_box = extract_face_(np.expand_dims(frame,axis=0))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE

        )
        if len(face) == 0:
            return
        for f in face:
            # x,y,w,h = bounding_box[0]
            x, y, w, h = f

            x1, y1 = abs(x), abs(y)
            x2, y2 = x1 + w, y1 + h
            # extract the face
            face = frame[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize((160, 160))
            extracted_face = asarray(image)
            embed = get_embedding(self.embbeding_model, extracted_face)
            embed = np.expand_dims(embed, axis=0)
            in_encoder = Normalizer(norm='l2')
            embed = in_encoder.transform(embed)
            y_hat = self.model.predict(asarray(embed))
            precision = self.model.predict_proba(asarray(embed))
            precision = np.amax(precision[0])
            result = self.out_encoder.inverse_transform(y_hat)
            result = result[0]
            print(stripoff(result))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 1)

            predicted_label = stripoff(str(result.astype(np.unicode))) if precision > 0.4 else "Unknown"

            if predicted_label == "Unknown":

                cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (36, 255, 12), 2)
            else:
                cv2.putText(frame, predicted_label + ":" + str(
                    round(precision * 100)) + "%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (36, 255, 12), 2)
