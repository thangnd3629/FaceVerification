from src.face_detector import extract_face, extract_face_
from src.feature_extractor import embed_and_save,load_embedded
from src.face_recognition import FaceRecognition
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from src.feature_extractor import embed_and_save
from numpy import expand_dims
recognition = FaceRecognition()
#embed_and_save(img_dir="",facenet_model=recognition.embbeding_model)
recognition.train()

import cv2
captureDevice = cv2.VideoCapture(0) #captureDevice = camera

while True:
    ret, frame = captureDevice.read()
    recognition.predict_single_face(frame)
    # extracted_face, bounding_box = extract_face_(np.expand_dims(frame, axis=0))
    # x, y, w, h = bounding_box[0]
    # if extracted_face[0] is None:
    #     continue
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 1)

    cv2.imshow('my frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
