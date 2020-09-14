from src.feature_extractor import embed_and_save,load_embedded
from src.face_recognition import FaceRecognition
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from src.feature_extractor import embed_and_save

recognition = FaceRecognition()
recognition.train()

captrueDevice =
