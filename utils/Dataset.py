import os

import cv2
from PIL import Image
from numpy import asarray
from os import listdir
from os.path import isdir
from src.face_detector import extract_face, extract_face_

import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

class Dataset(object):
    basedir = ""
    frame_dir = os.path.join(THIS_FOLDER,"cw2_frames")
    frameRate = 0.5

    @staticmethod
    def load_img_from_dir(directory):
        img = []
        for filename in listdir(directory):
            filename = os.path.join(directory, filename)
            image = Image.open(filename)

            image = image.convert('RGB')
            image = asarray(image)

            img.append(image)

        return img

    @staticmethod
    def load_dataset(directory):
        faces = list()
        labels = list()
        # enumerate files
        for filename in listdir(directory):
            # path

            path = os.path.join(directory, filename)

            img = Dataset.load_img_from_dir(path)
            # get face
            detected, bb = extract_face_(img)
            detected_faces = [_ for _ in detected if _ is not None]
            faces.extend(detected_faces)
            labels.extend([filename for _ in range(len(detected_faces))])

        return asarray(faces), asarray(labels)

    @staticmethod
    def getFrame(sec, name, vidcap, count, write_dir):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite(os.path.join(write_dir, name, + str(count) + ".jpg"), image)  # save frame as JPG file

        return hasFrames

    @staticmethod
    def get_dataset():
        for name in listdir(Dataset.basedir):
            new_path = os.path.join(Dataset.frame_dir, name[:-4])
            if not os.path.exists(new_path):
                os.makedirs(new_path)

            vidcap = cv2.VideoCapture(os.path.join(Dataset.frame_dir, name))  # name = Thang.mp4
            sec = 0
            count = 1
            success = Dataset.getFrame(sec, name=name[:-4], vidcap=vidcap, count=count)
            while success:
                count = count + 1
                sec = sec + Dataset.frameRate
                sec = round(sec, 2)
                success = Dataset.getFrame(sec, name=name[:-4], vidcap=vidcap, count=count)

    @staticmethod
    def get_all_frames(vid_path, write_dir, name):
        new_path = write_dir + name
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        vidcap = cv2.VideoCapture(vid_path)  # name = Thang.mp4
        sec = 0
        count = 1
        success = Dataset.getFrame(sec, name=name, vidcap=vidcap, count=count, write_dir=write_dir)
        while success:
            count = count + 1
            sec = sec + Dataset.frameRate
            sec = round(sec, 2)
            success = Dataset.getFrame(sec, name=name, vidcap=vidcap, count=count, write_dir=write_dir)
frame_dir = os.path.join(THIS_FOLDER,"cw2_frames")
print(frame_dir)