# get the face embedding for one face
import numpy
from numpy import expand_dims
from utils.Dataset import Dataset
from sklearn.model_selection import train_test_split
from keras.models import load_model
from numpy import asarray
from numpy import savez_compressed
from numpy import load
import os
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER,"detected_face_embedded_with_unknown.npz")
save_path = my_file

def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # prewhiten/standardize
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def embed_and_save(img_dir ,facenet_model):
    img_dir = r"C:\Users\Thang\Documents\FaceVerification\utils\cw2_frames"
    #embedded_face\cw2_frames
    # load the face dataset
    person, labels = Dataset.load_dataset(img_dir)
    x_train, x_test, y_train, y_test = train_test_split(person, labels, test_size=0.25, random_state=42)
    # convert each face in the train set to an embedding
    newTrainX = list()
    for face_pixels in x_train:
        embedding = get_embedding(facenet_model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)
    # convert each face in the test set to an embedding
    newTestX = list()
    for face_pixels in x_test:
        embedding = get_embedding(facenet_model, face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)
    print(newTestX.shape)
    # save arrays to one file in compressed format
    savez_compressed(save_path, newTrainX, y_train, newTestX,
                     y_test)


def load_embedded():
    data = load(save_path)
    return data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
