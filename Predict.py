import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from PIL import Image
from mtcnn.mtcnn import MTCNN
import cv2

num_classes_age = 101
num_classes_gender = 1
def load_model(modelname):

    json_file = open("Trained Models/"+modelname + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("Trained Models/"+modelname +".h5")
    return loaded_model


def extract_face(filename, required_size=(224, 224)):
	pixels = plt.imread(filename)
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = np.asarray(image)

	return face_array

def predict(pic_path):

    pixels = extract_face(pic_path)
    test_img = np.expand_dims(pixels, axis=0)
    plt.imshow(pixels)
    plt.show()
    loaded_model=load_model('model')
    ageprediction , genderprediction = loaded_model.predict(test_img)
    y_pos = np.arange(101)
    plt.bar(y_pos, ageprediction[0])
    plt.ylabel('predictions')
    plt.title('age')
    plt.show()
    output_indexes = np.array([i for i in range(0, num_classes_age)])
    apparent_age = np.round(np.sum(ageprediction * output_indexes, axis = 1))
    age = ( int(apparent_age[0] + np.argmax(ageprediction)))//2
    print("Age: ", age)
    if genderprediction[0][0]>=0.5:
            print("Male")
            gender = "Male"
    else :
        print("Female")
        gender = "Female"
    output = gender + " , " + "Aged: " + str(age)
    pixels = cv2.resize(pixels, (500,500), interpolation = cv2.INTER_AREA)
    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(pixels,output ,(5,20), font, 1,(255,255,255),1)
    cv2.imshow("img",pixels)


predict('Test/testImage.jpg')
