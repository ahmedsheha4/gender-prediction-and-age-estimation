import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import scipy.io
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Reshape,Flatten,Activation, Dropout,BatchNormalization,Convolution2D,ZeroPadding2D,MaxPooling2D,Input
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from sklearn.utils import shuffle
from PIL import Image
from mtcnn.mtcnn import MTCNN


desired_width = 350
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 15)
num_classes_age = 101
num_classes_gender = 1



def convertToArray(image_path ):
    img = load_img("Dataset/imdb_crop1/imdb_crop/" + image_path[0], grayscale=False, target_size=(224, 224))
    pixels_array = np.asarray(img,dtype=np.float32).reshape(1, -1)[0]
    print("*** Converting ***")
    return pixels_array


def load_data():
    dataset_row = scipy.io.loadmat('Dataset/imdb_crop1/imdb_crop/imdb.mat')
    Sample = dataset_row['imdb'][0][0][0]
    Samples_Length = Sample.shape[1]

    feature_names = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score","celeb_names","celeb_id"]

    df = pd.DataFrame(index= range(0,Samples_Length), columns = feature_names)
    print(Samples_Length)
    for k , i in enumerate(dataset_row):
        print(i)
        if k == 3:
            dataset = dataset_row[i][0][0]
            for index ,j  in enumerate((dataset)):
                df[feature_names[index]]=pd.DataFrame(data=j[0])
    return df


def DataProcessing():
    df = load_data()

    males_count = df[df["gender"] == 1.0].index
    females_count = df[df["gender"] == 0.0].index

    df['date_of_birth'] = df['dob'].apply(lambda dob: datetime.fromordinal(int(dob)) + timedelta(days=dob%1))
    df['year'] = df['date_of_birth'].map(lambda x:x.year)
    df['age'] = df['photo_taken'] - df['year']
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop(df[df['face_score'].isnull()].index)
    df = df.drop(df[df["second_face_score"] > 0].index)
    df = df[df['face_score'] > 3]
    df = df.drop(df[df['gender'].isnull()].index)
    df = df[(df['age'] > 0) & (df['age'] <= 100)]


    boollist = []
    if len(females_count) >= len(males_count):
        boollist.append(True)
    else:
        boollist.append(False)
    boollist.append(True)

    df=df.sort_values(by=['gender','face_score'],ascending=boollist)
    df = df.reset_index(drop=True)
    df = df.iloc[abs(len(females_count) - len(males_count) ):]
    df=shuffle(df)
    df = df.drop(columns=['name', 'face_score', 'second_face_score', 'date_of_birth', 'face_location','photo_taken','year','dob', "celeb_names","celeb_id"])
    print(df.groupby(df.gender).count())


    df['pixels'] = df['full_path'].apply(convertToArray)
    df = df.drop(columns=['full_path'])
    return df

def extract_features_and_labels():
    df = DataProcessing()

    age_labels = df['age'].values
    gender_labels = df['gender'].values
    age_labels = tf.keras.utils.to_categorical(age_labels, num_classes=num_classes_age)

    features = []
    print(df.shape)

    for i in range(0, df.shape[0]):
        features.append(df['pixels'].values[i])
    features = np.array(features , dtype=np.float32)
    features = features.reshape(features.shape[0], 224, 224, 3)
    print(features.shape)
    return features , (age_labels,gender_labels)


def AgeModel(input):
    x= Convolution2D(32, (3, 3), padding="same")(input)
    x= Activation("relu")(x)
    x= MaxPooling2D(pool_size=(3, 3))(x)
    x= Convolution2D(64, (3, 3), padding="same")(x)
    x= Activation("relu")(x)
    x= Convolution2D(64, (3, 3), padding="same")(x)
    x= Activation("relu")(x)
    x= MaxPooling2D(pool_size=(2, 2))(x)
    x= Convolution2D(128, (3, 3), padding="same")(x)
    x= Activation("relu")(x)
    x= Convolution2D(128, (3, 3), padding="same")(x)
    x= Activation("relu")(x)
    x= MaxPooling2D(pool_size=(2, 2))(x)
    x= Flatten()(x)
    x= Dense(256)(x)
    x= Activation("relu")(x)
    x= Dense(num_classes_age)(x)
    x= Activation('softmax' , name="AGE")(x)
    return x

#***********************************************************************


def GenderModel(input):

    y= Convolution2D(32, (3, 3), padding="same")(input)
    y= Activation("relu")(y)
    y= MaxPooling2D(pool_size=(3, 3))(y)
    y= Convolution2D(64, (3, 3), padding="same")(y)
    y= Activation("relu")(y)
    y= Convolution2D(64, (3, 3), padding="same")(y)
    y= Activation("relu")(y)
    y= MaxPooling2D(pool_size=(2, 2))(y)
    y= Convolution2D(128, (3, 3), padding="same")(y)
    y= Activation("relu")(y)
    y= Convolution2D(128, (3, 3), padding="same")(y)
    y= Activation("relu")(y)
    y= MaxPooling2D(pool_size=(2, 2))(y)
    y= Dense(256)(y)
    y= Activation("relu")(y)
    y= Flatten()(y)
    y= Dense(num_classes_gender)(y)
    y= Activation('sigmoid' , name="GENDER")(y)

    return y


def BuildModel():

    inputShape = (224, 224, 3)
    input = Input(shape=inputShape)
    x=AgeModel(input)
    y=GenderModel(input)

    model = tf.keras.Model(
        inputs=input,
        outputs=[x,y],
        name='AgeGenderEstimation'
    )
    model.summary()

    plot_model(model, to_file='shared_input_layer.png')

    losses = {
        "AGE": "categorical_crossentropy",
        "GENDER": "binary_crossentropy",
    }

    lossWeights = {"AGE": 1.0, "GENDER": 1.0}
    model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss_weights=lossWeights, metrics=['accuracy'] )

    epochs = 1
    batch_size = 8
    features , labels =extract_features_and_labels()
    model.fit(features, {"AGE": labels[0], "GENDER": labels[1]},
              epochs=epochs, batch_size=batch_size , validation_split = 0.1)

    model_json = model.to_json()
    with open("model4545.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model4545.h5")
    print("Saved model to disk")
    return model

BuildModel()
