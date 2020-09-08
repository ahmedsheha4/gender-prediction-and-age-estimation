# gender-prediction-and-gender-estimation
a deep multi-class CNNs architecture to estimate age and predict gender , using TensorFlow2 (Keras API) , numpy , matplotlib , OpenCV and MTCNN for face detection\
\
trained on IMDB-WIKI dataset\
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ \
\
Age estimation is the (apparent age + the label of the age with the highest probability ) // 2\
where apparent age is the probability of each label * age label\
age labels are from 0 to 100\
\
Model architcture : https://raw.githubusercontent.com/ahmedsheha4/gender-prediction-and-age-estimation/master/Model%20Architecture.png \
\
sample outputs : https://drive.google.com/drive/folders/1gVdAL6vuopVnhmlPl1PjgHoYjUBeOcHA?usp=sharing
