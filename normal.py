# To detect normal images (without rotation) 

from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


json_file = open('CNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("CNN_weights.h5")
print("Loaded model")



train_datagen = ImageDataGenerator(
    samplewise_center=False,
    samplewise_std_normalization=True,
    data_format='channels_last',
)


frame = cv2.imread("doggo.jpg")
frame = cv2.resize(frame, (256,256))
frame = np.reshape(frame,[1,256,256,3])
frame = frame.astype('float64')
frame = train_datagen.standardize(frame)
print(loaded_model.predict(frame))
