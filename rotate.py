#Rotate filters and feature map of trained CNN model


from numpy.lib.function_base import rot90
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import numpy
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


json_file = open('CNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
mod = loaded_model
loaded_model.load_weights("CNN_weights.h5")
print("Loaded model")

model = loaded_model


for layer in model.layers:
    if 'conv' not in layer.name:
        continue
    filters,bias = layer.get_weights()


model1 = Sequential()    #Feature extraction network
for i in range (12):
	if 'conv' in model.layers[i].name:
		filters,bias = mod.layers[i].get_weights()
		filters = numpy.rot90(filters)  # Rotate filters by 180 degrees
		filters = numpy.rot90(filters)
		model.layers[i].set_weights([filters,bias])
	model1.add(model.layers[i])



model2 = Sequential()   # Fully connected network
model2.add(Input(shape = (1,8,8,16)))
for i in range (12,17):
	model2.add(model.layers[i])


train_datagen = ImageDataGenerator(
    samplewise_center=False,
    samplewise_std_normalization=True,
    data_format='channels_last',
)


frame = cv2.imread('doggo.jpg')  # Location of rotated image
frame = cv2.resize(frame, (256,256))
frame = np.reshape(frame, [1,256,256,3])
frame = frame.astype('float64')
frame = train_datagen.standardize(frame)
feature_maps = model1.predict(frame)
print(feature_maps.shape)
feature_maps= np.rot90(feature_maps,axes=(1,2))  # Rotate feature map by 180 degrees 
feature_maps= np.rot90(feature_maps,axes=(1,2))
print(model2.predict(feature_maps))
