import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

# SETTINGS

NUM_TRAIN_SAMPLES = 8005
NUM_VALIDATION_SAMPLES = 2023
IMG_SIZE = 256

EPOCHS = 75
BATCH_SIZE = 16
TRAINING_STEPS = int(NUM_TRAIN_SAMPLES / BATCH_SIZE)
VALIDATION_STEPS = int(NUM_VALIDATION_SAMPLES / BATCH_SIZE)

LEARNING_RATE = 1e-5

TRAIN_DATA_DIR = 'data/train'
VALIDATION_DATA_DIR = 'data/validation'

print(os.listdir(TRAIN_DATA_DIR))


# MODEL DEFINITION
model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), padding='same', activation='relu', data_format='channels_last',
                        input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(layers.BatchNormalization(axis=3))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(8, (5, 5), padding='same', activation='relu', data_format='channels_last'))
model.add(layers.BatchNormalization(axis=3))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(16, (5, 5), padding='same', activation='relu', data_format='channels_last'))
model.add(layers.BatchNormalization(axis=3))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(16, (5, 5), padding='same', activation='relu', data_format='channels_last'))
model.add(layers.BatchNormalization(axis=3))
model.add(layers.MaxPooling2D(pool_size=(4, 4)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='linear', kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros'))

model.summary()

# TRAINING SETUP

run_model = model

run_model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"]
)

train_datagen = ImageDataGenerator(
    rescale = 1.0/255.0,
    samplewise_center=False,
    samplewise_std_normalization=True,
    data_format='channels_last',
)

validation_datagen = ImageDataGenerator(
    rescale = 1.0/255.0,
    samplewise_center=False,
    samplewise_std_normalization=True,
    data_format='channels_last',
)

train_generator = train_datagen.flow_from_directory(
    shuffle = True,
    directory=TRAIN_DATA_DIR,
    classes=['cats', 'dogs'],
    target_size=(IMG_SIZE, IMG_SIZE),
   batch_size=BATCH_SIZE,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    shuffle = True,
    directory = VALIDATION_DATA_DIR,
    classes=['cats', 'dogs'],
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')

 

checkpoint = ModelCheckpoint('CNN_weights.h5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)



H = run_model.fit_generator(
    generator=train_generator,
    steps_per_epoch=TRAINING_STEPS,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks = [checkpoint]
)


model_json = model.to_json()
with open("CNN.json", "w") as json_file:
    json_file.write(model_json)


