from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import load_img, img_to_array
from keras import applications
from keras import optimizers
import numpy as np

"""
# resize
img_width, img_height = 224, 224
train_data_dir = 'dataset_tumoroid'
nb_train_samples = 9*12  # toplam örnek sayısı: 9 sınıf, her sınıf için 12 örnek
epochs = 10
batch_size = 16

input_shape = (img_width, img_height, 3)

# model
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

for layer in base_model.layers:
    layer.trainable = False

# train
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs)

# model save
model.save("tumoroid.h5")
"""

# training output
"""
Found 108 images belonging to 9 classes.
Epoch 1/10
6/6 [==============================] - 22s 4s/step - loss: 2.4526 - accuracy: 0.2065
Epoch 2/10
6/6 [==============================] - 23s 4s/step - loss: 1.4719 - accuracy: 0.4457
Epoch 3/10
6/6 [==============================] - 22s 4s/step - loss: 1.0360 - accuracy: 0.6413
Epoch 4/10
6/6 [==============================] - 23s 4s/step - loss: 0.6051 - accuracy: 0.8478
Epoch 5/10
6/6 [==============================] - 24s 4s/step - loss: 0.4921 - accuracy: 0.8804
Epoch 6/10
6/6 [==============================] - 22s 4s/step - loss: 0.3033 - accuracy: 0.9674
Epoch 7/10
6/6 [==============================] - 22s 4s/step - loss: 0.2470 - accuracy: 0.9891
Epoch 8/10
6/6 [==============================] - 22s 4s/step - loss: 0.2254 - accuracy: 0.9674
Epoch 9/10
6/6 [==============================] - 22s 4s/step - loss: 0.1549 - accuracy: 1.0000
Epoch 10/10
6/6 [==============================] - 22s 4s/step - loss: 0.1613 - accuracy: 1.0000
"""

# load model
model=load_model("tumoroid.h5")

img = load_img('0.jpg', target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255

# make prediction
prediction = model.predict(x)
predicted_class = (np.argmax(prediction))+1

print(predicted_class)

