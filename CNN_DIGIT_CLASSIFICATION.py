#!/usr/bin/env python
# coding: utf-8

# # CNN DIGIT CLASSIFICATION

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


# 1. Visualization the first 50 images
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

fig, axes = plt.subplots(5, 10, figsize=(12,7), subplot_kw={'xticks':[], 'yticks':[]})

for i, ax in enumerate(axes.flat):
  ax.imshow(x_train[i], cmap=plt.cm.gray_r)
  ax.text(0.45, 1.05, str(y_train[i]), transform=ax.transAxes)
plt.show()


# In[ ]:


# //we can also try on fashion_mnist dataset


# In[ ]:


# 2. Preprocess the data
x_train = x_train.astype("float32") /255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# In[ ]:


# 3. Define the CNN model
model = keras.models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])
model.summary()


# In[ ]:


# 4. Train the model (without displaying epochs)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0) # Set verbosee=0


# In[ ]:


# 5. Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0) # Set verbose=0
print('Test accuracy:', accuracy)


# In[ ]:


# //If error will show then run all the code in a single cell


# In[ ]:


# 6. Display a test image and make a prediction
test_image = x_test[99]
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                right=False, labelbottom=False, labelleft=False)

plt.imshow(test_image.reshape(28,28), cmap=plt.cm.gray_r)
plt.show()


# In[ ]:


# 7
predicted_class = np.argmax(model.predict(test_image.reshape(1, 28, 28, 1)), axis=1)
print('Looks like a ' + str(predicted_class) + '!')


# In[ ]:




