
# coding: utf-8

# In[1]:


# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="9yH8N4PWtcN83X4oJCk7Gp8cG",
                        project_name="general", workspace="ambarish-moharil")

# Run your code and go to https://www.comet.ml/


# In[2]:


# ARAI FINAL PROJECT
#COMPONENT FALIURE PREDICTION
#@Project_Christopher

import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[3]:


clf = Sequential()


# In[4]:


clf.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation = "relu"))
clf.add(MaxPooling2D(pool_size = (2,2)))

clf.add(Convolution2D(64,3,3,  activation = "relu"))
clf.add(MaxPooling2D(pool_size = (2,2)))

clf.add(Convolution2D(128,3,3,  activation = "relu"))
clf.add(MaxPooling2D(pool_size = (2,2)))

clf.add(Convolution2D(256,3,3,  activation = "relu"))
clf.add(MaxPooling2D(pool_size = (2,2)))


# In[5]:


clf.add(Flatten())


# In[6]:


from keras.preprocessing.image import ImageDataGenerator

#def AHE(img):
#    img = train_datagen.flow_from_directory('/home/ambarish/Documents/Monkey Dataset/training')
 #   img_adapteq = skimage.exposure.equalize_adapthist(img, clip_limit=0.03)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2, 
        horizontal_flip=30,
        featurewise_center=False,
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2  # randomly shift images vertically (fraction of total height)
                                )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory("/home/ambarish/Documents/SIH_DATASET/NEW_FOR_FATIGUE/Training",
                                                 target_size=(64, 64),
                                                 #input_shape=(64,64,1),
                                                 batch_size=5,
                                                 class_mode='categorical')
        

test_set = test_datagen.flow_from_directory("/home/ambarish/Documents/SIH_DATASET/NEW_FOR_FATIGUE/Validation",
                                            target_size=(64, 64),
                                            batch_size=5,
                                            class_mode='categorical')


# In[7]:


clf.add(Dense(output_dim = 128 , activation = "relu"))
clf.add(Dense(output_dim = 128 , activation = "relu"))
#clf.add(Dense(output_dim = 128 , activation = "relu"))
#clf.add(Dense(output_dim = 128 , activation = "relu"))
#clf.add(Dense(output_dim = 128 , activation = "relu"))
#clf.add(Dense(output_dim = 128 , activation = "relu"))

clf.add(Dense(output_dim = 3, activation = "softmax"))


# In[8]:


from keras.callbacks import EarlyStopping, ModelCheckpoint


callbacks = [EarlyStopping(monitor='val_acc', patience= 100, restore_best_weights = True),ModelCheckpoint(filepath='best_model.h5', monitor='val_acc', save_best_only=True)]


# In[9]:


clf.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# In[10]:


clf.fit_generator(training_set,
                         samples_per_epoch = 128,
                         verbose=1,
                          callbacks= callbacks, 
                         nb_epoch = 500,
                         validation_data = test_set,
                         nb_val_samples =41 )


# In[159]:


import numpy as np
from keras.preprocessing import image
text = "/home/ambarish/Desktop/test/401.jpg"
test_image = image.load_img(text, target_size=(64, 64))
#test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = clf.predict_proba(test_image)
#training_set.class_indices


# In[160]:


if result[0][0] == 1:
    prediction = 'brittle'
elif result[0][1] == 1:
    prediction = 'ductile'
elif result[0][2] == 1:
    prediction = 'fatigue'
else:
    prediction = 'none'


# In[161]:


print(prediction)


# In[131]:


print(clf.predict_proba(test_image))

