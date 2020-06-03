
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#improt CIFAR10 data sample
from keras.datasets import cifar10
(X_train,y_train),(X_test,y_test)=cifar10.load_data()


X_train.shape

# check random image
i=2000
plt.imshow(X_train[i])
print(y_train[i])


#draw a grid of images
W_grid = 15
L_grid=15

fig,axes= plt.subplots(L_grid,W_grid,figsize=(25,25))
axes=axes.ravel()
n_training=len(X_train)
#draw a random grid
for i in np.arange(0,L_grid*W_grid):
    index = np.random.randint(0,n_training)
    axes[i].imshow(X_train[index])
    axes[i].axis('off')
    axes[i].set_title(y_train[index])
plt.subplots_adjust(hspace=0.1)    



# change image type to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

n_cat=10


import keras
# create categorical binary values for labels
y_train = keras.utils.to_categorical(y_train,n_cat)
y_test  = keras.utils.to_categorical(y_test,n_cat)


#scale to maximum 1
X_train=X_train/255
X_test=X_test/255



Input_shape=X_train.shape[1:]


# Train the model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard



#create the cnn network
cnn = Sequential()
cnn.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu',input_shape=Input_shape))
cnn.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu'))
cnn.add(AveragePooling2D(2,2))
cnn.add(Dropout(0.3))

cnn.add(Conv2D(filters=128, kernel_size=(3,3),activation='relu'))
cnn.add(Conv2D(filters=128, kernel_size=(3,3),activation='relu'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Dropout(0.5))

cnn.add(Flatten())
cnn.add(Dense(units=1024 ,activation='relu'))
cnn.add(Dense(units=1024 ,activation='relu'))
cnn.add(Dense(units=10 , activation='softmax'))




#add optimiser
cnn.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.rmsprop(lr=0.001), metrics=['accuracy'])




history = cnn.fit(X_train,y_train,batch_size=32,epochs=200,shuffle=True)
evaluation = cnn.evaluate(X_test,y_test)
print('Accuracy:{}'.format(evaluation[1]))



predicted_classes = cnn.predict_classes(X_test)
y_test = y_test.argmax(1)



L = 7
W = 7

fig ,axes =plt.subplots(L,W,figsize=(12,12))
axes= axes.ravel()
for i in np.arange(0,L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Pred={}\n True ={}'.format(predicted_classes[i],y_test[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=1)    



# plotting the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True)



# Saving the model
import os
directory =os.path.join(os.getcwd(),'saved_models')
if not os.path.isdir(directory):
    os.makedirs(directory)
model_path = os.path.join(directory, 'keras_cifar10_trained_model.h5')
cnn.save(model_path)



#image augmentation test
from keras.preprocessing.image import ImageDataGenerator
datagen= ImageDataGenerator(rotation_range=90,
                               width_shift_range=0.1,
                               horizontal_flip=True,
                               vertical_flip=True
                              )


data_train.fit(X_train)
cnn.fit_generator(datagen.flow(X_train, y_train, batch_size=32),epochs=200)



y_test  = keras.utils.to_categorical(y_test,n_cat)
score=cnn.evaluate(X_test,y_test)
print('Test accuracy',score[1])




# save the model augmented
directory =os.path.join(os.getcwd(),'saved_models')
if not os.path.isdir(directory):
    os.makedirs(directory)
model_path = os.path.join(directory, 'keras_cifar10_trained_model_augmented.h5')
cnn.save(model_path)

