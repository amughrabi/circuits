import keras
import math
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image


# Dimensions of our images.
img_width, img_height = 56, 56

nb_train_samples, nb_validation_samples = 6445, 4174

train_data_dir = '/home/amughrabi/datasets/circuits/data/train'
validation_data_dir = '/home/amughrabi/datasets/circuits/data/validation'

epochs = 100
batch_size = 16

def create_model():
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 1)

    model = Sequential()
    # 64 3x3 kernels
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    # Reduce by taking the max of each 2x2 block
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    # Reduce by taking the max of each 2x2 block
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    # Reduce by taking the max of each 2x2 block
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout to avoid overfitting
    model.add(Dropout(0.25))
    # Flatten the results to one dimension for passing into our final layer 2D -> 1D
    model.add(Flatten())
    # A hidden layer to learn with
    model.add(Dense(128, activation='relu'))
    # Another dropout
    model.add(Dropout(0.5))
    # Final categorization from 0-9 with softmax
    model.add(Dense(13, activation='softmax'))
    return model


def load_trained_model(weights_path):
   model = create_model()
   model.load_weights(weights_path)
   return model

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')


model = load_trained_model('circuits-model-weights.h5')

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Was it worth the wait?
score = model.evaluate_generator(validation_generator, nb_validation_samples // batch_size,  verbose=0, workers=12)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


import numpy as np

predictions = model.predict_generator(validation_generator, nb_validation_samples // batch_size, workers=12)
predictions = np.argmax(predictions, axis=1)

labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

print(labels)
print(predictions)

predictions = [labels[k] for k in predictions]

def process_image(loc):
    #
    # Checking the input_shape
    #
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 1)
    #
    # We need to:
    #      - load the image
    #      - resize it to be matched w/ the input_shape, we used lanczos for resizing interpolation
    #      - convert the colorspace into 'gray'
    #
    img = image.load_img(loc,
                         target_size=input_shape,
                         color_mode="grayscale", interpolation='lanczos')
    #
    # Converting image to array
    #
    img_prc = image.img_to_array(img)
    #
    # Normalizing the image will be useful for model.predict(..) and predict_proba(..) for providing some probabilities
    # for the output instead of Zero and ones.
    #
    img_prc = np.array(img_prc).astype(np.float) / 255
    #
    # Expand dimensions
    #
    img_prc = np.expand_dims(img_prc, axis=0)

    return (img, img_prc)

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

(img, img_prc) = process_image(loc='/home/amughrabi/Symbol G8_Protective Earth.JPG')

result = model.predict_proba(img_prc, verbose=0)

import matplotlib.pyplot as plt

plt.title(r'The symbol: $\bf{%s}$ ($\bf{%s}$)' % (labels[np.argmax(result[0])], truncate(max(result[0]) * 100, 3)) )
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()



print(result[0][np.argmax(result[0])])
print(result)

import os
from keras.preprocessing.image import img_to_array, load_img

def generate_plot_pics(datagen,orig_img):
    dir_augmented_data = "data/preview"
    try:
        ## if the preview folder does not exist, create
        os.mkdir(dir_augmented_data)
    except:
        ## if the preview folder exists, then remove
        ## the contents (pictures) in the folder
        for item in os.listdir(dir_augmented_data):
            os.remove(dir_augmented_data + "/" + item)

    ## convert the original image to array
    x = img_to_array(orig_img)
    ## reshape (Sampke, Nrow, Ncol, 3) 3 = R, G or B
    x = x.reshape((1,) + x.shape)
    ## -------------------------- ##
    ## randomly generate pictures
    ## -------------------------- ##
    i = 0
    Nplot = 11
    for batch in datagen.flow(x,batch_size=1,
                          save_to_dir=dir_augmented_data,
                          save_prefix="pic",
                          save_format='jpeg'):
        i += 1
        if i > Nplot - 1: ## generate 8 pictures
            break

    ## -------------------------- ##
    ##   plot the generated data
    ## -------------------------- ##
    fig = plt.figure(figsize=(11, 6))
    fig.subplots_adjust(hspace=0.02,wspace=0.01,
                    left=0,right=1,bottom=0, top=1)

    ## original picture
    ax = fig.add_subplot(3, 4, 1,xticks=[],yticks=[])
    ax.imshow(orig_img, cmap=plt.get_cmap('gray'))
    ax.set_title("original")

    i = 2
    for imgnm in os.listdir(dir_augmented_data):
        ax = fig.add_subplot(3, 4, i,xticks=[],yticks=[])
        img = load_img(dir_augmented_data + "/" + imgnm)
        ax.imshow(img)
        i += 1
    plt.show()

generate_plot_pics(datagen=train_datagen, orig_img=img)