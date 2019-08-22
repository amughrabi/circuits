import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD

print('loading modules...')
#
# If the dataset does not exist, download it.
#
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()
#
# We need to explicitly convert the data into the format Keras / TensorFlow
# expects. We divide the image data by 255 in order to normalize it into
# 0-1 range, after converting it into floating point values.
# Sample is 28x28 image, Keras is understanding it as 1D image (Raster),
# converting it will be 28x28 = 784
#
train_images = mnist_train_images.reshape(60000, 784)
test_images = mnist_test_images.reshape(10000, 784)

#
# Normalizing the input data to BW = 1/0
#
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images /= 255
test_images /= 255
#
# Now we'll convert the 0-9 labels into "one-hot" format, as we did for
# TensorFlow.
#
train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)
#
# Let's take a peek at one of the training images just to make sure it looks OK:
#
import matplotlib.pyplot as plt


def display_sample(num):
    # Print the one-hot array of this sample's label
    print(train_labels[num])
    # Print the label converted back to a number
    label = train_labels[num].argmax(axis=0)
    # Reshape the 768 values to a 28x28 image
    image = train_images[num].reshape([28, 28])
    plt.title('Sample: %d Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


display_sample(1234)

#
# Here's where things get exciting. All that code we wrote in Tensorflow creating
# placeholders, variables, and defining a bunch of linear algebra for each layer
# in our neural network? None of that is necessary with Keras!
#
# We can set up the same layers like this. The input layer of 784 features feeds
# into a ReLU layer of 512 nodes, which then goes into 10 nodes with softmax applied.
# Couldn't be simpler:
#
model = Sequential()
# Add layers
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#
# We can even get a nice description of the resulting model:
#
model.summary()
#
# Setting up our optimizer and loss function is just as simple. We will use
# the RMSProp optimizer here. Other choices include Adagrad, SGD, Adam, Adamax, and Nadam.
# See https://keras.io/optimizers/
#
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#
# Training our model is also just one line of code with Keras. Here we'll do 10 epochs
# with a batch size of 100. Keras is slower, and if we're not running on top of a
# GPU-accelerated Tensorflow this can take a fair amount of time (that's why
# I've limited it to just 10 epochs.)
#
history = model.fit(train_images, train_labels,
                    batch_size=100,
                    epochs=10,
                    verbose=2,
                    validation_data=(test_images, test_labels))
#
# But, even with just 10 epochs, we've outperformed our Tensorflow version considerably!
#
score = model.evaluate(test_images, test_labels, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
#
# As before let's visualize the ones it got wrong. As this model is much better, we'll have
# to search deeper to find mistakes to look at.
#
for x in range(1000):
    test_image = test_images[x, :].reshape(1, 784)
    predicted_cat = model.predict(test_image).argmax()
    label = test_labels[x].argmax()
    if predicted_cat != label:
        plt.title('Prediction: %d Label: %d' % (predicted_cat, label))
        plt.imshow(test_image.reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
        plt.show()
