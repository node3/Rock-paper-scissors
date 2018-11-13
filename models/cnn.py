from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

NONE=0
ROCK=1
PAPER=2
SCISSOR=3

def getLabel(filename):
    if 'rock' in filename:
        label = ROCK
    elif 'paper' in filename:
        label = PAPER
    elif 'scissor' in filename:
        label = SCISSOR
    else:
        label = NONE
    return label

def getImageList(imageSourceFolder):
    onlyfiles = [ f for f in os.listdir(imageSourceFolder) if os.path.isfile(os.path.join(imageSourceFolder,f)) ]
    images = np.empty(len(onlyfiles), dtype=object)
    labels = np.empty(len(onlyfiles), dtype=int)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread( os.path.join(imageSourceFolder, onlyfiles[n]) )
        labels[n] = getLabel(onlyfiles[n])
    return images, labels


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()


batch_size = 32
num_classes = 4
epochs = 10
img_dst='../img_dest'

# image dimensions
img_rows, img_cols = 300, 300

# load the images data set
X,y = getImageList(img_dst)

# split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='relu', input_shape=(1, img_rows, img_cols)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test),callbacks=[history])
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()