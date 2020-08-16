import os, cv2, random
import numpy as np
import pandas as pd
#pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import ticker
import seaborn as sns
#matplotlib inline 

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
#view rawGender Classification(Importing Packages)

# loading labels for each image from csv
labels = pd.read_csv('results.csv')
labels = labels.iloc[:,0:2]
labels.head()

#normal = 1      ---32
#abnormal = 0   ---32

# Separating abnormal labels
abnormal_data = labels[labels['state'] == 0]
abnormal_data.head()

# Splitting abnormal data into train and test
test_abnormal_data = abnormal_data.iloc[-4:,:]
train_abnormal_data = abnormal_data.iloc[:-4,:]


# Separating normal labels
normal_data = labels[labels['state'] == 1]
normal_data.head()

# Splitting normal data into train and test
test_normal_data = normal_data.iloc[-4:,:]
train_normal_data = normal_data.iloc[:-4,:]
#rint(train_normal_data)
#print(test_normal_data)

#just to diaplay the image
##img=mpimg.imread('images/2.jpg')
##imgplot = plt.imshow(img)
##plt.show()


# total test data combining the normal and abnormal
test_indices = test_normal_data.index.tolist() + test_abnormal_data.index.tolist()
test_data = labels.iloc[test_indices,:]
test_data.head()


# total train data (Filtering train_data from labels by dropping test_data)
train_data = pd.concat([labels, test_data, test_data]).drop_duplicates(keep=False)
train_data.head()


# checking count of normal and abnormal data
#sns.countplot(labels['state'])


# train and test with image name along with paths
path = 'images/' # path of your image folder
train_image_name = [path+each for each in train_data['Filename'].values.tolist()]
test_image_name = [path+each for each in test_data['Filename'].values.tolist()]
#print(train_image_name)
#print(test_image_name)

# preparing data by processing images using opencv
ROWS = 64
COLS = 64
CHANNELS = 3

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        #print(image_file)
        image = read_image(image_file)
        data[i] = image.T
        if i%5 == 0:
            print('Processed {} of {}'.format(i, count))
    
    return data

train = prep_data(train_image_name)
test = prep_data(test_image_name)

##img = cv2.imread('/images/1.jpg', cv2.IMREAD_COLOR)
## 
##print('Original Dimensions : ',img.shape)
##
##
##scale_percent = 60 # percent of original size
##width = int(img.shape[1] * scale_percent / 100)
##height = int(img.shape[0] * scale_percent / 100)
##dim = (width, height)
### resize image
##resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
## 
##print('Resized Dimensions : ',resized.shape)
##
##cv2.imshow("Resized image", resized)

# plotting normal and abnormal side by side just to check
def show_normal_and_abnormal():
    normal = read_image(train_image_name[0])
    abnormal = read_image(train_image_name[2])
    pair = np.concatenate((normal, abnormal), axis=1)
    plt.figure(figsize=(10,5))
    plt.imshow(pair)
    plt.show()
    
#show_normal_and_abnormal()



# splitting path of all images into abnormal and normal
train_abnormal_image = []
train_normal_image = []
for each in train_image_name:
    if each.split('/')[1] in train_normal_data['Filename'].values:
        train_normal_image.append(each)
    else:
        train_abnormal_image.append(each)

#print(train_normal_image)
#print(train_abnormal_image)

optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'


def normal_abnormal():
    
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS), activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))



    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model


model = normal_abnormal()

model.summary()


nb_epoch = 23
batch_size = 10
labs = train_data.iloc[:,1].values.tolist()

## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

#early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')  
history = LossHistory()


#training model
model.fit(train, labs, batch_size=batch_size, epochs=nb_epoch,
              validation_split=0.25, verbose=0, shuffle=True, callbacks=[history])

#predicting output
predictions = model.predict(test, verbose=0)
predictions

#Plotting training and validating loss
loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()
plt.show()


for i in range(0,8):
    if predictions[i] >= 0.5: 
        print('{} Normal'.format(predictions[i][0]))
    else: 
        print('{} abnormal'.format(predictions[i][0]))
        
    #plt.imshow(test[i].T)
    plt.show()

#check for indivudual data
##img = cv2.imread('/images/1.jpg', cv2.IMREAD_COLOR)
##prdct = mode.predict(img, verbose=0)
##print(prdct)
