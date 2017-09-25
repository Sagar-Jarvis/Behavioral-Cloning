import numpy as np
import keras
import matplotlib.pyplot as plt
import csv
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam

folder = './data_1/data/'
path = folder + 'driving_log.csv'

#resized image dimension in training
rows = 16
cols = 16

#batch size and epoch
batch_size=128
nb_epoch=9

# reading log files
log_tokens = []
with open(path,'rt') as f:
    reader = csv.reader(f)
    for line in reader:
        log_tokens.append(line)
log_labels = log_tokens.pop(0)

def image_preprocessing(img):
    """Training data in S channel of the HSV color space and resizing it to 16X16"""
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(cols, rows))
    return resized


def data_loading(imgs,steering,folder,correction=0.08):
# Loading log tokens    
    log_tokens = []
    with open(path,'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            log_tokens.append(line)
    log_labels = log_tokens.pop(0)

# Using for loop for loading and appending centre images with steering anles
    for i in range(len(log_tokens)):
        img_path = log_tokens[i][0]
        img_path = folder+'IMG'+(img_path.split('IMG')[1]).strip()
        img = plt.imread(img_path)
        imgs.append(image_preprocessing(img))
        steering.append(float(log_tokens[i][3]))
# Using for loop for loading and appending left images with steering anles and adding a little correction
    for i in range(len(log_tokens)):
        img_path = log_tokens[i][1]
        img_path = folder+'IMG'+(img_path.split('IMG')[1]).strip()
        img = plt.imread(img_path)
        imgs.append(image_preprocessing(img))
        steering.append(float(log_tokens[i][3]) + correction)
# Using for loop for loading and appending right images with steering anles and subtracting a little correction
    for i in range(len(log_tokens)):
        img_path = log_tokens[i][2]
        img_path = folder+'IMG'+(img_path.split('IMG')[1]).strip()
        img = plt.imread(img_path)
        imgs.append(image_preprocessing(img))
        steering.append(float(log_tokens[i][3]) - correction)


def main():
    # Loading data    
    images_train = np.array(data['Images']).astype('float32')
    steering_train = np.array(data['Steering']).astype('float32')
    
    # flipping using similar to np.fliplr() function but even faster and appending
    images_train = np.append(images_train, images_train[:,:,::-1],axis=0)
    steering_train = np.append(steering_train, -steering_train,axis=0)
    
    # shuffling the data for avoiding overfit
    X_train, y_train = shuffle(images_train, steering_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.1)
    
    # reshaping the shape of the images to the ones we want to input here it is 16X16
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1)
    X_val = X_val.reshape(X_val.shape[0], rows, cols, 1)
    
    # model building with lambda normalization, conv2D, Maxpool, dropout and elu activation
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(rows,cols,1)))
    model.add(Conv2D(2, 3, 3, border_mode='valid', input_shape=(rows,cols,1), activation='elu'))
    model.add(MaxPooling2D((4,4),(4,4),'valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))

    model.summary()
    # loss-mse and optimizer-adam
    model.compile(loss='mean_squared_error',optimizer='adam')
    history = model.fit(X_train, y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_val, y_val))
    
    #saving the model
    model.save("model21.h5")
    print("Model saved.")
    
    

if __name__ == '__main__':
    
    data={}
    data['Images'] = []
    data['Steering'] = []

    data_loading(data['Images'], data['Steering'],folder,0.3)
    main()