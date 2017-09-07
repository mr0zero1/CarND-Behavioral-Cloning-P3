#from my_model import MyDriver 
import keras
from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
import os
import numpy as np
import cv2
import csv



class MyDriver:
  UP_CROP = 50
  DOWN_CROP = 20
  IMAGE_WIDTH  = 200
  IMAGE_HEIGHT = 66
  NUM_CHANNELS = 3
  SEED = 66478
  BATCH_SIZE = 8
  NUM_EPOCHS = 30
  EVAL_FREQUENCY = 100
  IMAGE_PATH = "../data/IMG/"
  def __init__(self):
    pass

  def load_example_training_data(self, examples, width, height):
    STDEV = 64
    MU = 128
    X_train =  np.random.randn(examples, width, height, 3)*STDEV + MU
    values = range(0,examples)
    n_values = np.max(values) + 1
    y_train = np.eye(n_values)[values]
    return (X_train, y_train)

  @staticmethod
  def get_csv_lines_without_header(csv_file):
    lines = []
    with open(csv_file) as f:
       for i, line in enumerate(csv.reader(f)):
          if 0 == i : continue
          lines.append(line)
    return lines 

  @staticmethod
  def current_image_path(source_path):
    basename = source_path.split('/')[-1]
    current_path = MyDriver.IMAGE_PATH + basename
    return current_path

  @staticmethod
  def load_train_samples(csv_file,correction=0.2): 
#center,left,right,steering,throttle,brake,speed
#      0            1                   2        3  4  5  6
#IMG/center_.jpg, IMG/left_.jpg, IMG/right_.jpg, 0, 0, 0, 22.14829
    car_images = []
    steering_angles = []
    for i, line in enumerate(MyDriver.get_csv_lines_without_header(csv_file)):
        image_c = MyDriver.current_image_path(source_path=line[0])
        image_l = MyDriver.current_image_path(source_path=line[1])
        image_r = MyDriver.current_image_path(source_path=line[2])
        car_images.extend([image_c, image_l, image_r])
        steering_c = float(line[3])
        steering_l = steering_c + correction
        steering_r = steering_c - correction
        steering_angles.extend([steering_c, steering_l, steering_r])
    X_train = np.array(car_images)
    y_train = np.array(steering_angles)
    return X_train, y_train

  def simple_network(self, width, height):
    model = Sequential()
    model.add(Lambda(lambda x: x/255-0.5, input_shape=(height,width,3)))
    model.add(Conv2D(8,(3,3),activation="relu",strides=(1,1)))
    model.add(MaxPooling2D())
    model.add(Flatten()) # //1152
    model.add(Dropout(0.5)) #<--- how to???
    model.add(Dense(50))
    model.add(Dense(10))
    model.compile(loss='mse', optimizer='adam')
    return model

  def nvidia_cnn(self):
    input_shape = (MyDriver.IMAGE_HEIGHT, MyDriver.IMAGE_WIDTH, MyDriver.NUM_CHANNELS   )
    model = Sequential()
    model.add(Lambda(lambda x: x/255-0.5, input_shape=input_shape))
    model.add(Conv2D(24,(5,5),activation="relu",strides=(2,2)))
    model.add(Conv2D(36,(5,5),activation="relu",strides=(2,2)))
    model.add(Conv2D(48,(5,5),activation="relu",strides=(2,2)))
    model.add(Conv2D(64,(3,3),activation="relu",strides=(1,1)))
    model.add(Conv2D(64,(3,3),activation="relu",strides=(1,1)))
    model.add(Flatten()) # //1152
    model.add(Dropout(0.5)) #<--- how to???
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

  

  @staticmethod
  def train_generator(csv_filename,batch_size = 32):
    (X_train, y_train) = MyDriver.load_train_samples(csv_filename)
    samples = zip( X_train, y_train)
    num_samples = len(samples)
    batch_image = np.zeros((batch_size, MyDriver.IMAGE_HEIGHT, 
                      MyDriver.IMAGE_WIDTH, MyDriver.NUM_CHANNELS), dtype=np.uint8)
   # batch_image = np.zeros((batch_size, 160, 320,3 ),dtype=np.uint8)
    batch_steer = np.zeros((batch_size))
    while 1: # Loop forever so the generator never terminates
      for i in range(batch_size):
        random_index = np.random.randint(num_samples)
        #random_index = 1
        (filename,steer) = samples[random_index]
        org_image = MyDriver.load_image(filename)
        selected = 0
        image = org_image
        #(image, steer) = MyDriver.augment_and_preprocess((np.copy(org_image), steer))
        while 0 == selected: 
          (image, steer) = MyDriver.augment_and_preprocess((np.copy(org_image), steer))

          is_not_curve = abs(steer) < 0.1
          if is_not_curve:
            if np.random.random() > 0.5 :
              selected = 1
          else:
            selected = 1
        batch_image[i] = np.copy(image)
        batch_steer[i] = steer

      yield batch_image, batch_steer


  @staticmethod
  def train_generator_example(samples,batch_size = 32):
    width = 8 
    height = 8
    STDEV = 64
    MU = 128
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_samples = []
            y_samples = []
            for (X, y) in batch_samples:
                X_samples.append(X)
                y_samples.append(y)

            # trim image to only see section with road
            X_train = np.array(X_samples)
            y_train = np.array(y_samples)
            yield sklearn.utils.shuffle(X_train, y_train)
  
  @staticmethod
  def augment_and_preprocess(image_and_steer): 
    (image, steer) = image_and_steer
    image = MyDriver.crop(image, MyDriver.UP_CROP, MyDriver.DOWN_CROP)
    (image, steer) = MyDriver.augment_bright((image,steer), np.random.random())
    (image, steer) = MyDriver.augment_flip((image, steer), np.random.randint(2))

    (image, steer) = MyDriver.augment_translate( (image, steer), np.random.random(), np.random.random())

   # (image, steer) = MyDriver.augment_rotate_and_scale((image, steer), np.random.random(), np.random.random())

    #image = self.resize(image, 100, 32)
    image = MyDriver.resize(image, MyDriver.IMAGE_WIDTH, MyDriver.IMAGE_HEIGHT)
    return (image, steer)

  @staticmethod
  def augment_bright(image_and_steer, bright):
    assert( (0<=bright) & (bright<=1.0))
    (image, steer) = image_and_steer
    result = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    # WHY: float is used? => To check overflow!
    result = np.array(result, dtype=np.float64)
    V = 2
    result[:,:,V] = result[:,:,V] * ( 0.5 + bright )
    result[:,:,V][result[:,:,V]>255]  = 255
    result = np.array(result, dtype=np.uint8) 
    result = cv2.cvtColor(result,cv2.COLOR_HSV2RGB)
    return (result, steer)

  @staticmethod
  def augment_flip(image_and_steer, is_flip):
    H_FLIP = 1
    (image,steer) = image_and_steer
    if is_flip :
       image = cv2.flip(image,H_FLIP)
       steer = -steer
    return (image,steer)
  
  @staticmethod
  def augment_translate(image_and_steer, tx, ty): 
    #print(tx,ty)
    MAX_TX = 100
    MAX_TY = 40
    tx = (tx-0.5)*MAX_TX
    ty = (ty-0.5)*MAX_TY
    (image,steer) = image_and_steer
    # Translation
    image = MyDriver.translate_image(image, tx, ty)
    ANGLES_PER_PIXEL = 0.02
    steer = steer + tx * ANGLES_PER_PIXEL
    return (image,steer)


  @staticmethod
  def augment_rotate_and_scale(image_and_steer, rotate, scale):
    MAX_ROTATE = 10.0
    MAX_SCALE  = 0.0  # 30%
    (image,steer) = image_and_steer
    rows,cols,ch = image.shape
    rotate= 0.0 + (rotate-0.5)*MAX_ROTATE
    scale = 1.0 + (scale -0.5)*MAX_SCALE
    cx = cols/2
    cy = rows/2 
    # Translation
    (image, tx, ty, scale, rotate) = MyDriver.rotate_and_scale_image(image, cx, cy, rotate, scale)
    ANGLES_PER_PIXEL = 0.01
    steer = steer + tx * ANGLES_PER_PIXEL
    return image,steer

  @staticmethod
  def translate_image( image, tx, ty):
    rows,cols,ch = image.shape
    M = np.float32([
                    [1,0,tx],
                    [0,1,ty]
                   ])
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst

  @staticmethod
  def rotate_and_scale_image(image, tx, ty, rotate=0.0, scale=1.0):
    rows,cols,ch = image.shape
    #rotate= 0   + (np.random.random()*2-1)*rotate
    #scale = 1.0 + (np.random.random()*2-1)*scale
    M = cv2.getRotationMatrix2D((tx, ty),rotate,scale)
    dst = cv2.warpAffine(image,M,(cols,rows))
    return (dst, tx, ty, scale, rotate)

  @staticmethod
  def crop(image, crop_upper, crop_lower):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    dst = image[crop_upper:shape[0]-crop_lower, 0:shape[1]]
    return dst

  # 320:90 ---> 100:30
  # nick: 400x166
  @staticmethod
  def resize( image, target_xs, target_ys) :
    #print(image.shape)
    dst = cv2.resize(image,  (target_xs, target_ys), interpolation=cv2.INTER_AREA)
    return dst

  @staticmethod
  def load_image(image_filaname):
    assert( os.path.isfile(image_filaname) )
    image = cv2.imread(image_filaname) 
    #print(image.shape)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return np.copy(image)


if __name__ == "__main__":
  keras_model = MyDriver().nvidia_cnn()
  history_object = keras_model.fit_generator(
                      MyDriver().train_generator("../data/driving_log.csv", 256),
                      steps_per_epoch = 100,
                      epochs=10, verbose=1) 

  keras_model.save('train.md5')


