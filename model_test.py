import model
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
import unittest
class MyModelTest(unittest.TestCase):
  
  def _test_train_sample(self):
    d = model.MyDriver()
    (X_train, y_train) = d.load_train_samples("../data/driving_log.csv")

    assert(X_train[0] == "../data/IMAGE/center_2016_12_01_13_30_48_287.jpg")
    assert(X_train[1] == "../data/IMAGE/left_2016_12_01_13_30_48_287.jpg")
    assert(X_train[2] == "../data/IMAGE/right_2016_12_01_13_30_48_287.jpg")

    assert(y_train[0] == 0.0)
    assert(y_train[1] == 0.2)
    assert(y_train[2] == -0.2)
    return 

  def _test_train(self):
    d = model.MyDriver()
    (X_train, y_train)= d.load_example_training_data(examples=10, width=8,height=8) # //d.load_traing_data("../data/driving_log.csv"))
    model = d.simple_network(8, 8)
    model.fit( X_train, y_train, validation_split=0.0, shuffle=True, nb_epoch=100, verbose=1)
    if 0:
      model.save('test_train.md5')
      model = keras.models.load_model('test_train.md5')
    y = model.predict( X_train )
    y_index = np.argmax( y , 1 )
    print(y_index)

    for i in range(0,10):
      assert ( y_index[i] == i )
    return 

  def _test_train_generator(self):
    d = model.MyDriver()
    (X_train, y_train) = d.load_example_training_data(examples=10, width=8,height=8) # //d.load_traing_data("../data/driving_log.csv"))
    model = d.simple_network(8, 8)
    history_object = model.fit_generator(
                        d.train_generator_example( zip(X_train, y_train) , 10 ),
                        steps_per_epoch = 10,
                        epochs=10, verbose=1) 

    if 0:
      model.save('test_train.md5')
      model = keras.models.load_model('test_train.md5')
    y = model.predict( X_train )
    y_index = np.argmax( y , 1 )
    print(y_index)
    for i in range(0,10):
      assert ( y_index[i] == i )
    return 

 
  def test_augment(self):
    d = model.MyDriver()
    (X_train, y_train) = d.load_train_samples("../data/driving_log.csv")

    assert(X_train[0] == "../data/IMG/center_2016_12_01_13_30_48_287.jpg")
    assert(X_train[1] == "../data/IMG/left_2016_12_01_13_30_48_287.jpg")
    assert(X_train[2] == "../data/IMG/right_2016_12_01_13_30_48_287.jpg")
    assert( os.path.isfile(X_train[0]) )
    assert( os.path.isfile(X_train[1]) )
    assert( os.path.isfile(X_train[2]) )

    image = d.load_image(X_train[0])

    plt.figure(1)
    
    frow = 3
    fcol = 3
    index = 1
    plt.subplot(fcol,frow,index); index+=1
    steer = 0.0
    plt.imshow(image)
    plt.title("org")

    plt.subplot(fcol,frow,index) ; index+=1
    image = d.crop(image, 50, 20)
    plt.imshow(image)
    plt.title("crop")

    plt.subplot(fcol,frow,index) ; index+=1
    (image, steer) = d.augment_flip( (image, steer), 1)
    plt.imshow(image)
    plt.title("flip")

    plt.subplot(fcol,frow,index) ; index+=1
    (image, steer) = d.augment_translate( (image, steer), 1.0, 1.0)
    plt.imshow(image)
    plt.title("trans")

  #  plt.subplot(fcol,frow,index) ; index+=1
  #  (image, steer) = d.augment_rotate_and_scale( (image, steer), 1, 1.0)
  #  plt.imshow(image)
  #  plt.title("rotate_and_scale")


    plt.subplot(fcol,frow,index) ; index+=1
    image = d.resize(image,100,30)
    plt.imshow(image)
    plt.title("resize")

    plt.subplot(fcol,frow,index) ; index+=1
    (image, steer) = d.augment_bright( (image, steer), 1.0)
    plt.imshow(image)
    plt.title("augment_bright")

    plt.subplot(fcol,frow,index) ; index+=1
    (image, steer) = d.augment_bright( (image, steer), 0.0)
    plt.imshow(image)
    plt.title("augment_bright")

    plt.show()

    return

  def test_random_augment(self):
    d = model.MyDriver()
    (X_train, y_train) = d.load_train_samples("../data/driving_log.csv")

    assert(X_train[0] == "../data/IMG/center_2016_12_01_13_30_48_287.jpg")
    assert(X_train[1] == "../data/IMG/left_2016_12_01_13_30_48_287.jpg")
    assert(X_train[2] == "../data/IMG/right_2016_12_01_13_30_48_287.jpg")
    assert( os.path.isfile(X_train[0]) )
    assert( os.path.isfile(X_train[1]) )
    assert( os.path.isfile(X_train[2]) )
    plt.figure(2)
    
    frow = 4
    fcol = 4
    index = 1
    plt.subplot(fcol,frow,index); index+=1
    org_image = d.load_image(X_train[1])
    
    steer = 0.0
    plt.imshow(org_image)
    plt.title("org")
    
    for i in range(0,frow*fcol-1):
      plt.subplot(fcol,frow,index) ; index+=1
      print(org_image)
      # 130, 163, 208
      (image, steer) = d.augment_and_preprocess( (org_image, steer))

      plt.imshow(np.copy(image))
 
    plt.show()

    return

  def test_train_generator_call(self):
    d = model.MyDriver()
    plt.figure(3)
    
    frow = 4
    fcol = 4
    index = 1
    
    gen = d.train_generator( "../data/driving_log.csv", 1)
    for i in range(0,frow*fcol-1):
      plt.subplot(fcol,frow,index) ; index+=1
      (image, steer) = gen.next()
      plt.imshow(np.copy(image[0]))
      plt.title("steer=%f"%steer)
    plt.show()

    return

  
if __name__ == "__main__":
    unittest.main()

