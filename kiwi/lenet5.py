from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import os

class LeNet5():
  SAVE_PATH = os.path.abspath('models/model3/saved/lenet5')
  NUM_EPOCHS = 200
  BATCH_SIZE = 128
  CLASSES_SIZE = 43
  LEARNING_RATE = 0.001 #1e-4
  
  def __init__(self):
    pass

  def _build(self):
    mu = 0
    sigma = 0.1
    layer_units = {
        'C1' : 6,
        'C3' : 16,
        'C5' : 120,
        'F6' : 84,
        'F7' : self.CLASSES_SIZE, # originally 10, changed to match Traffic Signs classes
    }

    tf.reset_default_graph()
    self.session = tf.Session()

    self.x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    self.y = tf.placeholder(tf.int32, (None))
    self.one_hot_y = tf.one_hot(self.y, self.CLASSES_SIZE)

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    C1_w = tf.Variable(tf.truncated_normal(shape=[5,5,1,6], mean=mu, stddev=sigma))
    C1_b = tf.Variable(tf.zeros(layer_units['C1']))
    C1 = tf.nn.conv2d(self.x, C1_w, strides=[1,1,1,1], padding='VALID') + C1_b 
    C1 = tf.nn.relu(C1)
    
    # Layer 2: Pooling. Input = 28x28x6. Output = 14x14x6.
    S2 = tf.nn.max_pool(C1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    # Layer 3: Convolutional. Output = 10x10x16.
    C3_w = tf.Variable(tf.truncated_normal(shape=[5,5,6,16], mean=mu, stddev=sigma))
    C3_b = tf.Variable(tf.zeros(layer_units['C3']))
    C3 = tf.nn.conv2d(S2, C3_w, strides=[1,1,1,1], padding='VALID') + C3_b
    C3 = tf.nn.relu(C3)
    
    # Layer 4: Pooling. Input = 10x10x16. Output = 5x5x16.
    S4 = tf.nn.max_pool(C3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    #Flatten. Input = 5x5x16. Output = 400.
    C5 = flatten(S4)
    
    # Layer 5: Convolutional/Fully Connected. Input = 400. Output = 120.
    # Feature map is 1x1 because S4 is also 5x5 (practically fully-connected unless LenNet grows)
    C5_w = tf.Variable(tf.truncated_normal(shape=(400,120), mean=mu, stddev=sigma))
    C5_b = tf.Variable(tf.zeros(layer_units['C5']))
    C5 = tf.matmul(C5, C5_w) + C5_b
    C5 = tf.nn.relu(C5)
    
    # Layer 6: Fully Connected. Input = 120. Output = 84.
    F6_w = tf.Variable(tf.truncated_normal(shape=(120,84), mean=mu, stddev=sigma))
    F6_b = tf.Variable(tf.zeros(layer_units['F6']))
    F6 = tf.matmul(C5, F6_w) + F6_b
    F6 = tf.nn.relu(F6)
    
    # Layer 7: Fully Connected. Input = 84. Output = 43 (Originally 10).
    F7_w = tf.Variable(tf.truncated_normal(shape=(84,self.CLASSES_SIZE), mean=mu , stddev=sigma))
    F7_b = tf.Variable(tf.zeros(layer_units['F7']))
    self.logits = tf.matmul(F6, F7_w) + F7_b

    self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_y)
    self.loss_operation = tf.reduce_mean(self.cross_entropy)
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
    self.training_operation = self.optimizer.minimize(self.loss_operation)

    self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
    self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    self.saver = tf.train.Saver()

  def train(self, X_data, Y_data):
    print('Training Logistic Regression (tensorflow) model...')

    self._build()

    with self.session.as_default() as sess:
      sess.run(tf.global_variables_initializer())
      num_samples = len(X_data)

      print("Training model...")
      for epoch in range(self.NUM_EPOCHS):
        X_train, y_train = shuffle(X_data, Y_data)
        for offset in range(0, num_samples, self.BATCH_SIZE):
          end = offset + self.BATCH_SIZE
          batch_x, batch_y = X_data[offset:end], Y_data[offset:end]
          sess.run(self.training_operation, feed_dict={self.x: batch_x, self.y: batch_y})

        if (epoch+1) % 20 == 0:
            accuracy_val = self.calc_accuracy(X_data, Y_data, False)
            print("Epoch {}, Accuracy = {:.2f}%".format(epoch+1, accuracy_val*100))

      # Model persistence
      self._save()

  def predict(self, data):
    with self.session.as_default() as sess:
      prediction = tf.argmax(self.logits, 1)
      return prediction.eval(feed_dict={self.x: data}, session=sess)

  def _save(self):
    checkpoint_path = self.saver.save(tf.get_default_session(), self.SAVE_PATH)
    print("Model persisted at", checkpoint_path)
  
  def calc_accuracy(self, X_data, Y_data, showprint=True):
    with self.session.as_default() as sess:
      num_examples = len(X_data)
      total_accuracy = 0
      sess = tf.get_default_session()
      for offset in range(0, num_examples, self.BATCH_SIZE):
          batch_x, batch_y = X_data[offset:offset+self.BATCH_SIZE], Y_data[offset:offset+self.BATCH_SIZE]
          accuracy = sess.run(self.accuracy_operation, feed_dict={self.x: batch_x, self.y: batch_y})
          total_accuracy += (accuracy * len(batch_x))

      accuracy_val = total_accuracy / num_examples
      print("Model accuracy: {:.2f}%".format(accuracy_val*100)) if showprint else None
      return accuracy_val

  def restore_model(self):
    self._build()
    
    with self.session.as_default() as sess:
      sess.run(tf.global_variables_initializer())
      self.saver.restore(sess, self.SAVE_PATH)
  
  @staticmethod
  def load_model():
    model = LeNet5()
    model.restore_model()

    return model
