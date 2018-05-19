from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import tensorflow as tf
import numpy as np
import os

class LogitTf():
  SAVE_PATH = os.path.abspath('models/model2/saved/logittf')
  LEARNING_RATE = 0.5 #1e-4
  NUM_EPOCHS = 1000
  FEATURES_SIZE = 784 # 28x28 images
  
  def __init__(self):
    self.classes = np.array(range(0, 43))

  def _build(self):
    features = self.FEATURES_SIZE
    classes_length = self.classes.shape[0]

    tf.reset_default_graph()
    self.session = tf.Session()#.as_default()

    self.x = tf.placeholder(tf.float32, [None, features])
    self.y_ = tf.placeholder(tf.int32, [None, classes_length])

    self.W = tf.Variable(tf.zeros([features, classes_length]))
    self.b = tf.Variable(tf.zeros([classes_length]))
    self.y = tf.add(tf.matmul(self.x, self.W), self.b)

    self.cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y)
    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.LEARNING_RATE).minimize(self.cost)

    self.saver = tf.train.Saver()
  
  def encode_labels(self, Y_data):
    y_enc = LabelEncoder().fit_transform(self.classes)
    lblbin = LabelBinarizer().fit(y_enc)

    return lblbin.transform(Y_data)

  def train(self, X_data, Y_data):
    print('Training Logistic Regression (tensorflow) model...')

    self._build()
    Y_data_bin = self.encode_labels(Y_data)

    with self.session.as_default() as sess:
      sess.run(tf.global_variables_initializer())
      for epoch in range(self.NUM_EPOCHS):
          cost_in_each_epoch = 0

          _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: X_data, self.y_: Y_data_bin})
          cost_in_each_epoch += c

          if (epoch+1) % 200 == 0:
              print("Epoch {}:".format(epoch + 1), "cost={}".format(cost_in_each_epoch.max()))

      # Model persistence
      self._save()

      # Evaluation
      self.calc_accuracy(X_data, Y_data)
  
  def predict(self, data):
    with self.session.as_default() as sess:
      prediction = tf.argmax(self.y, 1)
      return prediction.eval(feed_dict={self.x: data}, session=sess)

  def _save(self):
    checkpoint_path = self.saver.save(tf.get_default_session(), self.SAVE_PATH)
    print("Model persisted at", checkpoint_path)
  
  def calc_accuracy(self, X_data, Y_data):
    Y_data_bin = self.encode_labels(Y_data)

    with self.session.as_default() as sess:
      correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      accuracy_percent = round(accuracy.eval({self.x: X_data, self.y_: Y_data_bin}) * 100, 2)
      print("Model accuracy: {}%".format(accuracy_percent))

  def restore_model(self):
    self._build()
    
    with self.session.as_default() as sess:
      sess.run(tf.global_variables_initializer())
      self.saver.restore(sess, self.SAVE_PATH)
  
  @staticmethod
  def load_model():
    model = LogitTf()
    model.restore_model()

    return model
