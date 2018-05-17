from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from kiwi.utils import calc_accuracy
import os

class LogitSk():
  SAVE_PATH = os.path.abspath('models/model1/saved/logitsk.pkl')

  def __init__(self, x, y):
    self.x_data = x
    self.y_data = y
    self.log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
  
  def train(self):
    print('Training LogitSk model...')
    self.log_reg.fit(self.x_data, self.y_data)
    self._save()
    calc_accuracy(self.log_reg, self.x_data, self.y_data)

  def _save(self):
    joblib.dump(self.log_reg, self.SAVE_PATH)
    print("Model persisted at", self.SAVE_PATH)
  
  @staticmethod
  def load_model():
    return joblib.load(LogitSk.SAVE_PATH)
  