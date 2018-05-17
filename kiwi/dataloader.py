import numpy as np
# import skimage
import cv2
import os
import imghdr

ROOT_PATH = os.path.abspath('.')

def prepare_image(fullpath):
  """ Resize the image, convert it to gray scale and flatten the array """
  # image = skimage.data.imread(fullpath)
  # gray = skimage.color.rb2gray(image)
  image = cv2.imread(fullpath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return cv2.resize(gray, (28, 28)).flatten()

def from_folder(target_folder):
  """ Transform the data into an input/output format suitable for model training
  Images are loaded as arrays """

  data_dir = os.path.join(target_folder)
  classes = [c for c in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, c))]

  print("Grabbing files for %s classes..." % len(classes))
  X_data = []
  Y_data = []
  for class_id in classes:
    class_folder = os.path.join(data_dir, class_id)
    for file in os.listdir(class_folder):
      if os.path.basename(os.path.join(class_folder, file)): # images only
        image_path = os.path.join(ROOT_PATH, data_dir, class_id, file)
        X_data.append(prepare_image(image_path))
        Y_data.append(int(class_id))

  return np.array(X_data), np.array(Y_data)

def from_infer_folder(target_folder):
  """ Transform the data into a format suitable for model prediction """
  X_data = []
  files = []
  data_dir = os.path.join(target_folder)
  print("Loading images from", data_dir)
  for file in os.listdir(data_dir):
    image_path = os.path.join(ROOT_PATH, data_dir, file)
    if os.path.basename(image_path) and imghdr.what(image_path):
      X_data.append(prepare_image(image_path))
      files.append(image_path)
  
  print("Found %s file(s)" % len(X_data))
  return np.array(X_data), files