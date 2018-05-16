import urllib.request as req
import os
from zipfile import ZipFile
import shutil
import re
import math
from kiwi.utils import download_progress

TEST_SPLIT = 0.2

DATASET_URL = "http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip"
DATASET_ROOT = "FullIJCNN2013"
# DATASET_URL = "file:/Users/orendon/dev/ml/German-Traffic-Signs-Detector/images/FullIJCNN2013.zip"
# DATASET_ROOT = "FullIJCNN2013"

ROOT_PATH = os.path.abspath('.')
TRAIN_PATH = "/images/train"
TEST_PATH = "/images/test"

def process():
  """ Download the dataset and proceed spliting into train/test folders """

  print("Downloading the GTSDB dataset, grab some coffee...")
  filename, response = req.urlretrieve(DATASET_URL, reporthook=download_progress)

  print("Dataset downloaded at", filename)
  unzip(filename)

def unzip(filename):
  """ Unzip the dataset and move files to the corresponding train/test folder """

  with ZipFile(filename, 'r') as zip_ref:
    classes = [f for f in zip_ref.namelist() if re.match(r"\A{}/\d+/\Z".format(DATASET_ROOT), f)]
    for class_id in classes:
      class_files = [file for file in zip_ref.namelist() 
                          if re.match(r"\A{}\d+.ppm\Z".format(class_id), file)]

      print("\nClass folder {} contains {} files".format(class_id, len(class_files)))
      train_files, test_files = split(class_files)

      train_folder = class_id.replace(DATASET_ROOT, ROOT_PATH + TRAIN_PATH)
      copy(zip_ref, train_files, train_folder)

      test_folder = class_id.replace(DATASET_ROOT, ROOT_PATH + TEST_PATH)
      copy(zip_ref, test_files, test_folder)

def split(class_files):
  """ Split a given list of files according to the TEST_SPLIT constant """
  files_count = len(class_files)

  split_count = math.ceil(TEST_SPLIT * files_count)
  test_files = class_files[:split_count]
  train_files = class_files[split_count:]

  return train_files, test_files

def copy(zip_ref, files, target_folder):
  """ Copy a bunch of files to an specific folder """

  print("Copying %s files to %s" % (len(files), target_folder))
  if not os.path.exists(target_folder):
    os.makedirs(target_folder)
  
  for file in files:
    filename = os.path.basename(file)
    source = zip_ref.open(file)
    target = open(target_folder + filename, 'wb')
    with source, target:
      shutil.copyfileobj(source, target)
