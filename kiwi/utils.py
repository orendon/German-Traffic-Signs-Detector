import sys
from sklearn.metrics import accuracy_score

def download_progress(blocknum, blocksize, totalsize):
  """ Display download progress bar in the command line.
      Reference: https://stackoverflow.com/questions/13881092/download-progressbar-for-python-3 """

  readsofar = blocknum * blocksize
  if totalsize > 0:
    percent = readsofar * 1e2 / totalsize
    s = "\r%5.1f%% %*d / %d" % (
      percent, len(str(totalsize)), readsofar, totalsize)
    sys.stderr.write(s)
    if readsofar >= totalsize: # near the end
      sys.stderr.write("\n")
  else: # total size is unknown
    sys.stderr.write("read %d\n" % (readsofar,))

def calc_accuracy(model, x, y):
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred) * 100
    print('Model accuracy {}%'.format(round(accuracy, 2)))

CLASS_LABELS = {
  "0": "speed limit 20 (prohibitory)",
  "1": "speed limit 30 (prohibitory)",
  "2": "speed limit 50 (prohibitory)",
  "3": "speed limit 60 (prohibitory)",
  "4": "speed limit 70 (prohibitory)",
  "5": "speed limit 80 (prohibitory)",
  "6": "restriction ends 80 (other)",
  "7": "speed limit 100 (prohibitory)",
  "8": "speed limit 120 (prohibitory)",
  "9": "no overtaking (prohibitory)",
  "10": "no overtaking (trucks) (prohibitory)",
  "11": "priority at next intersection (danger)",
  "12": "priority road (other)",
  "13": "give way (other)",
  "14": "stop (other)",
  "15": "no traffic both ways (prohibitory)",
  "16": "no trucks (prohibitory)",
  "17": "no entry (other)",
  "18": "danger (danger)",
  "19": "bend left (danger)",
  "20": "bend right (danger)",
  "21": "bend (danger)",
  "22": "uneven road (danger)",
  "23": "slippery road (danger)",
  "24": "road narrows (danger)",
  "25": "construction (danger)",
  "26": "traffic signal (danger)",
  "27": "pedestrian crossing (danger)",
  "28": "school crossing (danger)",
  "29": "cycles crossing (danger)",
  "30": "snow (danger)",
  "31": "animals (danger)",
  "32": "restriction ends (other)",
  "33": "go right (mandatory)",
  "34": "go left (mandatory)",
  "35": "go straight (mandatory)",
  "36": "go right or straight (mandatory)",
  "37": "go left or straight (mandatory)",
  "38": "keep right (mandatory)",
  "39": "keep left (mandatory)",
  "40": "roundabout (mandatory)",
  "41": "restriction ends (overtaking) (other)",
  "42": "restriction ends (overtaking (trucks)) (other)"
}