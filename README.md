# German Traffic Signs Detector

Solution for Kiwi/RutaN [Deep Learning Challenge](https://github.com/KiwiCampusChallenge/Kiwi-Campus-Challenge/blob/master/Deep-Learning-Challenge.md)

## Dependencies
More details on `requirements.txt` file, main packages are:

- Python 3.6.3
- numpy
- Click
- tensorflow
- scikit-learn
- scikit-image
- opencv-python
- jupyter
- scipy

## Implemented models

### model1 - Logistic Regression using Scikit-learn
More details on the report page.

### model2 - Logistic Regression using Tensorflow
More details on the report page.

### model3 - LeNet-5 using Tensorflow
Based on [Yann Lecun paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), more details on the report page.

## CLI usage

Download the traffic signs dataset:
```shell
python app.py download
```

Train `model1` specifying the training images folder:
```shell
python app.py train -m model1 -d images/train
```

Test `model1` specifying the test images folder:
```shell
python app.py test -m model1 -d images/test
```

Run inferences on `model1` and specifying a custom set of images folder:
```shell
python app.py infer -m model1 -d images/test
```