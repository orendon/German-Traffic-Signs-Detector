import click

import kiwi.dataset as dataset
import kiwi.dataloader as dataloader
import kiwi.inferer as inferer
from kiwi.logitsk import LogitSk
from kiwi.logittf import LogitTf
from kiwi.lenet5 import LeNet5

MODEL_MAPPING = {
  "model1": LogitSk,
  "model2": LogitTf,
  "model3": LeNet5
}
MY_MODELS = list(MODEL_MAPPING.keys())

# download command
##################
@click.group()
def data():
  pass

@data.command()
def download():
  dataset.process()

# train command
###############
@click.group()
def training():
  pass

@training.command()
@click.option('-m', required=True, type=click.Choice(MY_MODELS))
@click.option('-d', required=True)
def train(m, d):
  """Usage: train -m MODEL -d path/to/folder """
  lenet = MODEL_MAPPING[m] == LeNet5
  X_data, Y_data = dataloader.from_folder(d, lenet)
  print("Loaded %s images by %s datapoints each" % (len(X_data), X_data.shape))
  
  model = MODEL_MAPPING[m]()
  model.train(X_data, Y_data)

# test command
##############
@click.group()
def testing():
  pass

@testing.command()
@click.option('-m', required=True, type=click.Choice(MY_MODELS))
@click.option('-d', required=True)
def test(m, d):
  """Usage: test -m MODEL -d path/to/folder """
  lenet = MODEL_MAPPING[m] == LeNet5
  X_data, Y_data = dataloader.from_folder(d, lenet)
  print("Loaded %s images by %s datapoints each" % (len(X_data), X_data.shape))

  model = MODEL_MAPPING[m].load_model()
  model.calc_accuracy(X_data, Y_data)

# infer command
##############
@click.group()
def infering():
  pass

@infering.command()
@click.option('-m', required=True, type=click.Choice(MY_MODELS))
@click.option('-d', required=True)
def infer(m, d):
  """Usage: infer -m MODEL -d path/to/folder """
  lenet = MODEL_MAPPING[m] == LeNet5
  X_data, files = dataloader.from_infer_folder(d, lenet)

  model = MODEL_MAPPING[m].load_model()
  predictions = model.predict(X_data)

  inferer.display_inferences(files, predictions)

cli = click.CommandCollection(sources=[data, training, testing, infering])
if __name__ == '__main__':
    cli()