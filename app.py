import click

import kiwi.dataset as dataset
import kiwi.dataloader as dataloader
from kiwi.logitsk import LogitSk
from kiwi.utils import calc_accuracy

MODEL_MAPPING = {
  "model1": LogitSk,
  "model2": None,
  "model3": None
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
  """Usage: train -m MODEL -d FOLDER """
  X_data, Y_data = dataloader.from_folder(d)
  print("Loaded %s images by %s datapoints each" % X_data.shape)
  
  model = MODEL_MAPPING[m](X_data, Y_data)
  model.train()

# test command
##############
@click.group()
def testing():
  pass

@testing.command()
@click.option('-m', required=True, type=click.Choice(MY_MODELS))
@click.option('-d', required=True)
def test(m, d):
  """Usage: test -m MODEL -d FOLDER """
  X_data, Y_data = dataloader.from_folder(d)
  print("Loaded %s images by %s datapoints each" % X_data.shape)
  
  model = MODEL_MAPPING[m].load_model()
  calc_accuracy(model, X_data, Y_data)


cli = click.CommandCollection(sources=[data, training, testing])
if __name__ == '__main__':
    cli()