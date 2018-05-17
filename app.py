import click

import kiwi.dataset as dataset
import kiwi.dataloader as dataloader

MY_MODELS = ['model1', 'model2', 'model3']

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
  # click.echo('Training model %s on folder %s ' % (m, d))
  X_data, Y_data = dataloader.from_folder(d)
  print("Loaded %s images by %s datapoints each" % X_data.shape)

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
  click.echo('testing model %s on folder %s ' % (m, d))


cli = click.CommandCollection(sources=[data, training, testing])
if __name__ == '__main__':
    cli()