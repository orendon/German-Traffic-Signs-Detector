import click

import kiwi.dataset as dataset

# download command
@click.group()
def data():
  pass

@data.command()
def download():
  dataset.process()

# train command
@click.group()
def training():
  pass

@training.command()
@click.option('-m')
@click.option('-d')
def train(m, d):
  """Usage: train -m MODEL -d FOLDER """
  click.echo('training model %s on folder %s ' % (m, d))

# test command
@click.group()
def testing():
  pass

@testing.command()
@click.option('-m')
@click.option('-d')
def test(m, d):
  """Usage: test -m MODEL -d FOLDER """
  click.echo('testing model %s on folder %s ' % (m, d))

cli = click.CommandCollection(sources=[data, training, testing])
if __name__ == '__main__':
    cli()