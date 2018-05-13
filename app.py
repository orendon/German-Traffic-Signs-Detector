import click

# download command
@click.group()
def dataset():
  pass

@dataset.command()
def download():
  click.echo('Downloading German dataset...')

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

cli = click.CommandCollection(sources=[dataset, training, testing])
if __name__ == '__main__':
    cli()