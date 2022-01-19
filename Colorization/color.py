import rtoml
from pathlib import Path
from loguru import logger
from color.config import Config
from color.train import train as _train
import click

config: Config = None


@click.group()
@click.option('-c',
              '--config_file',
              type=click.Path(exists=True,
                              dir_okay=False,
                              readable=True),
              default=Path('config.toml'))
def cli(config_file):
    logger.info(f'Using {config_file} config')
    global config
    config = Config.parse_obj(rtoml.load(config_file))


@cli.command()
def train():
    _train(config)
    logger.success("Training complete")


if __name__ == '__main__':
    cli()
