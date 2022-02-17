import rtoml
from pathlib import Path
from loguru import logger
from color.config import Config
from color.train import train as _train
from color.colorize import colorize as _colorize
import click

config: Config = None


@click.group()
@click.option('-c',
              '--config_file',
              type=click.Path(exists=True, dir_okay=False, readable=True),
              default=Path('config.toml'))
def cli(config_file):
    logger.info(f'Using {config_file} config')
    global config
    config = Config.parse_obj(rtoml.load(config_file))


@cli.command()
def train():
    _train(config)
    logger.success("Training complete")


@cli.command()
@click.argument('input_img', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('output_img', type=click.Path(dir_okay=False, writable=True))
def colorize(input_img, output_img):
    _colorize(input_img, output_img)


if __name__ == '__main__':
    cli()
