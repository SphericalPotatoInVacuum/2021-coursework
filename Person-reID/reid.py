import rtoml
from pathlib import Path
from loguru import logger
from reid.config import Config
from reid.notifier import Notifier
from reid.prepare import prepare as _prepare
from reid.train import train as _train
from reid.test import test as _test
import click

config: Config
notifier: Notifier


@click.group()
@click.option('-c',
              '--config_file',
              type=click.Path(exists=True,
                              dir_okay=False,
                              readable=True),
              default=Path('config.toml'))
def cli(config_file):
    logger.info(f'Using {config_file} config')
    global config, notifier
    config = Config.parse_obj(rtoml.load(config_file))
    notifier = Notifier(config.token, config.chat_id)


@cli.command()
def prepare():
    _prepare(config.data_path)


@cli.command()
def train():
    _train(config)


@cli.command()
def test():
    _test(config)


if __name__ == '__main__':
    cli()
