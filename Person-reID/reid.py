import rtoml
from pathlib import Path
from loguru import logger
from reid.config import Config
from notifiers.logging import NotificationHandler
from reid.prepare import prepare as _prepare
from reid.train import train as _train
from reid.test import test as _test
import click

config: Config


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
    params = {
        "token": config.token,
        "chat_id": config.chat_id
    }
    handler = NotificationHandler("telegram", defaults=params)
    logger.add(handler, level="ERROR")
    logger.add(handler, level="SUCCESS")


@cli.command()
def prepare():
    _prepare(config.data_path)
    logger.success("Preparation complete")


@cli.command()
def train():
    _train(config)
    logger.success("Training complete")


@cli.command()
def test():
    _test(config)
    logger.success("Testing complete")


if __name__ == '__main__':
    cli()
