import rtoml
from pathlib import Path
from loguru import logger
from cli import train_
import click
from configs import spv as cfg


@click.group()
def cli():
    pass


@cli.command()
def train():
    train_()
    logger.success("Training complete")


if __name__ == '__main__':
    cli()
