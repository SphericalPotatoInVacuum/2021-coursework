from pathlib import Path
import click


@click.command()
@click.argument('path', type=click.Path(exists=True, file_okay=False, readable=True))
def rename(path: str):
    paths = list(Path(path).iterdir())
    m = max(len(x.stem) for x in paths)
    for path in paths:
        path.rename(Path(path.parent, f'{path.stem:0>{m}}{path.suffix}'))


if __name__ == '__main__':
    rename()
