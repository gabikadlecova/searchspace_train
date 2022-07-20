import click
import pandas as pd


@click.command()
@click.argument('dirs', nargs=-1)
@click.argument('out', nargs=1)
def run(dirs, out):
    """
    Read .csv files from DIRS and concatenate them to one output file OUT. Keep only unique hashes.
    """
    dataset = [pd.read_csv(d, index_col=0) for d in dirs]
    dataset = pd.concat(dataset)
    dataset = dataset[~dataset.index.duplicated(keep='first')]
    dataset.to_csv(out)


if __name__ == "__main__":  # pragma: no cover
    run()
