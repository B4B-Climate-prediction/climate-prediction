"""
A convert file that converts the existing dataset to a dataset that the other script can use.

Command-arguments:
    -d [--DATA]: Data filepath, must be a .csv file.
    -t [--TIMESTAMP]: The column which contains the timestamps
    -r [--RESAMPLE]: Resample the duration. Default: Min For all options see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
    -n [--NAME]: Name of the exported file
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser


def parse_args():
    """
    Parse the arguments from the command using argparse

    :return: argument-parser
    """

    parser = ArgumentParser(add_help=True)

    parser.add_argument(
        '-d', '--data',
        required=True,
        type=str,
        help='Data file path, must be a .csv file.'
    )

    parser.add_argument(
        '-t', '--timestamp',
        required=True,
        type=str,
        help='Feature column which included the timestamp.'
    )

    parser.add_argument(
        '-r', '--resample',
        default='min',
        type=str,
        help='Resample duration. Default: min. For all options see: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html'
    )

    parser.add_argument(
        '-n', '--name',
        type=str,
        help='Name of exported file'
    )

    return parser.parse_args()


def main(args):
    """
    The main method that runs whenever the file is being used.

    :param args: the arguments in the command

    It resamples the timestamps and generates indexes.

    :return: dataset as csv file.
    """
    df = pd.read_csv(args.data, parse_dates=['Timestamp'])
    
    df['Timestamp'] = df[args.timestamp]
    del df[args.timestamp]

    df.sort_values(by='Timestamp', inplace=True)
    df = df.resample(args.resample, on='Timestamp').median()
    df = df.reset_index()
    df.dropna(inplace=True)
    df['Index'] = df.index

    name = args.name
    if name is None:
        time = datetime.now().strftime('%H%M%S')
        name = f'{time}-{args.resample}'
    
    path = str(Path(__file__).parent / 'out' / 'datasets' / f'{name}.csv')
    df.to_csv(path, index=False)


if __name__ == '__main__':
    main(parse_args())
