import pandas as pd
from pathlib import Path
import os
from datetime import datetime
from argparse import ArgumentParser


def parse_args():
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
    df = pd.read_csv(args.data, parse_dates=[args.timestamp])

    default_timestamp = 'Timestamp'
    df = df.rename(columns={args.timestamp: default_timestamp})

    df.sort_values(by=default_timestamp, inplace=True)
    df = df.resample(args.resample, on=default_timestamp).median()
    df = df.reset_index()
    df.dropna(inplace=True)
    df['Index'] = df.index

    name = args.name
    if name is None:
        time = datetime.now().strftime('%H%M%S')
        name = f'{time}-{args.resample}'

    path = os.path.join('out/datasets', f'{name}.csv')
    df.to_csv(path, index=False)


if __name__ == '__main__':
    main(parse_args())
