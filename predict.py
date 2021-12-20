import importlib
import inspect
import sys

import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

model_classes = []


def parse_args(models):
    """
    Parse the arguments from the command using argparse

    This method also takes in the arguments from the models being used by the command

    arguments:

    -d [--DATA]: Filepath to data source
    -t [--TARGETS]: columns of the data that need to be predicted
    -g [--GROUPS]: groups in datasets
    -kr [--KNREELS]: Known reels features in dataset (Also known in the future). Default: []



    """
    parser = ArgumentParser(add_help=True)

    parser.add_argument(
        '-d', '--data',
        required=True,
        type=str,
        help='Data file path, must be a .csv file in the out/datasets directory.'
    )

    parser.add_argument(
        '-t', '--targets',
        required=True,
        action='extend',
        nargs='+',
        help='Target feature(s) in dataset, specify minimal one'
    )

    parser.add_argument(
        '-g', '--groups',
        required=True,
        action='extend',
        nargs='+',
        help='Groups in dataset, specify minimal one'
    )

    parser.add_argument(
        '-kr', '--knreels',
        action='extend',
        nargs='*',
        default=[],
        help='Known reels features in dataset (also known in the future). Default: []'
    )

    parser.add_argument(
        '-kc', '--kncats',
        action='extend',
        nargs='*',
        default=[],
        help='Known categoricals features in dataset (also known in the future). Default: []'
    )

    parser.add_argument(
        '-ur', '--unreels',
        action='extend',
        nargs='*',
        default=[],
        help='Unknown reels features in dataset. Default: []'
    )

    parser.add_argument(
        '-uc', '--uncats',
        action='extend',
        nargs='*',
        default=[],
        help='Unknown categoricals features in dataset. Default: []'
    )

    parser.add_argument(
        '-m', '--model',
        required=True,
        type=str,
        help='Model file path, must be a (.?) file. If specified this model will be trained'
    )

    parser.add_argument(
        '-b', '--batch',
        default=128,
        type=int,
        help='Batch size during training. Default: 128'
    )

    parser.add_argument(
        '-ts', '--timesteps',
        default=10,
        type=int,
        help='The amount of time step into the future you predict. Default: 10'
    )

    parser.add_argument(
        '-tu', '--timeunit',
        default=['10', 'minutes'],
        type=[],
        required=True,
        help='Specify the timeunit difference between rows. Default: [10, minutes]'
    )

    for model in models:
        parser.add_argument(model.name)
        model.add_arguments(parser)

    return parser.parse_args()


def main(args, chosen_models):
    print(args)

    path = str(Path(__file__).parent / 'out' / 'datasets' / args.data)
    df = pd.read_csv(path)

    for model in chosen_models:
        model_class = model(df)

        dataset = model_class.generate_time_series_dataset(**vars(args))

        trained_model = model_class.load_model(**vars(args))

        predictions = model_class.predict(trained_model, dataset, **vars(args))

        print(predictions)


if __name__ == '__main__':
    models = []

    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]

        if arg.startswith('-') | arg.startswith('--'):
            break

        models.append(arg)

    # search models in folder
    package_dir = Path(__file__).parent / 'models'

    c = importlib.import_module(f"{package_dir}")

    for name_local in dir(c):
        print(name_local)
        if inspect.isclass(getattr(c, name_local)):
            Model = getattr(c, name_local)
            model_classes.append(Model)
            print(f"{Model.name} model has been loaded in")

    loaded_models = []

    for m in models:
        for mc in model_classes:
            if mc.name:
                if not loaded_models.__contains__(mc):
                    loaded_models.append(mc)

    main(parse_args(loaded_models), loaded_models)

    # TODO:
    # -Let the user predict X amount of time-units into the future (see bullet point 5)
    # -Discuss what kind of output / prediction we want

    # -Specify Timestep (Bryan)
    # -Specify which is the date column
    # -Figure out why it only predicts 6 timesteps
    # -Move specified column to convert
    # -Generate metadata file for algoritme
    # -Make index more time-based (this will increase accuracy by a lot)
    # -Look into WANDB