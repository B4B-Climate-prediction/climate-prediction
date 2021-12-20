"""
A training file that can take multiple models, train them and evaluate them.

Command-arguments:
    -w [--WANDB]: API Key for WandB
    -d [--DATA]: Data file path, must be a .csv file.
    -t [--TARGETS]: columns of the data that need to be predicted
    -g [--GROUPS]: groups in datasets
    -kr [--KNREELS]: Known reels features in dataset (Also known in the future). Default: []
    -kc [--KNCATS]: Known categorical features in dataset (also know in the future). Default: []
    -ur [--UNREELS]: Unknown reel features in dataset. Default: []
    -uc [--UNCATS]: Unknown categorical features in dataset. Default: []
    -m [--MODEL]: Model file paths, must be a file. If specified it will continue training that model.
    -hy [--HYPER]: hypertunes the model if specified, this will take a long time.
    -e [--EPOCHS]: Amount of epochs to train. Default: 100
    -b [--BATCH]: Batch size during training. Default: 128
    -tr [--TRAILS]: The amount of times the hypertuning generates new hyper parameters to train the model. Default: 100
"""

import importlib
import inspect
import sys
import wandb
import os
import pandas as pd
from argparse import ArgumentParser

from pathlib import Path

from pytorch_lightning.loggers import WandbLogger

model_classes = []


def parse_args(models):
    """
    Parse the arguments from the command using argparse

    :param models: the models being used by the command

    :return: argument-parser
    """

    parser = ArgumentParser(add_help=True)

    parser.add_argument(
        '-w', '--wandb',
        type=str,
        help='API key for WandB'
    )

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
        action='extend',
        nargs='*',
        default=[],
        help='Model file paths, must be a (.?) file. If specified this model will be trained'
    )

    parser.add_argument(
        '-hy', '--hyper',
        action='store_true',
        help='Hypertunes the model if specified, this will take a long time'
    )

    parser.add_argument(
        '-e', '--epochs',
        default=100,
        type=int,
        help='Amount of epochs to training. Default: 100'
    )

    parser.add_argument(
        '-b', '--batch',
        default=128,
        type=int,
        help='Batch size during training. Default: 128'
    )

    parser.add_argument(
        '-tr', '--trials',
        default=100,
        type=int,
        help='Trials should only be used whenever hypertuning is activated. Trials specifies the amount of trials it '
             'will maximally run before coming to the best results. Default: 100 '
    )

    for model in models:
        parser.add_argument(model.name)
        model.add_arguments(parser)

    return parser.parse_args()


def main(args, chosen_models):
    """
    The main method that runs whenever the file is being used.

    :param args: the arguments in the command.
    :param chosen_models: the models have been specified in the command

    This method loops through the chosen models and executes

    Model.generate_time_series_dataset

    if hypertuning is enabled it will call:

        -Model.tune_hyper_parameter

    else it will check for an existing model otherwise it will generate a new model.
        -Model.load_model
        -Model.generate_model

    -Model.train_model
    -Model.evaluate_model

    :return: Nothing
    """

    # TODO Fix WandB logger!
    logger = None
    if args.wandb is not None:
        team = 'b4b-cp'
        project = 'climate-prediction'

        logger = WandbLogger(project=project)
        os.environ['WANDB_API_KEY'] = args.wandb

        wandb.init(project=project, entity=team)
        wandb.login()

    path = str(Path(__file__).parent / args.data)
    df = pd.read_csv(path)

    created_models = []

    index = 0  # Maybe improve with metadata files

    for model in chosen_models:
        model_class = model(df)

        print(model_class.name)

        created_models.append(model_class)

        training = model_class.generate_time_series_dataset(**vars(args))

        if args.hyper:
            trained_model = model_class.tune_hyper_parameter(training, **vars(args))

        else:
            if len(args.model) == 0:
                untrained_model = model_class.generate_model(training, **vars(args))
            else:
                untrained_model = model_class.load_model(args['model'][index])

            trained_model = model_class.train_model(training, untrained_model, **vars(args))

        # TODO: Maybe look into WandB
        model_class.evaluate_model(trained_model, training, **vars(args))
        index += 1


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
            if mc.name == m:
                print(mc.name)
                if not loaded_models.__contains__(mc):
                    loaded_models.append(mc)

    main(parse_args(loaded_models), loaded_models)
