import importlib
import inspect
import sys
import wandb
import pickle, os
import pandas as pd
from argparse import ArgumentParser

from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_forecasting.metrics import QuantileLoss, MultiLoss
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from models import *

model_classes = []


def parse_args(models):
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
        type=str,
        help='Model file path, must be a (.?) file. If specified this model will be trained'
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
        model.implement_command_args()

    return parser.parse_args()


def build_timeseries_dataset(data, targets, groups, knreels, kncats, unreels, uncats):
    if len(targets) == 1:
        target = targets[0]
    else:
        target = targets

    return TimeSeriesDataSet(
        data,
        target=target,
        time_idx='Index',
        group_ids=groups,
        min_encoder_length=0,
        max_encoder_length=27,
        min_prediction_length=6,
        max_prediction_length=6,
        time_varying_known_categoricals=kncats,
        time_varying_known_reals=knreels,
        time_varying_unknown_categoricals=uncats,
        time_varying_unknown_reals=unreels,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )


def build_timeseries_trainer(epochs, logger):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode='min')

    return Trainer(
        max_epochs=epochs,
        gpus=0,
        gradient_clip_val=0.15,
        limit_train_batches=50,
        callbacks=[early_stop_callback],
        weights_save_path=str(Path(__file__).parent / 'out' / 'models'),
        logger=logger
    )


def build_timeseries_model(data, targets):
    output_size = 7

    if len(targets) > 1:
        output = [output_size for _ in targets]
    else:
        output = output_size

    return TemporalFusionTransformer.from_dataset(
        data,
        learning_rate=0.01,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=output,
        # Onthoudt wel dat deze output_size op basis van de hoeveelheid targets aangepast moet worden.
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4
    )


def split_timeseries_dataset(dataset, df, batch):
    validation = TimeSeriesDataSet.from_dataset(dataset, df, predict=True, stop_randomization=True)

    return dataset.to_dataloader(train=True, batch_size=batch, num_workers=2, shuffle=False), validation.to_dataloader(
        train=False, batch_size=batch, num_workers=2, shuffle=False)


def evaluate_model(model, val):
    raw_predictions, x = model.predict(val, mode="raw", return_x=True)

    figs = []
    for i in range(len(x)):
        figs.append(model.plot_prediction(x, raw_predictions, idx=0, add_loss_to_title=True))


def main(args):
    print(f'{args=}')

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

    for k_cat in args.kncats:
        df[k_cat] = str(df[k_cat])

    for uk_cat in args.uncats:
        df[uk_cat] = str(df[uk_cat])

    # Datasets
    training = build_timeseries_dataset(
        df,
        targets=args.targets,
        groups=args.groups,
        knreels=args.knreels,
        kncats=args.kncats,
        unreels=args.unreels,
        uncats=args.uncats
    )

    # Dataloaders
    train_dataloader, val_dataloader = split_timeseries_dataset(training, df, args.batch)

    if args.hyper:
        study = optimize_hyperparameters(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model_path=args.model,
            max_epochs=args.epochs,
            n_trials=args.trials
        )

        with open("optimization_summary.pkl", "wb") as fout:
            pickle.dump(study, fout)  # the information from optimization.

        path = args.model + "/trial_" + str(study.best_trial.number)

        files = os.listdir(path)

        model = TemporalFusionTransformer.load_from_checkpoint(path + "/" + files[len(files) - 1])

        print("Best trial: " + str(study.best_trial.number))

    else:
        # Trainer
        trainer = build_timeseries_trainer(args.epochs, logger)

        # Model
        if args.model is None:
            model = build_timeseries_model(training, args.targets)
        else:
            model = TemporalFusionTransformer.load_from_checkpoint(args.model)

        # Training
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

    # Evaluate
    evaluate_model(model, val_dataloader)


if __name__ == '__main__':
    print(sys.argv)

    models = []

    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]

        if arg.startswith('-') | arg.startswith('--'):
            break

        models.append(arg)

    # search models in folder

    package_dir = Path(__file__).parent / 'models'

    for file in iter(os.listdir(package_dir)):
        c = importlib.import_module(f"{package_dir}")

        if file.startswith("__"):
            continue

        for name_local in dir(c):
            # print(name_local)
            # if inspect.isclass(getattr(c, name_local)):
            if name_local.startswith("__"):
                continue

            Model = getattr(c, name_local)
            model_classes.append(Model)
            print(f"{name_local} model has been loaded in")

    loaded_models = []

    for m in models:
        for mc in model_classes:
            if mc.get_model_name():
                if not loaded_models.__contains__(mc):
                    loaded_models.append(mc)

    main(parse_args(loaded_models))
