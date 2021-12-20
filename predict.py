import pickle, os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_forecasting.metrics import QuantileLoss, MultiLoss
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pandas.tseries.offsets import DateOffset
import ast


def parse_args():
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

    return parser.parse_args()

def build_timeseries_dataset(data, targets, groups, knreels, kncats, unreels, uncats):
    if len(targets) == 1:
        target = targets[0]
    else:
        target = targets

    print(target)
    
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
        allow_missing_timesteps=True,
        predict_mode=True
    )

def main(args):
    print(args)

    path = str(Path(__file__).parent / 'out' / 'datasets' / args.data)
    df = pd.read_csv(path)

    for k_cat in args.kncats:
        df[k_cat] = str(df[k_cat])

    for uk_cat in args.uncats:
        df[uk_cat] = str(df[uk_cat])

    df[args.Timestamp] = pd.to_datetime(df[args.Timestamp])

    dataset = build_timeseries_dataset(
        df, 
        targets=args.targets, 
        groups=args.groups,
        knreels=args.knreels,
        kncats=args.kncats,
        unreels=args.unreels,
        uncats=args.uncats
    )

    data_loader = dataset.to_dataloader(train=False, batch_size=args.batch, num_workers=2, shuffle=False)

    #init model
    model = TemporalFusionTransformer.load_from_checkpoint(args.model)

    encoder_data = df[lambda x: x.Index > x.Index.max() - 27]

    # select last known data point and create decoder data from it by repeating it and incrementing the month
    # in a real world dataset, we should not just forward fill the covariates but specify them to account
    # for changes in special days and prices (which you absolutely should do but we are too lazy here)
    last_data = df[lambda x: x.index == x.index.max()]
    decoder_data = pd.concat([last_data.assign(Timestamp=lambda x: x[args.Timestamp] + DateOffset(ast.parse('{args.timeunit[1]}={(int(args.timeunit[0])}'))) for i in range(1, args.timesteps + 1)],ignore_index=True)

    # add time index consistent with "data"
    decoder_data["Index"] += encoder_data["Index"].max() + decoder_data.index + 1 - decoder_data["Index"].min()

    print(decoder_data)

    # combine encoder and decoder data
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
    
    print(new_prediction_data)

    raw_predictions, x = model.predict(new_prediction_data, return_x=True)
    
    print(x)
    #for i in range(len(raw_predictions)):
    # model.plot_prediction(x, raw_predictions, idx=0, plot_attention=False, show_future_observed=False)
    # plt.plot()
    # path = str(Path(__file__).parent / 'out' / 'pictures' / 'prediction.png')
    # plt.savefig(path)        

if __name__ == '__main__':
    main(parse_args())


    #TODO:
    # -Let the user predict X amount of time-units into the future (see bullet point 5)
    # -Discuss what kind of output / prediction we want

    # -Specify Timestep
    # -Specify which is the date column
    # -Figure out why it only predicts 6 timesteps
    # -Move specified column to convert
    # -Generate metadata file for algoritme
    # -Make index more time-based (this will increase accuracy by a lot)
