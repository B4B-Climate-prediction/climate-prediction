import os
import pickle
import pandas as pd
from abc import ABC
from pathlib import Path

from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss, TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from model import Model


class Tft(Model, ABC):
    """Temporal Fusion Transformer model"""
    name = "Tft"

    add_arguments = lambda parser: [print(parser)]

    def __init__(self, data):
        super().__init__(data)

    # Maybe check if possible to move convert
    def generate_time_series_dataset(self, **kwargs):
        targets = kwargs['targets']

        if len(targets) == 1:
            target = targets[0]
        else:
            target = targets

        for k_cat in kwargs['kncats']:
            self.data[k_cat] = str(self.data[k_cat])

        for uk_cat in kwargs['uncats']:
            self.data[uk_cat] = str(self.data[uk_cat])

        return TimeSeriesDataSet(
            self.data,
            target=target,
            time_idx='Index',
            group_ids=kwargs['groups'],
            min_encoder_length=0,
            max_encoder_length=27,  # Zoek deze onzin nog uit!
            min_prediction_length=1,
            max_prediction_length=1,
            time_varying_known_categoricals=kwargs['kncats'],
            time_varying_known_reals=kwargs['knreels'],
            time_varying_unknown_categoricals=kwargs['uncats'],
            time_varying_unknown_reals=kwargs['unreels'],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

    def generate_model(self, dataset, **kwargs):
        targets = kwargs['targets']

        if len(targets) > 1:
            output = [7 for _ in targets]
        else:
            output = 7

        return TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=0.01,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,  # Check deze shit ook uit
            hidden_continuous_size=8,
            output_size=output,
            loss=QuantileLoss(),
            reduce_on_plateau_patience=4
        )

    def train_model(self, dataset, created_model, **kwargs):
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode='min')

        train_dataloader, val_dataloader = self.create_data_loaders(dataset, **kwargs)

        trainer = Trainer(
            max_epochs=kwargs['epochs'],
            gpus=0,
            gradient_clip_val=0.15,
            limit_train_batches=50,
            callbacks=[early_stop_callback],
            weights_save_path=str(Path(__file__).parent / 'out' / 'models'),
            # logger=kwargs['logger'] WANDB
        )

        trainer.fit(created_model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader
                    )

        return trainer  # should return something that is obtainable by wandb!

    def predict(self, model_, data_, **kwargs):
        encoder_length = 27

        predictions = [None for _ in range(encoder_length)]

        for j_lower in range(len(data_)):
            j_upper = j_lower + encoder_length

            if j_upper > (len(data_) - 1):
                break

            encoder_data = data_[j_lower:j_upper]

            last_data = encoder_data[lambda x: x.Index == x.Index.max()]
            decoder_data = last_data.assign(Timestamp=lambda y: y.Timestamp + pd.DateOffset(minutes=10))

            # add time index consistent with "data"
            decoder_data["Index"] += encoder_data["Index"].max() + decoder_data.index + 1 - decoder_data["Index"].min()

            # combine encoder and decoder data
            new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

            predictions.append((model_.predict(new_prediction_data)[0][0]).item())

        return predictions

    def tune_hyper_parameter(self, dataset, **kwargs):  # Add clean up after the creation of the best trial!
        """lol"""
        train_dataloader, val_dataloader = self.create_data_loaders(dataset, **kwargs)

        study = optimize_hyperparameters(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model_path=Path(__file__) / kwargs['model'],
            max_epochs=kwargs['epochs'],
            n_trials=kwargs['trials']
        )

        with open('optimization_summary.pkl', 'wb') as fout:
            pickle.dump(study, fout)

        # Use PATHLIB!
        path = kwargs['model'] + "/trial_" + str(study.best_trial.number)

        files = os.listdir(path)

        # SHOULD RETURN THE BEST TRIAL! IF THIS FUNCTION IS AVAILABLE!
        return TemporalFusionTransformer.load_from_checkpoint(path + "/" + files[len(files) - 1])

    def evaluate_model(self, model, dataset, **kwargs):
        _, validation_data_loader = self.create_data_loaders(dataset, **kwargs)

        raw_predictions, x = model.predict(validation_data_loader, mode="raw", return_x=True)

        for i in range(len(x)):
            model.plot_prediction(x, raw_predictions, idx=i, add_loss_to_title=True)

    def create_data_loaders(self, dataset, **kwargs):
        validation = TimeSeriesDataSet.from_dataset(dataset, self.data, predict=True, stop_randomization=True)

        return dataset.to_dataloader(train=True, batch_size=kwargs['batch'], num_workers=2,
                                     shuffle=False), validation.to_dataloader(train=False, batch_size=kwargs['batch'],
                                                                              num_workers=2, shuffle=False)

    def load_model(self, **kwargs):
        return TemporalFusionTransformer.load_from_checkpoint(Path(__file__).parent / kwargs['model'])
