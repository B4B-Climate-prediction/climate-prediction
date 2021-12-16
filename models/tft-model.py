import os
import pickle
from abc import ABC
from pathlib import Path

from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss, TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from model import Model


class Tft(Model, ABC):
    # Maybe check if possible to move convert
    def generate_time_series_dataset(self, data, **kwargs):
        targets = kwargs['targets']

        if len(targets) == 1:
            target = targets[0]
        else:
            target = targets

        return TimeSeriesDataSet(
            data,
            target=target,
            time_idx='Index',
            group_ids=kwargs['groups'],
            min_encoder_length=0,
            max_encoder_length=27,  # Zoek deze onzin nog uit!
            min_prediction_length=6,
            max_prediction_length=6,
            time_varying_known_categoricals=kwargs['kncats'],
            time_varying_known_reals=kwargs['knreels'],
            time_varying_unknown_categoricals=kwargs['uncats'],
            time_varying_unknown_reals=kwargs['unreels'],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

    def generate_model(self, data, **kwargs):
        targets = kwargs['target']

        if len(targets) > 1:
            output = [kwargs['output_size'] for _ in targets]
        else:
            output = kwargs['output_size']

        return TemporalFusionTransformer.from_dataset(
            data,
            learning_rate=0.01,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,  # Check deze shit ook uit
            hidden_continuous_size=8,
            output_size=output,
            loss=QuantileLoss(),
            reduce_on_plateau_patience=4
        )

    def train_model(self, data, model, **kwargs):
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode='min')

        trainer = Trainer(
            max_epochs=kwargs['epochs'],
            gpus=0,
            gradient_clip_val=0.15,
            limit_train_batches=50,
            callbacks=[early_stop_callback],
            weights_save_path=str(Path(__file__).parent / 'out' / 'models'),
            logger=kwargs['logger']
        )

        trainer.fit(model,
                    # train_dataloaders=,
                    # val_dataloaders=
                    )

        return trainer  # should return something that is obtainable by wandb!

    # TODO
    def predict(self):
        raise NotImplementedError()

    def tune_hyper_parameter(self, **kwargs):  # Add clean up after the creation of the best trial!
        study = optimize_hyperparameters(
            # train_dataloader=train_dataloader,
            # val_dataloader=val_dataloader,
            model_path=kwargs['model'],
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

    def evaluate_model(self, model, validation):
        raw_predictions, x = model.predict(validation, mode="raw", return_x=True)

        for i in range(len(x)):
            model.plot_prediction(x, raw_predictions, idx=i, add_loss_to_title=True)

