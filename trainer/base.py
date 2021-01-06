from pathlib import Path
import os
import pickle
import time

import matplotlib.pyplot as plt
import torch

from domain.metadata import ModelFileMetadata
from logger import logger
from properties import APPLICATION_PROPERTIES
from utils import Utils

time.time()


class TrainerBase(object):

    def __init__(self, model, model_file_metadata: ModelFileMetadata, train_loader, val_loader, test_loader, hyperparameters, tqdm_env='script', _logger=logger, is_saved=True):
        self.model = model
        self.model_file_metadata = model_file_metadata
        self.train_loader = train_loader
        self.val_loader = val_loader if val_loader else test_loader
        self.test_loader = test_loader
        self.hyperparameters = hyperparameters
        self.logger = _logger

        # Set environments
        self.tqdm = None
        self.is_plot_showed = False
        self.tqdm_disable = False

        self.set_tqdm_env(tqdm_env=tqdm_env)

        if is_saved:
            self.create_model_directory()

        # Set model configuration
        if isinstance(self.model, torch.nn.Module):
            self.model.to(self.hyperparameters.device)
            logger.info(f"Model set to '{self.hyperparameters.device}'")

        self.best_model = dict()

    def set_tqdm_env(self, tqdm_env):
        tqdm_env_dict = Utils.get_tqdm_env_dict(tqdm_env=tqdm_env)

        self.tqdm = tqdm_env_dict["tqdm"]
        self.is_plot_showed = tqdm_env_dict["tqdm_disable"]
        self.tqdm_disable = tqdm_env_dict["is_plot_showed"]

    def create_model_directory(self):
        Path(self.model_file_metadata.model_dir_path).mkdir(parents=True, exist_ok=True)

    def save_model_checkpoint(self, epoch):
        Path(self.model_file_metadata.model_checkpoint_dir_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_file_metadata.get_save_model_checkpoint_file_path(epoch=epoch))

    def save_record_checkpoint(self, record, epoch):
        Path(self.model_file_metadata.model_checkpoint_dir_path).mkdir(parents=True, exist_ok=True)
        with open(self.model_file_metadata.get_save_record_checkpoint_file_path(epoch=epoch), "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model_checkpoint(self, epoch):
        model_file_path = self.model_file_metadata.get_save_model_checkpoint_file_path(epoch=epoch)
        self.load_model_with_path(model_file_path=model_file_path)

    def get_record_checkpoint(self, epoch):
        record = None

        with open(self.model_file_metadata.get_save_record_checkpoint_file_path(epoch=epoch), "rb") as f:
            pickle.load(f)

        return record

    def save_best_model(self, model):
        Path(self.model_file_metadata.model_dir_path).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.model_file_metadata.get_best_model_file_path())

    def load_best_model(self):
        best_model_file_path = self.model_file_metadata.get_best_model_file_path()
        self.load_model_with_path(model_file_path=best_model_file_path)

    def load_model_with_path(self, model_file_path):
        if os.path.isfile(model_file_path):
            self.model.load_state_dict(torch.load(model_file_path, map_location=APPLICATION_PROPERTIES.DEVICE_CPU))
            self.model.to(self.hyperparameters.device)

            logger.info(f"Succeed to load best model, device: '{self.hyperparameters.device}'")
        else:
            logger.error(f"Failed to load best model, file not exist")

    def save_entire_record(self, record):
        Path(self.model_file_metadata.model_dir_path).mkdir(parents=True, exist_ok=True)
        with open(self.model_file_metadata.get_entire_record_file_path(), "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_entire_record(self):
        entire_record_file = None
        entire_record_file_path = self.model_file_metadata.get_entire_record_file_path()

        if os.path.isfile(entire_record_file_path):
            with open(entire_record_file_path, "rb") as f:
                entire_record_file = pickle.load(f)

            logger.info(f"Succeed to get entire record file")
        else:
            logger.error(f"Failed to get entire record file, file not exist")

        return entire_record_file

    def save_plot(self):
        plt.savefig(self.model_file_metadata.get_plot_file_path(), dpi=300)

    def train(self, *args, **kwargs):
        pass

    def validate(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
