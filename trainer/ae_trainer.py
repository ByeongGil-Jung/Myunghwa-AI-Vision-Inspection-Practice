import json
import os
import time
from collections import deque
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# from sklearn.metrics import roc_curve, auc

from dataset.generator.ae_data_generator import AutoencoderDataGenerator
from domain.criterion import Criterion
from domain.image import BoundingBox
from logger import logger
from trainer.base import TrainerBase
from trainer.plot import *

time.time()


class AutoencoderTrainer(TrainerBase):

    def __init__(self, model, model_file_metadata, train_loader, val_loader, test_loader, hyperparameters, tqdm_env='script'):
        super(AutoencoderTrainer, self).__init__(model, model_file_metadata, train_loader, val_loader, test_loader, hyperparameters, tqdm_env)
        self.property_file_path = os.path.join(self.model_file_metadata.model_dir_path, "config.json")

        self.best_boundary_loss = None

    def _load_best_boundary_loss(self):
        assert os.path.isfile(self.property_file_path), "Plz train, first"

        with open(self.property_file_path, encoding="utf-8") as f:
            config = json.load(f)

        self.best_boundary_loss = config['best_boundary_loss']

    def _save_best_boundary_loss(self):
        config = dict(
            best_boundary_loss=self.best_boundary_loss
        )

        with open(self.property_file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def train(self):
        self.model.train()

        train_result_dict_list = list()
        val_result_dict_list = list()

        # Set hyperparameters
        optimizer_cls = self.hyperparameters.optimizer_cls
        lr = self.hyperparameters.lr
        weight_decay = self.hyperparameters.weight_decay
        n_epoch = self.hyperparameters.n_epoch
        early_stopping_patience = self.hyperparameters.early_stopping_patience
        criterion = self.hyperparameters.criterion
        device = self.hyperparameters.device
        is_saved = self.hyperparameters.is_saved
        checkpoint_period = self.hyperparameters.checkpoint_period

        optimizer = optimizer_cls(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Early stopping variables
        early_stopping_value_queue = deque(maxlen=early_stopping_patience)
        early_stopping_epoch = 0
        early_stopping_count = 0
        early_stopping_best_value = 99999  # Temp loss
        early_stopping_best_model = None

        for epoch in self.tqdm.tqdm(range(n_epoch)):
            running_loss = 0
            batch_n = 0

            for i, (img_batch, is_NG_batch, defect_category_batch, _, _, serial_number_batch) in enumerate(self.train_loader):
                img_batch = img_batch.to(device)
                is_NG_batch = is_NG_batch.to(device)

                # Optimization
                optimizer.zero_grad()
                gen_img_batch = self.model(img_batch)
                loss = criterion(img_batch, gen_img_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_n += 1

            current_loss = running_loss / batch_n

            # Validating
            val_result_dict = self.validate()

            train_result_dict = dict(
                loss=current_loss
            )

            train_result_dict_list.append(train_result_dict)
            val_result_dict_list.append(val_result_dict)

            ####################################################################################################
            """
            Early Stopping
            """
            if early_stopping_patience is not None:
                # early stopping 큐가 비어있지 않다면 수행
                if epoch == 0:
                    early_stopping_best_value = val_result_dict["normal_loss"]
                else:
                    if early_stopping_count == early_stopping_patience:
                        print(f"Early stopping, Epoch : {early_stopping_epoch}")
                        break

                early_stopping_value_queue.append(val_result_dict["normal_loss"])
                # 만약 새로 들어온 loss 가 기존의 loss 보다 작을 때, 새로 카운트 시작
                current_lowest_value = sorted(early_stopping_value_queue)[0]

                if current_lowest_value < early_stopping_best_value:
                    early_stopping_epoch = epoch
                    early_stopping_best_model = deepcopy(self.model.state_dict())

                    early_stopping_count = 0
                    early_stopping_best_value = current_lowest_value
                else:
                    early_stopping_count += 1

            ################################################################################################

            # Print
            print(
                f"[ Epoch {epoch} ]\n"
                f"Train Loss : {round(train_result_dict['loss'], 7)}\n"
                f"Validation Loss : {round(val_result_dict['normal_loss'], 7)}"
            )

            if early_stopping_patience is not None:
                print(
                    f"Early Stopping Count : {early_stopping_count}, "
                    f"Best Validation Loss : {early_stopping_best_value}"
                )

            # Checkpoint save
            if checkpoint_period:
                if epoch % checkpoint_period == 0:
                    self.save_model_checkpoint(epoch=epoch)
                    self.save_record_checkpoint(record=dict(
                        early_stopping_epoch=early_stopping_epoch,
                        result_dict=val_result_dict
                    ), epoch=epoch)

                    print(f"Save checkpoint - epoch : {epoch}")

        if early_stopping_patience is not None:
            self.model.load_state_dict(early_stopping_best_model)

        best_result_dict = self.validate()
        """ 여기 할 것 ~~~~~~ 
        """

        record_dict = dict(
            early_stopping_epoch=early_stopping_epoch,
            train_history=train_result_dict_list,
            val_history=val_result_dict_list,
            best_result_dict=best_result_dict
        )

        """
        Save Best Model
        """
        if is_saved:
            self.save_best_model(model=self.model)
            self.save_entire_record(record=record_dict)

            # Save best boundary loss
            self._save_best_boundary_loss()

            # Save plot
            train_loss_list = list(map(lambda train_result_dict: train_result_dict["loss"], train_result_dict_list))
            val_loss_list = list(map(lambda val_result_dict: val_result_dict["normal_loss"], val_result_dict_list))

            ax = plot_loss(
                train_data_list=train_loss_list,
                val_data_list=val_loss_list,
                early_stopping_epoch=early_stopping_epoch,
                ax=None
            )

            self.save_plot()

            if self.is_plot_showed:
                plt.show()

            logger.debug("Success to save model")
        else:
            train_loss_list = list(map(lambda train_result_dict: train_result_dict["loss"], train_result_dict_list))
            val_loss_list = list(map(lambda val_result_dict: val_result_dict["normal_loss"], val_result_dict_list))

            ax = plot_loss(
                train_data_list=train_loss_list,
                val_data_list=val_loss_list,
                early_stopping_epoch=early_stopping_epoch,
                ax=None
            )

            plt.show()

        return record_dict

    def validate(self):
        self.model.eval()
        result_dict = dict()
        result_list = list()
        reconstruction_error_list = list()
        is_NG_list = list()

        # Set hyperparameters
        criterion = self.hyperparameters.criterion
        device = self.hyperparameters.device

        total_normal_loss = 0
        normal_data_n = 0
        n_batch = 0

        for i, (original_img_batch, is_NG_batch, defect_category_batch, _, _, serial_number_batch) in enumerate(self.test_loader):
            original_img_batch = original_img_batch.to(device)
            is_NG_batch = is_NG_batch.to(device)

            with torch.no_grad():
                generated_img_batch = self.model(original_img_batch)

                for original_img, generated_img, is_NG in zip(original_img_batch, generated_img_batch, is_NG_batch):
                    loss = criterion(original_img, generated_img).item()
                    # original_img = original_img.cpu().numpy().squeeze()
                    # generated_img = generated_img.cpu().numpy().squeeze()

                    # OK 인 것만 loss 측정
                    if not is_NG:
                        total_normal_loss += loss
                        normal_data_n += 1

                    result_list.append(dict(
                        loss=loss,
                        is_NG=int(is_NG.item())
                    ))

                reconstruction_error_batch = Criterion.reconstruction_error(x=original_img_batch, x_hat=generated_img_batch)

            reconstruction_error_list.append(reconstruction_error_batch)
            is_NG_list.append(is_NG_batch)

            n_batch += 1

        # Calculate AUC
        # reconstruction_error_list = torch.cat(reconstruction_error_list).cpu().numpy()
        # is_NG_list = torch.cat(is_NG_list).cpu().numpy()  # 1d 행렬로 바꾸기
        #
        # fpr, tpr, thresholds = roc_curve(is_NG_list, reconstruction_error_list)
        # auc_value = auc(fpr, tpr)

        current_normal_loss = total_normal_loss / normal_data_n

        return dict(
            normal_loss=current_normal_loss,
            # auc=auc_value,
            result_list=result_list
        )

    # Testing
    def evaluate_with_save(self, image_metadata, dataset_name, dataset_properties, is_difference_img_preprocessed=True, is_removed=True):
        self.model.eval()

        serial_number_list = list()
        original_img_list = list()
        generated_img_list = list()
        defect_category_list = list()
        bbox_pt_x_list = list()
        bbox_pt_y_list = list()
        bbox_list = list()

        # Set hyperparameters
        device = self.hyperparameters.device

        ae_data_generator = AutoencoderDataGenerator(image_metadata=image_metadata, resize_rate=0.5)

        for i, (original_img_batch, is_NG_batch, defect_category_batch, bbox_pt_x_list_batch, bbox_pt_y_list_batch, serial_number_batch) in enumerate(self.test_loader):
            original_img_batch = original_img_batch.to(device)

            with torch.no_grad():
                generated_img_batch = self.model(original_img_batch)

            serial_number_list += serial_number_batch
            original_img_list.append(original_img_batch.cpu())
            generated_img_list.append(generated_img_batch.cpu())
            defect_category_list.append(defect_category_batch.cpu())

            # Appending bounding box to list
            for bbox_pt_x, bbox_pt_y in zip(bbox_pt_x_list_batch, bbox_pt_y_list_batch):
                bbox_pt_x_list.append(bbox_pt_x.numpy())
                bbox_pt_y_list.append(bbox_pt_y.numpy())

            # Set bounding box
            bbox_pt_x_list = np.asarray(bbox_pt_x_list).transpose()
            bbox_pt_y_list = np.asarray(bbox_pt_y_list).transpose()

            for bbox_pt_x_values, bbox_pt_y_values in zip(bbox_pt_x_list, bbox_pt_y_list):
                x_st = int(bbox_pt_x_values[0])
                y_st = int(bbox_pt_y_values[0])
                x_ed = int(bbox_pt_x_values[2])
                y_ed = int(bbox_pt_y_values[2])

                bbox_list.append(BoundingBox(x_st=x_st, y_st=y_st, x_ed=x_ed, y_ed=y_ed))

        original_img_list = torch.cat(original_img_list, dim=0).numpy().squeeze(axis=1)
        generated_img_list = torch.cat(generated_img_list, dim=0).numpy().squeeze(axis=1)
        defect_category_list = torch.cat(defect_category_list, dim=0).numpy()

        ae_data_generator.generate(
            serial_number_list=serial_number_list,
            original_img_list=original_img_list,
            generated_img_list=generated_img_list,
            defect_category_list=defect_category_list,
            bbox_list=bbox_list,
            dataset_name=dataset_name,
            is_difference_img_preprocessed=is_difference_img_preprocessed,
            is_removed=is_removed,
            properties=dataset_properties
        )

    def predict(self, data_loader, threshold, condition: str="gt"):
        self.model.eval()

        original_img_list = list()
        generated_img_list = list()
        is_NG_pred_list = list()
        score_list = list()
        bbox_point_list = list()
        bbox_pt_x_list = list()
        bbox_pt_y_list = list()

        # Set hyperparameters
        criterion = self.hyperparameters.criterion
        device = self.hyperparameters.device

        elapsed_time_st = time.time()

        if self.best_boundary_loss is None:
            self._load_best_boundary_loss()

        for original_img_batch, is_NG, defect_category, bbox_pt_x_list_batch, bbox_pt_y_list_batch, serial_number_batch in data_loader:
            original_img_batch = original_img_batch.to(device)

            with torch.no_grad():
                generated_img_batch = self.model(original_img_batch)

            for original_img, generated_img in zip(original_img_batch, generated_img_batch):
                loss = criterion(original_img, generated_img).item()

                original_img = original_img.squeeze().cpu().numpy()
                generated_img = generated_img.squeeze().cpu().numpy()

                original_img_list.append(original_img)
                generated_img_list.append(generated_img)
                score_list.append(loss)

                if condition == "lt":
                    is_NG_pred_list.append(loss <= threshold)
                elif condition == "gt":
                    is_NG_pred_list.append(loss >= threshold)

            # Appending bounding box to list
            for bbox_pt_x, bbox_pt_y in zip(bbox_pt_x_list_batch, bbox_pt_y_list_batch):
                bbox_pt_x_list.append(bbox_pt_x.numpy())
                bbox_pt_y_list.append(bbox_pt_y.numpy())

        # Set bounding box
        bbox_pt_x_list = np.asarray(bbox_pt_x_list).transpose()
        bbox_pt_y_list = np.asarray(bbox_pt_y_list).transpose()

        for bbox_pt_x_values, bbox_pt_y_values in zip(bbox_pt_x_list, bbox_pt_y_list):
            bbox_point_dict = dict()

            for i in range(4):
                bbox_point_dict[f"pt_{i + 1}"] = dict(
                    x=int(bbox_pt_x_values[i]),
                    y=int(bbox_pt_y_values[i])
                )

            bbox_point_list.append(bbox_point_dict)

        elapsed_time_ed = time.time()

        return dict(
            original_img_list=original_img_list,
            generated_img_list=generated_img_list,
            NG_pred_list=is_NG_pred_list,
            score_list=score_list,
            bbox_point_list=bbox_point_list,
            elapsed_time=elapsed_time_ed - elapsed_time_st
        )
