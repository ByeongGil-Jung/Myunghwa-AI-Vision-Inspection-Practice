import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
from torchvision import transforms

from dataset.base import Dataset
from domain.data import EOPData, InputData
from domain.image import BoundingBox, Defect, EOPImage
from properties import DATASET_PROPERTIES


class EOPDataset(Dataset):

    def __init__(self, data_list, normalization_variables_file_path, normalization_variables="save"):
        self.data_list = data_list
        self.data_type = None
        self.normalization_variables_file_path = normalization_variables_file_path
        self.normalization_variables = None

        self._set_data_type()
        self._set_normalization_variables(normalization_variables)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def _set_data_type(self):
        self.data_type = type(self.data_list[0])

    def _set_normalization_variables(self, normalization_variables):
        if normalization_variables == "save":
            self.normalization_variables = self.get_normalization_variables(is_saved=True)
        elif normalization_variables == "load":
            with open(self.normalization_variables_file_path, encoding="utf-8") as f:
                normalization_variables_dict = json.load(f)

            mean = normalization_variables_dict["mean"]
            std = normalization_variables_dict["std"]

            self.normalization_variables = (mean,), (std,)
            print(f"Normalization variables loaded : {self.normalization_variables}")
        else:
            self.normalization_variables = normalization_variables

    def get_normalization_variables(self, is_saved=True):
        data_list = self.data_list
        data_list_size = len(data_list)
        entire_mean = 0
        entire_std = 0

        for data in data_list:
            img_object = None

            """
            @TODO
            이거 수정하기 (EOPImage 에서 모두 handling 되도록)
            """
            if self.data_type == str:
                img_path = data
                # img_object = EOPImage(img=cv2.cvtColor(cv2.imread(data).astype(np.float32), cv2.COLOR_BGR2GRAY))
                img_object = EOPImage(img=img_path)
            elif isinstance(self.data_type, np.ndarray):
                img_object = EOPImage(img=data)

            img = img_object.img

            entire_mean += np.mean(img)
            entire_std += np.std(img)

        entire_mean /= data_list_size
        entire_std /= data_list_size

        # Save to json
        if is_saved:
            normalization_variables_dict = dict()

            normalization_variables_dict["mean"] = entire_mean
            normalization_variables_dict["std"] = entire_std

            Path(self.normalization_variables_file_path).parent.mkdir(parents=True, exist_ok=True)

            with open(self.normalization_variables_file_path, "w", encoding="utf-8") as f:
                json.dump(normalization_variables_dict, f, indent=4)

        return (entire_mean,), (entire_std,)


class EOPTrainingDataset(EOPDataset):

    def __init__(self, img_path_list, normalization_variables_file_path, normalization_variables="save", resize_size: tuple = None, resize_rate: float = None,
                 transform=None, is_shuffle=True, is_buffered=False):
        super(EOPTrainingDataset, self).__init__(img_path_list, normalization_variables_file_path, normalization_variables)
        # self.data_list = img_path_list
        self.img_data_list = list()
        self.resize_size = resize_size
        self.resize_rate = resize_rate
        self.transform = None
        self.is_buffered = is_buffered

        # Shuffling
        if is_shuffle:
            random.shuffle(self.data_list)

        # 만약 메모리에 바로 데이터를 올리고 싶을 때
        if self.is_buffered:
            self.set_buffer()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*self.normalization_variables)
            ])

    def __getitem__(self, idx):
        # 이미지 path 기반 데이터 반환
        if not self.is_buffered:
            img_path = self.data_list[idx]

            img_file_name, ext = os.path.splitext(os.path.basename(img_path))
            serial_number, img_idx, defect_category = img_file_name.split("_")
            defect_category = int(defect_category)

            if ext == ".npy":
                img_object = EOPImage(img=np.load(img_path))
            elif ext == ".png":
                img_object = EOPImage(img=img_path)

            # Resize
            if self.resize_size is not None:
                img_object.img = cv2.resize(
                    img_object.img, dsize=(self.resize_size[0], self.resize_size[1]), interpolation=cv2.INTER_AREA
                )
            if self.resize_rate is not None:
                img_object.img = cv2.resize(
                    img_object.img, dsize=(0, 0), fx=self.resize_rate, fy=self.resize_rate, interpolation=cv2.INTER_AREA
                )

            input_data = InputData(img_object=img_object)

            # Set defect
            input_data.is_NG = np.zeros(1, dtype=np.float32)
            input_data.defect_category = np.zeros(DATASET_PROPERTIES.CLASS_NUMBER, dtype=np.float32)

            if defect_category != 0:
                input_data.is_NG[0] = 1

            input_data.defect_category[defect_category] = 1

            # Transform image
            input_data.img = self.transform(input_data.img)

            return input_data.img, input_data.is_NG, input_data.defect_category, list(), list(), img_file_name
        # 메모리에 올려진 이미지 데이터 반환
        else:
            input_data: InputData = self.img_data_list[idx]
            img_data = self.transform(input_data.img)

            return img_data, input_data.is_NG, input_data.defect_category, list(), list(), ""

    def __len__(self):
        return len(self.data_list)

    def set_buffer(self):
        for img_path in self.data_list:
            img_file_name = os.path.splitext(os.path.basename(img_path))[0]
            serial_number, img_idx, defect_category = img_file_name.split("_")
            defect_category = int(defect_category)

            img_object = EOPImage(img=cv2.cvtColor(cv2.imread(img_path).astype(np.float32), cv2.COLOR_BGR2GRAY))
            input_data = InputData(img_object=img_object)

            # Set defect
            input_data.is_NG = np.zeros(1, dtype=np.float32)
            input_data.defect_category = np.zeros(DATASET_PROPERTIES.CLASS_NUMBER, dtype=np.float32)

            if defect_category != 0:
                input_data.is_NG[0] = 1

            input_data.defect_category[defect_category] = 1

            self.img_data_list.append(input_data)

        print("Completed to set image data to memory")


class EOPViewDataset(Dataset):

    def __init__(self, data_dict: dict, transform=None):
        super(EOPViewDataset, self).__init__()
        self.serial_number_list = data_dict['serial_number_list']
        self.image_path_list = data_dict['image_path_list']
        self.is_NG_list = data_dict['is_NG_list']
        self.cam_list = data_dict['cam_list']
        self.defect_category_lists = data_dict['defect_category_list']
        self.x_st_lists = data_dict['x_st_list']
        self.x_ed_lists = data_dict['x_ed_list']
        self.y_st_lists = data_dict['y_st_list']
        self.y_ed_lists = data_dict['y_ed_list']
        self.ratio_lists = data_dict['ratio_list']
        self.transform = transform

    def __getitem__(self, idx):
        eop_data = EOPData(
            serial_number=self.serial_number_list[idx],
            image_path=self.image_path_list[idx],
            is_NG=self.is_NG_list[idx],
            cam=self.cam_list[idx],
            defect_list=None
        )

        defect_list = list()

        # 만약 결함이 있으면 defect 추가
        if eop_data.is_NG:
            for defect_category, x_st, x_ed, y_st, y_ed, ratio in \
                    zip(self.defect_category_lists[idx], self.x_st_lists[idx], self.x_ed_lists[idx],
                        self.y_st_lists[idx], self.y_ed_lists[idx], self.ratio_lists[idx]):
                defect_abst_bbox = BoundingBox(x_st=x_st, x_ed=x_ed, y_st=y_st, y_ed=y_ed)
                defect_abst_bbox.calculate_with_ratio(ratio)

                defect = Defect(
                    abst_bbox=defect_abst_bbox,
                    category=defect_category,
                    location_img=eop_data.img_object
                )
                defect_list.append(defect)

        eop_data.defect_list = defect_list

        return eop_data

    def __len__(self):
        return len(self.image_path_list)