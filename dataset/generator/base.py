import os

import pandas as pd

from cropping.cropper.housing.cam1_cropper import HousingCam1Cropper
from cropping.cropper.housing.cam2_cropper import HousingCam2Cropper
from cropping.cropper.housing.cam3_cropper import HousingCam3Cropper
from cropping.cropper.housing.cam4_cropper import HousingCam4Cropper
from dataset.dataframe import DataFrameFactory
from properties import DATASET_PROPERTIES
from utils import Utils


class DataGenerator(object):

    def __init__(self, tqdm_env='script', is_dataframe_loaded=True):
        self.housing_dataframe = None
        self.cover_dataframe = None
        self.tqdm = None
        self.is_plot_showed = False
        self.tqdm_disable = False

        self.set_tqdm_env(tqdm_env=tqdm_env)

        if is_dataframe_loaded:
            self.housing_dataframe = self.get_entrie_dataframe(product="housing")
            self.cover_dataframe = self.get_entrie_dataframe(product="cover")

    def set_tqdm_env(self, tqdm_env):
        tqdm_env_dict = Utils.get_tqdm_env_dict(tqdm_env=tqdm_env)

        self.tqdm = tqdm_env_dict["tqdm"]
        self.is_plot_showed = tqdm_env_dict["tqdm_disable"]
        self.tqdm_disable = tqdm_env_dict["is_plot_showed"]

    def get_entrie_dataframe(self, product):
        dataframe_list = list()
        cam_number_list = [1, 2, 3, 4]

        for cam_number in cam_number_list:
            dataframe_list.append(DataFrameFactory.get_dataframe(product=product, cam_number=cam_number))

        return pd.concat(dataframe_list, axis=0)

    def get_img_path(self, serial_number, product, cam_number):
        df: pd.DataFrame = None

        # Product controller
        if product == "housing":
            df = self.housing_dataframe
        elif product == "cover":
            df = self.cover_dataframe
        
        row = df.query("(serial_number_list == @serial_number) and (cam_list == @cam_number)")

        return row.iloc[0]["image_path_list"]

    @classmethod
    def get_product_cropper(cls, img_object, product, cam_number):
        cropper = None

        if product == "housing":
            if cam_number == 1:
                cropper = HousingCam1Cropper(img_object=img_object, defect_list=list())
            elif cam_number == 2:
                cropper = HousingCam2Cropper(img_object=img_object, defect_list=list())
            elif cam_number == 3:
                cropper = HousingCam3Cropper(img_object=img_object, defect_list=list())
            elif cam_number == 4:
                cropper = HousingCam4Cropper(img_object=img_object, defect_list=list())
        elif product == "cover":
            if cam_number == 1:
                cropper = None
            elif cam_number == 2:
                cropper = None
            elif cam_number == 3:
                cropper = None
            elif cam_number == 4:
                cropper = None

        return cropper


class TrainTestSplitter(object):

    TRAIN_CSV_FILE_NAME = "train.csv"
    TEST_CSV_FILE_NAME = "test.csv"

    def __init__(self):
        self._default_dir_path = DATASET_PROPERTIES.DATA_DIRECTORY_PATH
        self.train_csv_path = os.path.join(self.default_dir_path, TrainTestSplitter.TRAIN_CSV_FILE_NAME)
        self.test_csv_path = os.path.join(self.default_dir_path, TrainTestSplitter.TEST_CSV_FILE_NAME)

        self._train_df = pd.DataFrame()
        self._test_df = pd.DataFrame()

    @property
    def default_dir_path(self):
        return self._default_dir_path

    @default_dir_path.setter
    def default_dir_path(self, value):
        self._default_dir_path = value
        self.train_csv_path = os.path.join(self.default_dir_path, TrainTestSplitter.TRAIN_CSV_FILE_NAME)
        self.test_csv_path = os.path.join(self.default_dir_path, TrainTestSplitter.TEST_CSV_FILE_NAME)

    @property
    def train_df(self):
        if self._train_df.empty:
            self._train_df = pd.read_csv(self.train_csv_path)

        return self._train_df

    @train_df.setter
    def train_df(self, value):
        self._train_df = value

    @property
    def test_df(self):
        if self._test_df.empty:
            self._test_df = pd.read_csv(self.test_csv_path)

        return self._test_df

    @test_df.setter
    def test_df(self, value):
        self._test_df = value

    @classmethod
    def get_entire_default_img_path_list(cls, dataset_name, ext='.npy'):
        default_dir_path = os.path.join(DATASET_PROPERTIES.DATA_DIRECTORY_PATH, dataset_name)

        entire_ok_img_path_list = list()
        entire_ng_img_path_list = list()

        product_dir_path_list = [os.path.join(default_dir_path, dir_path)
                                 for dir_path in os.listdir(default_dir_path)
                                 if os.path.isdir(os.path.join(default_dir_path, dir_path))]

        for product_dir_path in product_dir_path_list:

            cam_dir_path_list = [os.path.join(product_dir_path, dir_path)
                                 for dir_path in os.listdir(product_dir_path)
                                 if os.path.isdir(os.path.join(product_dir_path, dir_path))
                                 and dir_path.isdigit()]

            for cam_dir_path in cam_dir_path_list:
                # Check all crop_part
                crop_dir_path_list = [os.path.join(cam_dir_path, dir_path)
                                      for dir_path in os.listdir(cam_dir_path)
                                      if os.path.isdir(os.path.join(cam_dir_path, dir_path))
                                      and dir_path.split("_")[0] == "grid"
                                      and dir_path.split("_")[1].isdigit()]

                for crop_dir_path in crop_dir_path_list:
                    ok_img_dir_path = os.path.join(crop_dir_path, "OK")
                    ng_img_dir_path = os.path.join(crop_dir_path, "NG")

                    ok_img_path_list = [os.path.join(ok_img_dir_path, img_path)
                                        for img_path in os.listdir(ok_img_dir_path)
                                        if os.path.isfile(os.path.join(ok_img_dir_path, img_path))
                                        and os.path.splitext(os.path.join(ok_img_dir_path, img_path))[-1] == ext]

                    ng_img_path_list = [os.path.join(ng_img_dir_path, img_path)
                                        for img_path in os.listdir(ng_img_dir_path)
                                        if os.path.isfile(os.path.join(ng_img_dir_path, img_path))
                                        and os.path.splitext(os.path.join(ng_img_dir_path, img_path))[-1] == ext]

                    entire_ok_img_path_list += ok_img_path_list
                    entire_ng_img_path_list += ng_img_path_list

        return entire_ok_img_path_list, entire_ng_img_path_list

    @classmethod
    def get_entire_defect_img_path_list(cls, ext='.npy'):
        default_dir_path = DATASET_PROPERTIES.DEFECT_DATA_DIRECTORY_PATH
        entire_img_path_list = list()

        product_dir_path_list = [os.path.join(default_dir_path, dir_path)
                                 for dir_path in os.listdir(default_dir_path)
                                 if os.path.isdir(os.path.join(default_dir_path, dir_path))]

        for product_dir_path in product_dir_path_list:
            cam_dir_path_list = [os.path.join(product_dir_path, dir_path)
                                 for dir_path in os.listdir(product_dir_path)
                                 if os.path.isdir(os.path.join(product_dir_path, dir_path))
                                 and dir_path.isdigit()]

            for cam_dir_path in cam_dir_path_list:
                defect_dir_path_list = [os.path.join(cam_dir_path, dir_path)
                                        for dir_path in os.listdir(cam_dir_path)
                                        if os.path.isdir(os.path.join(cam_dir_path, dir_path))]

                for defect_dir_path in defect_dir_path_list:
                    img_path_list = [os.path.join(defect_dir_path, img_path)
                                     for img_path in os.listdir(defect_dir_path)
                                     if os.path.isfile(os.path.join(defect_dir_path, img_path))
                                     and os.path.splitext(os.path.join(defect_dir_path, img_path))[-1] == ext]

                    entire_img_path_list += img_path_list

        return entire_img_path_list

    @classmethod
    def get_entire_img_path_list(cls, dataset_name, ext=".npy"):
        # default_dir_path = os.path.join(DATASET_PROPERTIES.DATA_DIRECTORY_PATH, defect_img_category, product, str(cam_number))
        entire_ok_img_path_list, entire_ng_img_path_list = list(), list()

        if dataset_name == "defect":
            entire_ng_img_path_list = cls.get_entire_defect_img_path_list(ext=ext)
        else:
            entire_ok_img_path_list, entire_ng_img_path_list = cls.get_entire_default_img_path_list(dataset_name=dataset_name, ext=ext)

        return entire_ok_img_path_list, entire_ng_img_path_list
