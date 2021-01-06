from dataclasses import dataclass
import json
import os
import pathlib
import random

import numpy as np
import torch

from logger import logger


def get_config(config_file_path):
    with open(config_file_path, encoding="utf-8") as f:
        config = json.load(f)

    return config


class Configuration(object):

    DEFAULT_RANDOM_SEED = 777

    @classmethod
    def apply(cls, random_seed=DEFAULT_RANDOM_SEED):
        Configuration.set_torch_seed(random_seed=random_seed)
        Configuration.set_numpy_seed(random_seed=random_seed)
        Configuration.set_python_random_seed(random_seed=random_seed)

        logger.info(f"Complete to apply the random seed, RANDOM_SEED : {random_seed}")

    @classmethod
    def set_torch_seed(cls, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def set_numpy_seed(cls, random_seed):
        np.random.seed(random_seed)

    @classmethod
    def set_python_random_seed(cls, random_seed):
        random.seed(random_seed)


@dataclass
class ApplicationProperties:
    HOME_MODULE_PATH = pathlib.Path(__file__).parent.absolute()
    CONFIG = get_config(os.path.join(HOME_MODULE_PATH, "config.json"))

    CROPPING_DIRECTORY_PATH = os.path.join(HOME_MODULE_PATH, "cropping")
    DATASET_DIRECTORY_PATH = os.path.join(HOME_MODULE_PATH, "dataset")
    DOMAIN_DIRECTORY_PATH = os.path.join(HOME_MODULE_PATH, "domain")
    MODEL_DIRECTORY_PATH = os.path.join(HOME_MODULE_PATH, "model")
    NOTEBOOK_DIRECTORY_PATH = os.path.join(HOME_MODULE_PATH, "notebook")
    TRAINER_DIRECTORY_PATH = os.path.join(HOME_MODULE_PATH, "trainer")

    DEFAULT_RANDOM_SEED = 777

    DEVICE_CPU = "cpu"
    DEVICE_GPU = "cuda"

    def __post_init__(self):
        Configuration.apply(random_seed=self.DEFAULT_RANDOM_SEED)


@dataclass
class CroppingProperties:
    CROPPING_DIRECTORY_PATH = ApplicationProperties.CROPPING_DIRECTORY_PATH
    HOME_MODULE_PATH = ApplicationProperties.HOME_MODULE_PATH
    CONFIG = ApplicationProperties.CONFIG

    # Mask
    MASK_DIR_PATH = os.path.join(CROPPING_DIRECTORY_PATH, "mask")
    HOUSING_MASK_DIR_PATH = os.path.join(MASK_DIR_PATH, "housing")
    COVER_MASK_DIR_PATH = os.path.join(MASK_DIR_PATH, "cover")

    HOUSING_CAM1_MASK_IMG_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "1.png")
    HOUSING_CAM1_BOUNDARY_MASK_IMG_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "1_boundary.png")
    HOUSING_CAM1_AREA_MASK_IMG_HOME_DIR_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "1", "area")

    HOUSING_CAM2_MASK_IMG_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "2.png")
    HOUSING_CAM2_BOUNDARY_MASK_IMG_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "2_boundary.png")
    HOUSING_CAM2_AREA_MASK_IMG_HOME_DIR_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "2", "area")
    HOUSING_CAM2_AREA_C_INNER_MASK_IMG_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "2", "2_C_inner.png")
    HOUSING_CAM2_AREA_C_OUTER_MASK_IMG_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "2", "2_C_outer.png")

    HOUSING_CAM3_MASK_IMG_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "3.png")
    HOUSING_CAM3_BOUNDARY_MASK_IMG_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "3_boundary.png")
    HOUSING_CAM3_AREA_MASK_IMG_HOME_DIR_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "3", "area")

    HOUSING_CAM4_MASK_IMG_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "4.png")
    HOUSING_CAM4_BOUNDARY_MASK_IMG_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "4_boundary.png")
    HOUSING_CAM4_AREA_MASK_IMG_HOME_DIR_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "4", "area")
    HOUSING_CAM4_AREA_C_ENTIRE_MASK_IMG_PATH = os.path.join(HOUSING_MASK_DIR_PATH, "4", "4_C_entire.png")

    COVER_CAM1_MASK_IMG_PATH = os.path.join(COVER_MASK_DIR_PATH, "1.png")
    COVER_CAM1_BOUNDARY_MASK_IMG_PATH = os.path.join(COVER_MASK_DIR_PATH, "1_boundary.png")
    COVER_CAM2_MASK_IMG_PATH = os.path.join(COVER_MASK_DIR_PATH, "2.png")
    COVER_CAM2_BOUNDARY_MASK_IMG_PATH = os.path.join(COVER_MASK_DIR_PATH, "2_boundary.png")
    COVER_CAM3_MASK_IMG_PATH = os.path.join(COVER_MASK_DIR_PATH, "3.png")
    COVER_CAM3_BOUNDARY_MASK_IMG_PATH = os.path.join(COVER_MASK_DIR_PATH, "3_boundary.png")
    COVER_CAM4_MASK_IMG_PATH = os.path.join(COVER_MASK_DIR_PATH, "4.png")
    COVER_CAM4_BOUNDARY_MASK_IMG_PATH = os.path.join(COVER_MASK_DIR_PATH, "4_boundary.png")


@dataclass
class DatasetProperties:
    DATASET_DIRECTORY_PATH = ApplicationProperties.DATASET_DIRECTORY_PATH
    HOME_MODULE_PATH = ApplicationProperties.HOME_MODULE_PATH
    CONFIG = ApplicationProperties.CONFIG

    # Data path
    DATA_DIRECTORY_PATH = os.path.join(HOME_MODULE_PATH, "data")

    # EOP_DATA_DIRECTORY_PATH = CONFIG["eop_data_directory_path"]
    EOP_DATA_DIRECTORY_PATH = os.path.join(DATA_DIRECTORY_PATH, "sample")

    LABEL_PATH = os.path.join(EOP_DATA_DIRECTORY_PATH, CONFIG["label_file_name"])
    LABEL_CSV_SEPARATE = CONFIG["label_csv_separate"]

    HOUSING_NG_PATH = os.path.join(EOP_DATA_DIRECTORY_PATH, CONFIG["housing_NG_directory_name"])
    HOUSING_OK_PATH = os.path.join(EOP_DATA_DIRECTORY_PATH, CONFIG["housing_OK_directory_name"])

    COVER_NG_PATH = os.path.join(EOP_DATA_DIRECTORY_PATH, CONFIG["cover_NG_directory_name"])
    COVER_OK_PATH = os.path.join(EOP_DATA_DIRECTORY_PATH, CONFIG["cover_OK_directory_name"])

    # Save Path
    CROPPING_DATA_DIRECTORY_PATH = os.path.join(DATA_DIRECTORY_PATH, "cropping")
    DETECT_DATA_DIRECTORY_PATH = os.path.join(DATA_DIRECTORY_PATH, "detect")
    DEFECT_DATA_DIRECTORY_PATH = os.path.join(DATA_DIRECTORY_PATH, "defect")

    # The number of defect classes
    CLASS_NUMBER = 9

    # Defect category
    NORMAL = 0  # 정상
    DEFECT_UNDER_FILL = 1  # 결육
    DEFECT_STAB = 2  # 찍힘
    DEFECT_BUBBLE = 3  # 기포
    DEFECT_CHIP_PRESSURE = 4  # 칩눌림
    DEFECT_FOREIGN_OBJECT = 5  # 이물질
    DEFECT_OTHERS = 6  # 기타
    DEFECT_BLACK_SKIN = 7  # 흑피
    DEFECT_DISCOLOR = 8  # 변색


@dataclass
class ModelProperties:
    MODEL_DIRECTORY_PATH = ApplicationProperties.MODEL_DIRECTORY_PATH
    RESULT_DIRECTORY_PATH = os.path.join(MODEL_DIRECTORY_PATH, "results")
    HOME_MODULE_PATH = ApplicationProperties.HOME_MODULE_PATH
    CONFIG = ApplicationProperties.CONFIG

    DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_CPU = torch.device("cpu")


APPLICATION_PROPERTIES = ApplicationProperties()
CROPPING_PROPERTIES = CroppingProperties()
DATASET_PROPERTIES = DatasetProperties()
MODEL_PROPERTIES = ModelProperties()
