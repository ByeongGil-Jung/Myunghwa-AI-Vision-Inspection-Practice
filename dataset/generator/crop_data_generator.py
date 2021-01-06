import os
import random
import shutil
from pathlib import Path

import cv2
import pandas as pd

from cropping.cam_properties import cam1_properties as c1p
from cropping.cam_properties import cam2_properties as c2p
from cropping.cam_properties import cam3_properties as c3p
from cropping.cam_properties import cam4_properties as c4p
from cropping.cropper.base import CropperBase
from cropping.cropping_factory import CropperFactory
from cropping.image_data import BoundingBox, Defect, EOPImage, Coordinate
from dataset.generator.base import DataGenerator
from dataset.dataframe import DataFrameFactory
from domain.data import EOPData
from domain.metadata import ImageMetadata
from model.model_set import ModelSet
from properties import DATASET_PROPERTIES


"""
@TODO
이거 시간과 용량 때문에 dataset 들어가기 직전의 list 를 따로 저장해둘까
>> 이렇게 하면 나중에 데이터가 추가 됐을 때도 관리하기 편할 것 같음
>> pickle ? 등으로 관리 ? list 만 저장 ?
"""
class CropDataGenerator(DataGenerator):

    def __init__(self, image_metadata: ImageMetadata, dataframe=None, tqdm_env='script', is_dataframe_loaded=True):
        super(CropDataGenerator, self).__init__(tqdm_env=tqdm_env, is_dataframe_loaded=is_dataframe_loaded)
        self.image_metadata = image_metadata

        """
        @TODO
        이거 나중에 지울 것 (부모 클래스에서 다 참조할 수 있도록 하기)
        """
        if isinstance(dataframe, pd.DataFrame):
            self.serial_number_list = dataframe['serial_number_list']
            self.image_path_list = dataframe['image_path_list']
            self.is_NG_list = dataframe['is_NG_list']
            self.cam_list = dataframe['cam_list']
            self.defect_category_lists = dataframe['defect_category_list']
            self.x_st_lists = dataframe['x_st_list']
            self.x_ed_lists = dataframe['x_ed_list']
            self.y_st_lists = dataframe['y_st_list']
            self.y_ed_lists = dataframe['y_ed_list']
            self.ratio_lists = dataframe['ratio_list']

        # Set Path
        self.cropping_data_save_dir_path = os.path.join(
            DATASET_PROPERTIES.CROPPING_DATA_DIRECTORY_PATH,
            self.image_metadata.product,
            str(self.image_metadata.cam_number),
            self.image_metadata.crop_part
        )
        self.cropping_ok_data_save_dir_path = os.path.join(self.cropping_data_save_dir_path, "OK")
        self.cropping_ng_data_save_dir_path = os.path.join(self.cropping_data_save_dir_path, "NG")
        self.cropping_data_label_csv_file_path = os.path.join(self.cropping_data_save_dir_path, "label.csv")
        self.normalization_variables_file_name = os.path.join(self.cropping_data_save_dir_path, "normalization_variables.json")

        self.cropper: CropperBase = None

    @classmethod
    def create(cls, image_metadata: ImageMetadata, tqdm_env='script'):
        product = image_metadata.product
        cam_number = image_metadata.cam_number

        return cls(image_metadata=image_metadata, dataframe=DataFrameFactory.get_dataframe(product=product, cam_number=cam_number), tqdm_env=tqdm_env)

    @classmethod
    def get_label_df(cls, image_metadata):
        cdg = cls(image_metadata=image_metadata)

        return pd.read_csv(cdg.cropping_data_label_csv_file_path, sep=",")

    @classmethod
    def get_entire_label_df(cls, is_saved=False, is_loaded=True):
        model_set: ModelSet = None
        label_df_list = list()
        model_set_list = [
            ModelSet.housing_cam1_autoencoder_model_set_1(),
            ModelSet.housing_cam2_autoencoder_model_set_1(),
            ModelSet.housing_cam3_autoencoder_model_set_1(),
            ModelSet.housing_cam4_autoencoder_model_set_1()
        ]

        # # Controller
        # if product == "housing":
        #     if cam_number == 1:
        #         model_set = ModelSet.housing_cam1_model_set_grid_all()
        #     elif cam_number == 2:
        #         model_set = ModelSet.housing_cam2_model_set_grid_all()
        #     elif cam_number == 3:
        #         model_set = ModelSet.housing_cam3_model_set_grid_all()
        #     elif cam_number == 4:
        #         model_set = ModelSet.housing_cam4_model_set_grid_all()

        # Get all label dataframes
        for model_set in model_set_list:
            for model_object in model_set.models.values():
                image_metadata = model_object.image_metadata

                label_df_list.append(CropDataGenerator.get_label_df(image_metadata=image_metadata))

        # Concat
        entire_label_df = pd.concat(label_df_list, axis=0)

        # Save
        """
        default directory 에 entire_label.csv 저장하는 함수 만들기
        """

        return entire_label_df

    @classmethod
    def get_anomaly_data_with_image_metadata(cls, image_metadata, train_ok_ratio_against_ng=1.0, train_sample_ratio=None, is_loaded=True, is_saved=False, is_shuffle=False, is_train_entire=False):
        image_metadata = image_metadata

        def __split_train_test_with_ok_ng_ratio(ok_data_list, ng_data_list, train_ok_ratio_against_ng):
            test_ok_data_size = int(len(ng_data_list) * train_ok_ratio_against_ng)

            test_ok_data_list = ok_data_list[:test_ok_data_size]
            train_ok_data_list = ok_data_list[test_ok_data_size:]

            train_data_list = train_ok_data_list
            test_data_list = ng_data_list + test_ok_data_list

            print(f"Train OK data size : {len(train_ok_data_list)}, "
                  f"Test OK data size : {len(test_ok_data_list)}, "
                  f"Test NG data size : {len(ng_data_list)}")

            return train_data_list, test_data_list

        default_dir_path = DATASET_PROPERTIES.CROPPING_DATA_DIRECTORY_PATH
        default_dir_path = os.path.join(default_dir_path, image_metadata.product, str(image_metadata.cam_number), image_metadata.crop_part)

        ok_img_dir_path = os.path.join(default_dir_path, "OK")
        ng_img_dir_path = os.path.join(default_dir_path, "NG")

        train_csv_file_path = os.path.join(default_dir_path, "train.csv")
        test_csv_file_path = os.path.join(default_dir_path, "test.csv")

        # Load if was saved before
        if not is_loaded:
            ok_img_path_list = [os.path.join(ok_img_dir_path, img_path)
                                for img_path in os.listdir(ok_img_dir_path)
                                if os.path.isfile(os.path.join(ok_img_dir_path, img_path))
                                and os.path.splitext(os.path.join(ok_img_dir_path, img_path))[-1] == ".png"]

            ng_img_path_list = [os.path.join(ng_img_dir_path, img_path)
                                for img_path in os.listdir(ng_img_dir_path)
                                if os.path.isfile(os.path.join(ng_img_dir_path, img_path))
                                and os.path.splitext(os.path.join(ng_img_dir_path, img_path))[-1] == ".png"]

            # Shuffling
            if is_shuffle:
                random.shuffle(ok_img_path_list)
                random.shuffle(ng_img_path_list)

            # Split with train ratio
            train_img_path_list, test_img_path_list = __split_train_test_with_ok_ng_ratio(
                ok_data_list=ok_img_path_list,
                ng_data_list=ng_img_path_list,
                train_ok_ratio_against_ng=train_ok_ratio_against_ng
            )

            # Save to csv
            if is_saved:
                train_path_df = pd.DataFrame(dict(
                    PATH=train_img_path_list
                ))
                test_path_df = pd.DataFrame(dict(
                    PATH=test_img_path_list
                ))

                train_path_df.to_csv(train_csv_file_path, sep=",", index=False)
                test_path_df.to_csv(test_csv_file_path, sep=",", index=False)
        else:
            train_img_path_df = pd.read_csv(train_csv_file_path, sep=",")
            test_img_path_df = pd.read_csv(test_csv_file_path, sep=",")

            train_img_path_list = train_img_path_df["PATH"].tolist()
            test_img_path_list = test_img_path_df["PATH"].tolist()

            if is_train_entire:
                test_OK_img_path_list = [test_img_path for test_img_path in test_img_path_list if 'OK' in test_img_path]
                test_NG_img_path_list = [test_img_path for test_img_path in test_img_path_list if 'NG' in test_img_path]

                train_img_path_list += test_OK_img_path_list
                test_img_path_list = test_NG_img_path_list

        # Train sampling
        if train_sample_ratio:
            train_img_path_list = random.sample(train_img_path_list, int(len(train_img_path_list) * train_sample_ratio))

        return train_img_path_list, test_img_path_list

    """
    @TODO
    이 컨트롤러들 다 부모클래스 classmethod 로 빼기
    """
    def get_grid_left_up_coord(self, image_metadata: ImageMetadata):
        left_up_coord: Coordinate = None

        product = image_metadata.product
        cam_number = image_metadata.cam_number
        crop_part = image_metadata.crop_part

        if product == "housing":
            if cam_number == 1:
                if crop_part == "grid_1":
                    left_up_coord = c1p.CROP_GRID_1_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_2":
                    left_up_coord = c1p.CROP_GRID_2_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_3":
                    left_up_coord = c1p.CROP_GRID_3_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_4":
                    left_up_coord = c1p.CROP_GRID_4_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_5":
                    left_up_coord = c1p.CROP_GRID_5_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_6":
                    left_up_coord = c1p.CROP_GRID_6_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_7":
                    left_up_coord = c1p.CROP_GRID_7_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_8":
                    left_up_coord = c1p.CROP_GRID_8_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_9":
                    left_up_coord = c1p.CROP_GRID_9_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_10":
                    left_up_coord = c1p.CROP_GRID_10_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_11":
                    left_up_coord = c1p.CROP_GRID_11_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_12":
                    left_up_coord = c1p.CROP_GRID_12_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_13":
                    left_up_coord = c1p.CROP_GRID_13_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_14":
                    left_up_coord = c1p.CROP_GRID_14_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_15":
                    left_up_coord = c1p.CROP_GRID_15_VARIABLE.LEFT_UP_COORDINATE
            if cam_number == 2:
                if crop_part == "grid_1":
                    left_up_coord = c2p.CROP_GRID_1_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_2":
                    left_up_coord = c2p.CROP_GRID_2_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_3":
                    left_up_coord = c2p.CROP_GRID_3_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_4":
                    left_up_coord = c2p.CROP_GRID_4_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_5":
                    left_up_coord = c2p.CROP_GRID_5_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_6":
                    left_up_coord = c2p.CROP_GRID_6_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_7":
                    left_up_coord = c2p.CROP_GRID_7_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_8":
                    left_up_coord = c2p.CROP_GRID_8_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_9":
                    left_up_coord = c2p.CROP_GRID_9_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_10":
                    left_up_coord = c2p.CROP_GRID_10_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_11":
                    left_up_coord = c2p.CROP_GRID_11_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_12":
                    left_up_coord = c2p.CROP_GRID_12_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_13":
                    left_up_coord = c2p.CROP_GRID_13_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_14":
                    left_up_coord = c2p.CROP_GRID_14_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_15":
                    left_up_coord = c2p.CROP_GRID_15_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_16":
                    left_up_coord = c2p.CROP_GRID_16_VARIABLE.LEFT_UP_COORDINATE
            if cam_number == 3:
                if crop_part == "grid_1":
                    left_up_coord = c3p.CROP_GRID_1_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_2":
                    left_up_coord = c3p.CROP_GRID_2_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_3":
                    left_up_coord = c3p.CROP_GRID_3_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_4":
                    left_up_coord = c3p.CROP_GRID_4_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_5":
                    left_up_coord = c3p.CROP_GRID_5_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_6":
                    left_up_coord = c3p.CROP_GRID_6_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_7":
                    left_up_coord = c3p.CROP_GRID_7_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_8":
                    left_up_coord = c3p.CROP_GRID_8_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_9":
                    left_up_coord = c3p.CROP_GRID_9_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_10":
                    left_up_coord = c3p.CROP_GRID_10_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_11":
                    left_up_coord = c3p.CROP_GRID_11_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_12":
                    left_up_coord = c3p.CROP_GRID_12_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_13":
                    left_up_coord = c3p.CROP_GRID_13_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_14":
                    left_up_coord = c3p.CROP_GRID_14_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_15":
                    left_up_coord = c3p.CROP_GRID_15_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_16":
                    left_up_coord = c3p.CROP_GRID_16_VARIABLE.LEFT_UP_COORDINATE
                if crop_part == "grid_17":
                    left_up_coord = c3p.CROP_GRID_17_VARIABLE.LEFT_UP_COORDINATE
            if cam_number == 4:
                if crop_part == "grid_1":
                    left_up_coord = c4p.CROP_GRID_1_VARIABLE.LEFT_UP_COORDINATE

        return left_up_coord

    def _set_cropper(self, img_object: EOPImage, defect_list: list):
        self.cropper = CropperFactory.get(self.image_metadata, img_object, defect_list)

    def _get_last_file_number(self, dir_path):
        file_name_list = os.listdir(dir_path)

        if not file_name_list:
            return -1
        else:
            return int(sorted(list(map(lambda file_name: file_name.split("_")[0], file_name_list)))[-1])

    def _resize(self, img_object: EOPImage, resize_size, resize_rate):
        # Resize
        if resize_size is not None:
            img_object.img = cv2.resize(
                img_object.img, dsize=(resize_size[0], resize_size[1]), interpolation=cv2.INTER_AREA
            )
        if resize_rate is not None:
            img_object.img = cv2.resize(
                img_object.img, dsize=(0, 0), fx=resize_rate, fy=resize_rate, interpolation=cv2.INTER_AREA
            )

    def get_normalization_variables_file_path(self):
        return

    # def generate_defect(self, generate_size: int = None, img_size=(128, 128), is_saved=True, is_removed=True):
    #     product = self.image_metadata.product
    #     cam_number = self.image_metadata.cam_number
    #
    #     generate_size = self._adjust_generate_size(generate_size)
    #
    #     # Create directory
    #     default_save_dir_path = DATASET_PROPERTIES.DEFECT_DATA_DIRECTORY_PATH
    #
    #     if is_removed:
    #         if os.path.isdir(default_save_dir_path):
    #             shutil.rmtree(default_save_dir_path)
    #
    #     Path(default_save_dir_path).mkdir(parents=True, exist_ok=True)
    #
    #     # Generate and save
    #     img_list = list()
    #
    #     for i in self.tqdm.tqdm(range(generate_size)):
    #         eop_data: EOPData = self._get_eop_data(index=i)
    #         cropper = self.get_product_cropper(img_object=eop_data.img_object, product=product, cam_number=cam_number)
    #         img = cropper.img_object.img
    #         defect_list = eop_data.defect_list
    #
    #         serial_number = self.serial_number_list[i]
    #
    #         if defect_list:
    #             # Save image
    #
    #             for idx, defect in enumerate(defect_list):
    #                 defect: Defect
    #                 defect_category = defect.category
    #
    #                 save_path = os.path.join(default_save_dir_path, str(defect_category))
    #                 Path(save_path).mkdir(parents=True, exist_ok=True)
    #
    #                 defect_abst_bbox: BoundingBox = defect.abst_bbox
    #
    #                 if not img_size:
    #                     x_st = defect_abst_bbox.x_st
    #                     x_ed = defect_abst_bbox.x_ed
    #                     y_st = defect_abst_bbox.y_st
    #                     y_ed = defect_abst_bbox.y_ed
    #                 else:
    #                     # img_size 를 넘어가는 것들 거르기
    #                     defect_x_size = defect_abst_bbox.x_ed - defect_abst_bbox.x_st
    #                     defect_y_size = defect_abst_bbox.y_ed - defect_abst_bbox.y_st
    #                     if img_size[0] < defect_y_size or img_size[1] < defect_x_size:
    #                         continue
    #
    #                     y_difference = round(img_size[0] / 2)
    #                     x_difference = round(img_size[1] / 2)
    #
    #                     x_st = defect.center_x - x_difference
    #                     x_ed = defect.center_x + x_difference
    #                     y_st = defect.center_y - y_difference
    #                     y_ed = defect.center_y + y_difference
    #
    #                 defect_img = img[y_st:y_ed, x_st:x_ed]
    #
    #                 # 이상한 이미지 거르기
    #                 if defect_img.shape[0] == 0 or defect_img.shape[1] == 0:
    #                     continue
    #
    #                 normalized_img = defect_img.copy()
    #                 normalized_img = (normalized_img - np.min(normalized_img)) / (np.max(normalized_img) - np.min(normalized_img))
    #
    #                 original_img_save_path = os.path.join(save_path, f"{serial_number}_{product}_{cam_number}_{idx}_o.npy")
    #                 normalized_img_save_path = os.path.join(save_path, f"{serial_number}_{product}_{cam_number}_{idx}_on.npy")
    #
    #                 if is_saved:
    #                     np.save(original_img_save_path, defect_img)
    #                     np.save(normalized_img_save_path, normalized_img)
    #
    #                 img_list.append(defect_img)
    #
    #     return img_list


    def generate(self, generate_size: int = None, resize_size: tuple=None, resize_rate: float=None, is_saved=True, is_removed=True, **properties):
        product = self.image_metadata.product
        cam_number = self.image_metadata.cam_number
        crop_part = self.image_metadata.crop_part

        generate_size = self._adjust_generate_size(generate_size)

        # Create directory

        # cropping_data_save_dir_path = os.path.join(DATASET_PROPERTIES.CROPPING_DATA_DIRECTORY_PATH, product, str(cam_number), crop_part)
        # cropping_ok_data_save_dir_path = os.path.join(cropping_data_save_dir_path, "OK")
        # cropping_ng_data_save_dir_path = os.path.join(cropping_data_save_dir_path, "NG")
        # cropping_data_label_csv_file_path = os.path.join(self.cropping_data_save_dir_path, "label.csv")

        # Remove all old cropped images
        if is_removed:
            if os.path.isdir(self.cropping_ok_data_save_dir_path):
                shutil.rmtree(self.cropping_ok_data_save_dir_path)
            if os.path.isdir(self.cropping_ng_data_save_dir_path):
                shutil.rmtree(self.cropping_ng_data_save_dir_path)

        Path(self.cropping_ok_data_save_dir_path).mkdir(parents=True, exist_ok=True)
        Path(self.cropping_ng_data_save_dir_path).mkdir(parents=True, exist_ok=True)

        # ok_file_last_number = self._get_last_file_number(dir_path=cropping_ok_data_save_dir_path)
        # ng_file_last_number = self._get_last_file_number(dir_path=cropping_ng_data_save_dir_path)

        ok_file_path_list = list()
        ng_file_path_list = list()

        # csv list
        csv_serial_number_list = list()
        csv_product_list = list()
        csv_cam_number_list = list()
        csv_crop_part_list = list()
        csv_defect_categoty_list = list()
        csv_x_st_list = list()
        csv_x_ed_list = list()
        csv_y_st_list = list()
        csv_y_ed_list = list()

        # Generate and save
        for i in self.tqdm.tqdm(range(generate_size)):
            eop_data = self._get_eop_data(index=i)
            defect_list = eop_data.defect_list
            self._set_cropper(img_object=eop_data.img_object, defect_list=defect_list)

            is_defect = True if defect_list else False

            serial_number = self.serial_number_list[i]

            # Get cropped image data list
            crop_img_data_list = self.cropper.crop(**properties)
            
            # Grid 용 시작 위치 설정
            left_up_coord = self.get_grid_left_up_coord(image_metadata=self.image_metadata)

            # 만약 결함 이미지일 경우, 결함 부분만 추출하고 나머진 버림
            if is_defect:
                # Save image
                for idx, crop_img_data in enumerate(crop_img_data_list):
                    crop_img_data: EOPImage
                    crop_img_defect_list = crop_img_data.defect_list

                    # Resize
                    self._resize(img_object=crop_img_data, resize_size=resize_size, resize_rate=resize_rate)
                    img = crop_img_data.img

                    # 결함 별 이미지 저장
                    for defect in crop_img_defect_list:
                        defect_category = defect.category

                        save_path = os.path.join(self.cropping_ng_data_save_dir_path,
                                                 f"{serial_number}_{idx}_{defect_category}.png")
                        if is_saved:
                            cv2.imwrite(save_path, img)

                        # csv 정보 저장
                        defect_abst_bbox: BoundingBox = defect.abst_bbox
                        csv_serial_number_list.append(serial_number)
                        csv_product_list.append(product)
                        csv_cam_number_list.append(cam_number)
                        csv_crop_part_list.append(crop_part)
                        csv_defect_categoty_list.append(defect_category)
                        csv_x_st_list.append(int((defect_abst_bbox.x_st - left_up_coord.x) / 2))
                        csv_x_ed_list.append(int((defect_abst_bbox.x_ed - left_up_coord.x) / 2))
                        csv_y_st_list.append(int((defect_abst_bbox.y_st - left_up_coord.y) / 2))
                        csv_y_ed_list.append(int((defect_abst_bbox.y_ed - left_up_coord.y) / 2))

            # 만약 정상 이미지일 경우, 모두 추출
            else:
                for idx, crop_img_data in enumerate(crop_img_data_list):
                    crop_img_data: EOPImage

                    # Resize
                    self._resize(img_object=crop_img_data, resize_size=resize_size, resize_rate=resize_rate)
                    img = crop_img_data.img

                    save_path = os.path.join(self.cropping_ok_data_save_dir_path, f"{serial_number}_{idx}_{0}.png")

                    # Save
                    if is_saved:
                        cv2.imwrite(save_path, img)

        # Save to csv
        label_df = pd.DataFrame(dict(
            SERIAL_NO=csv_serial_number_list,
            PRODUCT_NAME=csv_product_list,
            CAMERA_INFO=csv_cam_number_list,
            CROP_PART=csv_crop_part_list,
            LABEL_BAD_STATUS=csv_defect_categoty_list,
            START_X=csv_x_st_list,
            END_X=csv_x_ed_list,
            START_Y=csv_y_st_list,
            END_Y=csv_y_ed_list
        ))
        label_df.to_csv(self.cropping_data_label_csv_file_path, sep=",", index=False)

    def _get_eop_data(self, index: int):
        eop_data = EOPData(
            serial_number=self.serial_number_list[index],
            image_path=self.image_path_list[index],
            is_NG=self.is_NG_list[index],
            cam=self.cam_list[index],
            defect_list=None
        )

        defect_list = list()

        # 만약 결함이 있으면 defect 추가
        if eop_data.is_NG:
            for defect_category, x_st, x_ed, y_st, y_ed, ratio in \
                    zip(self.defect_category_lists[index], self.x_st_lists[index], self.x_ed_lists[index],
                        self.y_st_lists[index], self.y_ed_lists[index], self.ratio_lists[index]):
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

    def _adjust_generate_size(self, generate_size: int=None):
        if generate_size is None:
            generate_size = len(self.serial_number_list)
        elif generate_size > len(self.serial_number_list):
            generate_size = len(self.serial_number_list)

        return generate_size
