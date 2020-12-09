import sys
from properties import *

sys.path.append(APPLICATION_PROPERTIES.HOME_MODULE_PATH)


import os

import pandas as pd


class DataFrameFactory(object):

    def __init__(self):
        pass

    @classmethod
    def get_dataframe(cls, product, cam_number):
        # product = model_data.product
        # cam_number = model_data.cam_number
        label_df = cls.get_label_df(DATASET_PROPERTIES.LABEL_PATH, sep=DATASET_PROPERTIES.LABEL_CSV_SEPARATE)
        ng_image_path_dict = None
        ok_image_path_dict = None

        if product == "housing":
            ng_image_path_dict = cls.get_image_path_dict(
                dir_path=DATASET_PROPERTIES.HOUSING_NG_PATH,
                cam_number=cam_number
            )
            ok_image_path_dict = cls.get_image_path_dict(
                dir_path=DATASET_PROPERTIES.HOUSING_OK_PATH,
                cam_number=cam_number
            )
        elif product == "cover":
            ng_image_path_dict = cls.get_image_path_dict(
                dir_path=DATASET_PROPERTIES.COVER_NG_PATH,
                cam_number=cam_number
            )
            ok_image_path_dict = cls.get_image_path_dict(
                dir_path=DATASET_PROPERTIES.COVER_OK_PATH,
                cam_number=cam_number
            )

        ng_dataframe = cls.get_data(
            cam_num=cam_number,
            image_path_dict=ng_image_path_dict,
            label_df=label_df,
            is_NG=True
        )
        ok_dataframe = cls.get_data(
            cam_num=cam_number,
            image_path_dict=ok_image_path_dict,
            label_df=label_df,
            is_NG=False
        )

        ng_discolor_dataframe, ng_dataframe = cls.split_dataframe_with_defect_category(
            df=ng_dataframe,
            defect_category=DATASET_PROPERTIES.DEFECT_DISCOLOR
        )
        ok_dataframe = ok_dataframe.reset_index()  # ??
        ok_dataframe = ok_dataframe.drop(columns="index")

        total_dataframe = ng_dataframe.append(ok_dataframe)

        """
        @TODO
        변색 따로 할 것
        """
        # ng_discolor_dataframe

        total_dataframe = total_dataframe.reset_index()
        total_dataframe = total_dataframe.drop(columns="index")

        return total_dataframe

    @classmethod
    def get_image_path_dict(cls, dir_path, cam_number) -> dict:
        image_path_dict = dict()

        dir_path_list_with_serial = [os.path.join(dir_path, directory) for directory in os.listdir(dir_path) if
                                     os.path.isdir(os.path.join(dir_path, directory))]

        for dir_path_with_serial in dir_path_list_with_serial:
            for file in os.listdir(dir_path_with_serial):
                if os.path.splitext(file)[1] != ".png":
                    continue

                serial_number, file_cam_name = file.split("_")[-5:-3]
                file_cam_number = int(file_cam_name[-1])

                if cam_number == file_cam_number:
                    image_path_dict[serial_number] = os.path.join(dir_path_with_serial, file)

        return image_path_dict

    @classmethod
    def get_label_df(cls, label_path, sep='\t'):
        label_df = pd.read_csv(label_path, sep=sep)

        """
        @TODO
        이거 나중에 cover 파일 바뀌었으므로 수정할 것
        """
        # Preprocessing columns
        label_df.set_index('SERIAL_NO', inplace=True)
        label_df['CAM_INDEX'] = label_df['CAMERA_INFO'].map(lambda x: int(x[-1]))
        label_df['DEFECT_CATEGORY'] = label_df['LABEL_BAD_STATUS'].map(lambda x: int(x[1:]))
        label_df['PRODUCT_CATEGORY'] = label_df['WORK_SHOP_ID'].map(lambda x: 'Housing' if x == 1680 else 'Cover')

        return label_df

    @classmethod
    def get_data(cls, cam_num, image_path_dict, label_df, is_NG: bool):
        data_dict = {
            'serial_number_list': list(),
            'image_path_list': list(),
            'is_NG_list': list(),
            'cam_list': list(),
            'defect_category_list': list(),
            'x_st_list': list(),
            'x_ed_list': list(),
            'y_st_list': list(),
            'y_ed_list': list(),
            'ratio_list': list()
        }

        for serial_number, image_path in image_path_dict.items():
            if is_NG:
                label_df_with_cam = label_df[(label_df.index == serial_number) & (label_df['CAM_INDEX'] == cam_num)]

                if label_df_with_cam.size == 0:
                    continue

                defect_category_list_by_cam = list()
                x_st_list_by_cam = list()
                x_ed_list_by_cam = list()
                y_st_list_by_cam = list()
                y_ed_list_by_cam = list()
                ratio_list_by_cam = list()

                for idx, row in label_df_with_cam.iterrows():
                    defect_category_list_by_cam.append(row['DEFECT_CATEGORY'])
                    x_st_list_by_cam.append(int(row['START_X']))
                    x_ed_list_by_cam.append(int(row['END_X']))
                    y_st_list_by_cam.append(int(row['START_Y']))
                    y_ed_list_by_cam.append(int(row['END_Y']))
                    ratio_list_by_cam.append(row['RESIZE_RATE'])

                data_dict['serial_number_list'].append(serial_number)
                data_dict['image_path_list'].append(image_path)
                data_dict['is_NG_list'].append(1)
                data_dict['cam_list'].append(cam_num)
                data_dict['defect_category_list'].append(defect_category_list_by_cam)
                data_dict['x_st_list'].append(x_st_list_by_cam)
                data_dict['x_ed_list'].append(x_ed_list_by_cam)
                data_dict['y_st_list'].append(y_st_list_by_cam)
                data_dict['y_ed_list'].append(y_ed_list_by_cam)
                data_dict['ratio_list'].append(ratio_list_by_cam)
            else:
                data_dict['serial_number_list'].append(serial_number)
                data_dict['image_path_list'].append(image_path)
                data_dict['is_NG_list'].append(0)
                data_dict['cam_list'].append(cam_num)
                data_dict['defect_category_list'].append([0])
                data_dict['x_st_list'].append([0])
                data_dict['x_ed_list'].append([0])
                data_dict['y_st_list'].append([0])
                data_dict['y_ed_list'].append([0])
                data_dict['ratio_list'].append([0])

        return pd.DataFrame(data_dict)

    @classmethod
    def split_dataframe_with_defect_category(cls, df, defect_category: int):
        defect_discolor_index_list = list()

        for idx, defect_category_list in enumerate(df['defect_category_list']):
            if defect_category in defect_category_list:
                defect_discolor_index_list.append(idx)

        defect_df = df.iloc[defect_discolor_index_list].reset_index()
        other_df = df.drop(index=defect_df.index).reset_index()

        defect_df = defect_df.drop(columns="index")
        other_df = other_df.drop(columns="index")

        return defect_df, other_df
