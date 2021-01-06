import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from cropping.cropping_factory import CropperFactory
from dataset.generator.base import DataGenerator
from domain.metadata import ImageMetadata
from properties import DATASET_PROPERTIES


class AutoencoderDataGenerator(DataGenerator):

    def __init__(self, image_metadata: ImageMetadata, resize_rate=0.5, tqdm_env='script', is_dataframe_loaded=True):
        super(AutoencoderDataGenerator, self).__init__(tqdm_env=tqdm_env, is_dataframe_loaded=is_dataframe_loaded)
        self.image_metadata = image_metadata
        self.resize_rate = resize_rate

        if self.image_metadata.product == "housing" and self.image_metadata.cam_number == 4:
            self.resize_rate = 2.0
        # self.grid_mask = self.get_grid_mask(resize_rate=resize_rate)

    @classmethod
    def get_defect_coord(cls, difference_img):
        return np.where(np.any(difference_img == np.max(difference_img), axis=1))[0][0], \
               np.where(np.any(difference_img == np.max(difference_img), axis=0))[0][0]


    """
    @TODO
    이거 가장자리 noise 없애는 건데 사용하지 않는 것이 나을 듯
    """
    @classmethod
    def get_cut_difference_img(cls, difference_original_img):
        difference_cut_img = difference_original_img.copy()

        while True:
            defect_coord = cls.get_defect_coord(difference_cut_img)
            defect_row, defect_col = defect_coord
            len_row = difference_cut_img.shape[0]
            len_col = difference_cut_img.shape[1]

            # End condition
            if (defect_row != 0 and defect_row != len_row) and (defect_col != 0 and defect_col != len_col):
                break
            else:
                if defect_row == 0:
                    difference_cut_img = difference_cut_img[1:, :]
                elif defect_row == len_row:
                    difference_cut_img = difference_cut_img[:len_row, :]
                if defect_col == 0:
                    difference_cut_img = difference_cut_img[:, 1:]
                elif defect_col == len_col:
                    difference_cut_img = difference_cut_img[:, :len_col]

        return difference_cut_img

    @classmethod
    def get_cut_defect_img(cls, img, defect_coord, cut_size):
        cut_size_half = cut_size // 2
        len_row = img.shape[0]
        len_col = img.shape[1]

        defect_row = defect_coord[0]
        defect_col = defect_coord[1]

        part_img_x_st = defect_row - cut_size_half
        part_img_x_ed = defect_row + cut_size_half
        part_img_y_st = defect_col - cut_size_half
        part_img_y_ed = defect_col + cut_size_half

        output_img = np.zeros((cut_size, cut_size))
        x_st_difference, x_ed_difference, y_st_difference, y_ed_difference = 0, 0, 0, 0

        if part_img_x_st <= 0:
            x_st_difference = abs(part_img_x_st)
            part_img_x_st = 0
        if part_img_x_ed >= len_row:
            x_ed_difference = abs(part_img_x_ed - len_row)
            part_img_x_ed = len_row
        if part_img_y_st <= 0:
            y_st_difference = abs(part_img_y_st)
            part_img_y_st = 0
        if part_img_y_ed >= len_col:
            y_ed_difference = abs(part_img_y_ed - len_col)
            part_img_y_ed = len_col

        defect_part_img = img[part_img_x_st:part_img_x_ed, part_img_y_st:part_img_y_ed]

        output_img[x_st_difference:cut_size - x_ed_difference, y_st_difference:cut_size - y_ed_difference] = defect_part_img

        return output_img

    @classmethod
    def get_resize_img(cls, img_object, resize_size, resize_rate):
        output_img = img_object.img.copy()

        # Resize
        if resize_size is not None:
            output_img = cv2.resize(
                img_object.img, dsize=(resize_size[0], resize_size[1]), interpolation=cv2.INTER_AREA
            )
        if resize_rate is not None:
            output_img = cv2.resize(
                img_object.img, dsize=(0, 0), fx=resize_rate, fy=resize_rate, interpolation=cv2.INTER_AREA
            )

        return output_img

    @classmethod
    def preprocess_difference_img(cls, difference_img, mask_img):
        preprocessed_difference_img = difference_img.copy()

        preprocessed_difference_img = preprocessed_difference_img * mask_img
        preprocessed_difference_img = np.negative(preprocessed_difference_img)
        # preprocessed_difference_img = preprocessed_difference_img / 255
        preprocessed_difference_img = np.where(preprocessed_difference_img < 0, 0, preprocessed_difference_img)

        return preprocessed_difference_img

    def get_grid_mask(self, is_boundary_mask=False, resize_rate=0.5):
        cropper = CropperFactory.get(image_metadata=self.image_metadata, img_object=None, defect_list=[], is_mask=True)
        cropping_properties = CropperFactory.get_properties(image_metadata=self.image_metadata)
        cropping_properties["is_boundary_mask"] = is_boundary_mask

        crop_grid_image_list = cropper.crop_grid(**cropping_properties)
        crop_grid_image = crop_grid_image_list[0]
        crop_grid_image = self.get_resize_img(crop_grid_image, resize_size=None, resize_rate=resize_rate)

        return crop_grid_image

    def _create_save_directory(self, directory_name, is_removed=True):
        # Create directory
        defect_data_save_dir_path = os.path.join(DATASET_PROPERTIES.DATA_DIRECTORY_PATH, directory_name, self.image_metadata.product, str(self.image_metadata.cam_number), self.image_metadata.crop_part)
        defect_ok_data_save_dir_path = os.path.join(defect_data_save_dir_path, "OK")
        defect_ng_data_save_dir_path = os.path.join(defect_data_save_dir_path, "NG")

        # Remove all old cropped images
        if is_removed:
            if os.path.isdir(defect_ok_data_save_dir_path):
                shutil.rmtree(defect_ok_data_save_dir_path)
            if os.path.isdir(defect_ng_data_save_dir_path):
                shutil.rmtree(defect_ng_data_save_dir_path)

        Path(defect_ok_data_save_dir_path).mkdir(parents=True, exist_ok=True)
        Path(defect_ng_data_save_dir_path).mkdir(parents=True, exist_ok=True)

        return defect_ok_data_save_dir_path, defect_ng_data_save_dir_path

    def get_difference_img(self, original_img, generated_img, is_preprocessed, is_boundary_mask=False):
        grid_mask = self.get_grid_mask(is_boundary_mask=is_boundary_mask, resize_rate=self.resize_rate)

        difference_img = original_img - generated_img
        difference_img = difference_img * grid_mask

        # if is_preprocessed:
        #     difference_img = self.preprocess_difference_img(difference_img=difference_img, mask_img=self.grid_mask)


        """
        여기서 안 썼음.
        """
        # difference_cut_img = self.get_cut_difference_img(difference_original_img=difference_img)

        return difference_img

    def get_important_pixel_coord_list(self, difference_original_img, n=10):
        difference_img = difference_original_img.copy()
        coord_list = list()

        for i in range(n):
            defect_coord = self.get_defect_coord(difference_img)
            coord_list.append(defect_coord)
            difference_img[defect_coord] = 0

        return coord_list

    def get_selected_pixel_img(self, difference_img, coord_list, img_size=256):
        output_img = np.zeros((img_size, img_size))

        for row, col in coord_list:
            pixel_value = difference_img[row, col]
            output_img[row, col] = pixel_value

        return output_img

    def get_defect_img(self, difference_img, cut_size=32):
        defect_coord = self.get_defect_coord(difference_img)
        defect_part_img = self.get_cut_defect_img(difference_img, defect_coord=defect_coord, cut_size=cut_size)

        return defect_part_img

    def get_important_pixel_img(self, difference_img, n=100, img_size=256):
        important_pixel_coord_list = self.get_important_pixel_coord_list(difference_original_img=difference_img, n=n)
        important_pixel_img = self.get_selected_pixel_img(
            difference_img=difference_img,
            coord_list=important_pixel_coord_list,
            img_size=img_size
        )

        return important_pixel_img

    def generate(self, serial_number_list, original_img_list, generated_img_list, defect_category_list, bbox_list, dataset_name, is_difference_img_preprocessed, is_removed=True, properties=None):
        ok_data_save_dir_path, ng_data_save_dir_path = self._create_save_directory(directory_name=dataset_name, is_removed=is_removed)

        serial_number_list = list(map(lambda serial_number: serial_number.split("_")[0], serial_number_list))
        original_img_path_list = list()


        for i, (serial_number, original_img, generated_img, defect_category, bbox) in enumerate(zip(serial_number_list, original_img_list, generated_img_list, defect_category_list, bbox_list)):
            difference_img = self.get_difference_img(
                original_img=original_img,
                generated_img=generated_img,
                is_preprocessed=is_difference_img_preprocessed,
                is_boundary_mask=False
            )
            save_img = None

            # Generate label.csv


            # Controller with save category
            if dataset_name == "difference_img":
                pos_difference_img = np.where(difference_img >= 0, difference_img, 0)
                neg_difference_img = np.negative(np.where(difference_img <= 0, difference_img, 0))
                abs_difference_img = np.abs(difference_img)

                # pos_difference_img = pos_difference_img.astype(np.uint8)
                # neg_difference_img = neg_difference_img.astype(np.uint8)

                # Save
                if defect_category[0] == 1:
                    # Original image
                    original_save_path = os.path.join(ok_data_save_dir_path, f"{serial_number}_{i}_{0}_o.npy")
                    np.save(original_save_path, original_img)

                    # Generated image
                    generated_save_path = os.path.join(ok_data_save_dir_path, f"{serial_number}_{i}_{0}_g.npy")
                    np.save(generated_save_path, generated_img)

                    # Positive image
                    pos_save_path = os.path.join(ok_data_save_dir_path, f"{serial_number}_{i}_{0}_p.npy")
                    np.save(pos_save_path, pos_difference_img)
                    # cv2.imwrite(pos_save_path, pos_difference_img)

                    # Negative image
                    neg_save_path = os.path.join(ok_data_save_dir_path, f"{serial_number}_{i}_{0}_n.npy")
                    np.save(neg_save_path, neg_difference_img)
                    # cv2.imwrite(neg_save_path, neg_difference_img)

                    # Absolute image
                    abs_save_path = os.path.join(ok_data_save_dir_path, f"{serial_number}_{i}_{0}_a.npy")
                    np.save(abs_save_path, abs_difference_img)
                    # cv2.imwrite(abs_save_path, abs_difference_img)
                else:
                    for category_idx, defect in enumerate(defect_category):
                        if defect == 1:
                            # Original image
                            original_save_path = os.path.join(ng_data_save_dir_path, f"{serial_number}_{i}_{category_idx}_o.npy")
                            np.save(original_save_path, original_img)

                            # Generated image
                            generated_save_path = os.path.join(ng_data_save_dir_path, f"{serial_number}_{i}_{category_idx}_g.npy")
                            np.save(generated_save_path, generated_img)

                            # Positive image
                            pos_save_path = os.path.join(ng_data_save_dir_path, f"{serial_number}_{i}_{category_idx}_p.npy")
                            np.save(pos_save_path, pos_difference_img)
                            # cv2.imwrite(pos_save_path, pos_difference_img)

                            # Negative image
                            neg_save_path = os.path.join(ng_data_save_dir_path, f"{serial_number}_{i}_{category_idx}_n.npy")
                            np.save(neg_save_path, neg_difference_img)
                            # cv2.imwrite(neg_save_path, neg_difference_img)

                            # Absolute image
                            abs_save_path = os.path.join(ng_data_save_dir_path, f"{serial_number}_{i}_{category_idx}_a.npy")
                            np.save(abs_save_path, abs_difference_img)
                            # cv2.imwrite(abs_save_path, abs_difference_img)
                        else:
                            continue

            else:
                if dataset_name == "defect_part":
                    """
                    * properties
                    cut_size=32
                    """
                    save_img = self.get_defect_img(difference_img=difference_img, **properties)
                    save_img = save_img.astype(np.uint8)
                elif dataset_name == "important_pixel":
                    """
                    * properties
                    n=100
                    img_size=256
                    """
                    save_img = self.get_important_pixel_img(difference_img=difference_img, **properties)
                    save_img = save_img.astype(np.uint8)

                # Save
                if defect_category[0] == 1:
                    save_path = os.path.join(ok_data_save_dir_path, f"{serial_number}_{i}_{0}.png")
                    cv2.imwrite(save_path, save_img)
                else:
                    for category_idx, defect in enumerate(defect_category):
                        if defect == 1:
                            save_path = os.path.join(ng_data_save_dir_path, f"{serial_number}_{i}_{category_idx}.png")
                            cv2.imwrite(save_path, save_img)
                        else:
                            continue
