from abc import *

import cv2
import numpy as np

from domain.image import EOPImage


class CropperBase(object, metaclass=ABCMeta):

    def __init__(self, img_object: EOPImage, defect_list: list):
        self._set_img_object(img_object)
        self.defect_list = defect_list
        self.mask = None

    @property
    def img_object(self):
        return self._img_object

    @img_object.setter
    def img_object(self, img_object):
        self._set_img_object(img_object=img_object)

    def _set_img_object(self, img_object: EOPImage):
        img_object.base_node = img_object
        self._img_object = img_object

    # def _set_mask(self, mask_img_path):
    #     self.mask = self.get_mask(mask_img_path)

    def _crop_with_mask(self):
        assert self.mask is not None, "Please create the mask, first"
        self.img_object.img *= (self.mask / 255)

    @classmethod
    def get_mask(cls, mask_img_path):
        with open(mask_img_path, "rb") as f:
            mask = bytearray(f.read())
            mask = np.asarray(mask, dtype=np.uint8)

        mask = cv2.imdecode(mask, cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Convert values to 1 and 0
        mask = np.where(mask > 0, 255, mask)

        return mask.astype(np.float32)

    @classmethod
    def get_grid_num(cls, product, cam_number):  # @TODO 이거 properties 로 뺄 것
        grid_num = None

        if product == "housing":
            if cam_number == 1:
                grid_num = 15
            elif cam_number == 2:
                grid_num = 4
            elif cam_number == 3:
                grid_num = 17
            elif cam_number == 4:
                grid_num = 1

        return grid_num



    @abstractmethod
    def crop(self, **properties):
        pass


class HousingCropper(CropperBase, metaclass=ABCMeta):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCropper, self).__init__(img_object, defect_list)

    @abstractmethod
    def crop(self, **properties):
        pass


class CoverCropper(CropperBase, metaclass=ABCMeta):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(CoverCropper, self).__init__(img_object, defect_list)

    @abstractmethod
    def crop(self, **properties):
        pass
