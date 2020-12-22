import sys
from properties import CROPPING_PROPERTIES

sys.path.append(CROPPING_PROPERTIES.HOME_MODULE_PATH)

# from cropping.cropper
from cropping.cropper.base import CropperBase
from domain.image import EOPImage
import numpy as np


class AreaCropper(CropperBase):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(AreaCropper, self).__init__(img_object=img_object, defect_list=defect_list)

    def crop_grid(self, product, cam_number, grid_name):
        zeros_like_entire_img = np.zeros_like(self.img_object.img)

        # if product == "housing":


    def crop(self, **properties):
        pass