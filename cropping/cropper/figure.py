from cropping.cropper.base import CropperBase
from cropping.cropper.crop_helper import RectangleCropHelper
from cropping.image_data import *


class CropperSquare(CropperBase):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(CropperSquare, self).__init__(img_object, defect_list)

    def crop(self, bbox_list):
        # If bbox exists only one
        if isinstance(bbox_list, BoundingBox):
            bbox_list = [bbox_list]

        rectangle_crop_helper = RectangleCropHelper(img_object=self.img_object)
        rectangle_img_list = list()

        for bbox in bbox_list:
            rectangle_crop_helper.crop_rectangle(bbox=bbox)
            cropped_img_object = rectangle_crop_helper.cropped_img_object

            # Checking defect
            for defect in self.defect_list:
                defect: Defect

                if defect.is_defect_in_bounding_box(bbox=bbox):
                    rectangle_crop_helper.relocate_defect(
                        img_object=cropped_img_object,
                        defect=defect
                    )

            rectangle_img_list.append(cropped_img_object)

        return rectangle_img_list
