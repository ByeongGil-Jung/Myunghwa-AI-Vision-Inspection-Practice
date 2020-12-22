from cropping.cropper.base import HousingCropper
from cropping.cropper.crop_helper import RectangleCropHelper
from domain.image import EOPImage, Defect
from properties import CROPPING_PROPERTIES

import cv2


class HousingCam2Cropper(HousingCropper):

    def __init__(self, img_object: EOPImage, defect_list: list, is_cropped_with_mask=True):
        super(HousingCam2Cropper, self).__init__(img_object, defect_list)
        self._set_mask(mask_img_path=CROPPING_PROPERTIES.HOUSING_CAM2_MASK_IMG_PATH)

        if is_cropped_with_mask:
            self._crop_with_mask()

    def crop(self, **properties):
        pass


class HousingCam2CropperC(HousingCam2Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam2CropperC, self).__init__(img_object, defect_list)

    def crop(self, bbox_list: list, flip_index: int = 0):
        rectangle_img_list = list()
        rectangle_crop_helper = RectangleCropHelper(img_object=self.img_object)

        for idx, bbox in enumerate(bbox_list):
            rectangle_crop_helper.crop_rectangle(bbox=bbox)
            cropped_img_object = rectangle_crop_helper.cropped_img_object

            # Checking defect
            for defect in self.defect_list:
                defect: Defect

                # 해당 bbox 에 defect 가 있을 때
                if defect.is_defect_in_bounding_box(bbox=bbox):
                    rectangle_crop_helper.relocate_defect(
                        img_object=cropped_img_object,
                        defect=defect
                    )

            # Flip the image
            if idx == flip_index:
                flipped_cropped_img = cv2.flip(cropped_img_object.img, 1)
                cropped_img_object.img = flipped_cropped_img

            rectangle_img_list.append(cropped_img_object)

        return rectangle_img_list
