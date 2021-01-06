from cropping.cam_properties import cam2_properties as c2p
from cropping.cropper.base import HousingCropper
from cropping.cropper.crop_helper import RectangleCropHelper
from cropping.cropper.figure import CropperSquare
from domain.image import EOPImage, Defect
from properties import CROPPING_PROPERTIES

import cv2


class HousingCam2Cropper(HousingCropper):

    AREA = c2p.CAM2_AREA
    MASK_IMG_PATH = CROPPING_PROPERTIES.HOUSING_CAM2_MASK_IMG_PATH
    BOUNDARY_MASK_IMG_PATH = CROPPING_PROPERTIES.HOUSING_CAM2_BOUNDARY_MASK_IMG_PATH
    AREA_MASK_IMG_DIR_PATH = CROPPING_PROPERTIES.HOUSING_CAM2_AREA_MASK_IMG_HOME_DIR_PATH
    AREA_C_INNER_MASK_IMG_PATH = CROPPING_PROPERTIES.HOUSING_CAM2_AREA_C_INNER_MASK_IMG_PATH
    AREA_C_OUTER_MASK_IMG_PATH = CROPPING_PROPERTIES.HOUSING_CAM2_AREA_C_OUTER_MASK_IMG_PATH

    def __init__(self, img_object: EOPImage, defect_list: list, is_cropped_with_mask=True):
        super(HousingCam2Cropper, self).__init__(img_object, defect_list)
        self.mask = self.get_mask(mask_img_path=self.MASK_IMG_PATH)
        self.boundary_mask = self.get_mask(mask_img_path=self.BOUNDARY_MASK_IMG_PATH)

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


class HousingCam2CropperGrid(HousingCam2Cropper):

    MASK_IMG = HousingCam2Cropper.get_mask(mask_img_path=HousingCam2Cropper.MASK_IMG_PATH)
    BOUNDARY_MASK_IMG = HousingCam2Cropper.get_mask(mask_img_path=HousingCam2Cropper.BOUNDARY_MASK_IMG_PATH)

    # AREA_MASK_IMG_DICT = {
    #     area.name: {f"grid_{i + 1}": HousingCam2Cropper.get_mask(
    #         mask_img_path=os.path.join(HousingCam2Cropper.AREA_MASK_IMG_DIR_PATH, area.name, f"grid_{i + 1}.png")
    #     ) for i in range(HousingCam2Cropper.GRID_NUM)}
    #     for area in c2p.CAM2_AREA.get_list()
    # }

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam2CropperGrid, self).__init__(img_object, defect_list)
        self.cropper_square = CropperSquare(self.img_object, self.defect_list)

    def crop(self, bbox_list):
        return self.cropper_square.crop(bbox_list=bbox_list)

    @classmethod
    def crop_grid(cls, bbox_list, is_boundary_mask):
        if is_boundary_mask:
            mask = cls.BOUNDARY_MASK_IMG_PATH
        else:
            mask = cls.MASK_IMG

        cropper = cls(img_object=EOPImage(img=mask), defect_list=[])

        return cropper.crop(bbox_list=bbox_list)
