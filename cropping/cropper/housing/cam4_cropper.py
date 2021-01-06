from cropping.cam_properties import cam4_properties as c4p
from cropping.cropper.base import HousingCropper
from cropping.cropper.crop_helper import RectangleCropHelper
from domain.image import BoundingBox, EOPImage, Defect
from properties import CROPPING_PROPERTIES


class HousingCam4Cropper(HousingCropper):

    AREA = c4p.CAM4_AREA
    MASK_IMG_PATH = CROPPING_PROPERTIES.HOUSING_CAM4_MASK_IMG_PATH
    BOUNDARY_MASK_IMG_PATH = CROPPING_PROPERTIES.HOUSING_CAM4_BOUNDARY_MASK_IMG_PATH
    AREA_MASK_IMG_DIR_PATH = CROPPING_PROPERTIES.HOUSING_CAM4_AREA_MASK_IMG_HOME_DIR_PATH
    AREA_C_ENTIRE_MASK_IMG_PATH = CROPPING_PROPERTIES.HOUSING_CAM4_AREA_C_ENTIRE_MASK_IMG_PATH

    def __init__(self, img_object: EOPImage, defect_list: list, is_cropped_with_mask=True):
        super(HousingCam4Cropper, self).__init__(img_object, defect_list)
        self.mask = self.get_mask(mask_img_path=self.MASK_IMG_PATH)
        self.boundary_mask = self.get_mask(mask_img_path=self.BOUNDARY_MASK_IMG_PATH)

        if is_cropped_with_mask:
            self._crop_with_mask()
        # self._crop_empty_space(self.img_object, self.defect_list, y_crop_st=self.y_crop_st, y_crop_ed=self.y_crop_ed)

    def crop(self, **properties):
        pass


class HousingCam4CropperC(HousingCam4Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam4CropperC, self).__init__(img_object, defect_list)
        self.first_crop_x_st = 0
        self.crop_size = 0
        self.first_crop_y_ed = 0
        self.crop_empty_space_y_gap = 0
        self.crop_y_gap = 0
        self.crop_x_ed = self.first_crop_x_st + self.crop_size
        self.crop_y_ed = self.first_crop_y_ed + self.crop_empty_space_y_gap

    def _crop_up_part(self, img_object: EOPImage):
        rectangle_crop_helper = RectangleCropHelper(img_object=img_object)

        # Crop
        crop_bbox = BoundingBox(
            x_st=self.first_crop_x_st,
            x_ed=self.crop_x_ed,
            y_st=0,
            y_ed=self.first_crop_y_ed
        )
        rectangle_crop_helper.crop_rectangle(bbox=crop_bbox)
        cropped_img_object = rectangle_crop_helper.cropped_img_object

        # Checking defect
        for defect in self.defect_list:
            defect: Defect

            # 해당 bbox 에 defect 가 있을 때
            if defect.is_defect_in_bounding_box(bbox=crop_bbox):
                rectangle_crop_helper.relocate_defect(
                    img_object=cropped_img_object,
                    defect=defect
                )
        return cropped_img_object

    def _crop_down_part(self, img_object: EOPImage):
        rectangle_crop_helper = RectangleCropHelper(img_object=img_object)

        # Crop
        crop_bbox = BoundingBox(
            x_st=self.first_crop_x_st,
            x_ed=self.crop_x_ed,
            y_st=self.crop_y_ed,
            y_ed=img_object.abst_bbox.y_ed
        )
        rectangle_crop_helper.crop_rectangle(bbox=crop_bbox)
        cropped_img_object = rectangle_crop_helper.cropped_img_object

        # Checking defect
        for defect in self.defect_list:
            defect: Defect

            # 해당 bbox 에 defect 가 있을 때
            if defect.is_defect_in_bounding_box(bbox=crop_bbox):
                rectangle_crop_helper.relocate_defect(
                    img_object=cropped_img_object,
                    defect=defect
                )
        return cropped_img_object

    def _divide_squares(self, img_object: EOPImage):
        rectangle_crop_helper = RectangleCropHelper(img_object=img_object.base_img)
        rectangle_img_list = list()
        height, width = img_object.img.shape

        square_bbox = BoundingBox(
            x_st=img_object.abst_bbox.x_st,
            x_ed=img_object.abst_bbox.x_ed,
            y_st=img_object.abst_bbox.y_st,
            y_ed=img_object.abst_bbox.y_st + self.crop_size
        )

        update_height_value = self.crop_y_gap
        current_y_st = square_bbox.y_st
        current_y_ed = square_bbox.y_ed

        is_last_square = False

        while not is_last_square:
            rectangle_crop_helper.crop_rectangle(bbox=square_bbox)
            cropped_img_object = rectangle_crop_helper.cropped_img_object

            # Checking defect:
            for defect in self.defect_list:
                defect: Defect

                if defect.is_defect_in_bounding_box(bbox=square_bbox):
                    rectangle_crop_helper.relocate_defect(
                        img_object=cropped_img_object,
                        defect=defect
                    )

            rectangle_img_list.append(cropped_img_object)

            # Move bounding box
            current_y_st += update_height_value
            current_y_ed += update_height_value

            square_bbox = BoundingBox(
                x_st=img_object.abst_bbox.x_st,
                x_ed=img_object.abst_bbox.x_ed,
                y_st=current_y_st,
                y_ed=current_y_ed
            )

            if img_object.abst_bbox.y_ed < current_y_ed:
                is_last_square = True

        ##########################################################################################
        # Append last square
        square_bbox = BoundingBox(
            x_st=img_object.abst_bbox.x_st,
            x_ed=img_object.abst_bbox.x_ed,
            y_st=img_object.abst_bbox.y_ed - self.crop_size,
            y_ed=img_object.abst_bbox.y_ed
        )

        rectangle_crop_helper.crop_rectangle(bbox=square_bbox)
        cropped_img_object = rectangle_crop_helper.cropped_img_object

        # Checking defect:
        for defect in self.defect_list:
            defect: Defect

            if defect.is_defect_in_bounding_box(bbox=square_bbox):
                rectangle_crop_helper.relocate_defect(
                    img_object=cropped_img_object,
                    defect=defect
                )

        rectangle_img_list.append(cropped_img_object)

        return rectangle_img_list

    def crop(self, first_crop_x_st, crop_size, first_crop_y_ed, crop_empty_space_y_gap, crop_y_gap):
        self.first_crop_x_st = first_crop_x_st
        self.crop_size = crop_size
        self.crop_x_ed = self.first_crop_x_st + self.crop_size

        self.first_crop_y_ed = first_crop_y_ed
        self.crop_empty_space_y_gap = crop_empty_space_y_gap
        self.crop_y_ed = self.first_crop_y_ed + self.crop_empty_space_y_gap

        self.crop_y_gap = int(crop_y_gap)

        up_img_object = self._crop_up_part(img_object=self.img_object)
        down_img_object = self._crop_down_part(img_object=self.img_object)

        up_square_img_list = self._divide_squares(img_object=up_img_object)
        down_square_img_list = self._divide_squares(img_object=down_img_object)

        return up_square_img_list + down_square_img_list


class HousingCam4CropperCEntire(HousingCam4Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam4CropperCEntire, self).__init__(img_object, defect_list)

    def crop(self, bbox):
        rectangle_img_list = list()
        rectangle_crop_helper = RectangleCropHelper(img_object=self.img_object)

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

        rectangle_img_list.append(cropped_img_object)

        return rectangle_img_list


class HousingCam4CropperGrid(HousingCam4CropperC):

    MASK_IMG = HousingCam4Cropper.get_mask(mask_img_path=HousingCam4Cropper.MASK_IMG_PATH)
    BOUNDARY_MASK_IMG = HousingCam4Cropper.get_mask(mask_img_path=HousingCam4Cropper.BOUNDARY_MASK_IMG_PATH)

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam4CropperGrid, self).__init__(img_object, defect_list)

    @classmethod
    def crop_grid(cls, first_crop_x_st, crop_size, first_crop_y_ed, crop_empty_space_y_gap, crop_y_gap, is_boundary_mask):
        if is_boundary_mask:
            mask = cls.BOUNDARY_MASK_IMG_PATH
        else:
            mask = cls.MASK_IMG

        cropper = cls(img_object=EOPImage(img=mask), defect_list=[])

        return cropper.crop(
            first_crop_x_st=first_crop_x_st,
            crop_size=crop_size,
            first_crop_y_ed=first_crop_y_ed,
            crop_empty_space_y_gap=crop_empty_space_y_gap,
            crop_y_gap=crop_y_gap
        )
