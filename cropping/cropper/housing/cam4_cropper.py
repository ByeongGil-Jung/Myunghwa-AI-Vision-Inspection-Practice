from cropping.cropper.base import HousingCropper
from cropping.cropper.crop_helper import RectangleCropHelper
from domain.image import BoundingBox, EOPImage, Defect
from properties import CROPPING_PROPERTIES


class HousingCam4Cropper(HousingCropper):

    def __init__(self, img_object: EOPImage, defect_list: list, is_cropped_with_mask=True):
        super(HousingCam4Cropper, self).__init__(img_object, defect_list)
        self.crop_x_st = 40
        self.crop_size = 145
        self.crop_x_ed = self.crop_x_st + self.crop_size
        self.y_crop_st = 1579
        self.y_crop_gap = 622
        self.y_crop_ed = self.y_crop_st + self.y_crop_gap
        self._set_mask(mask_img_path=CROPPING_PROPERTIES.HOUSING_CAM4_MASK_IMG_PATH)

        if is_cropped_with_mask:
            self._crop_with_mask()
        # self._crop_empty_space(self.img_object, self.defect_list, y_crop_st=self.y_crop_st, y_crop_ed=self.y_crop_ed)

    def crop(self, **properties):
        pass

    def _crop_up_part(self, img_object: EOPImage):
        rectangle_crop_helper = RectangleCropHelper(img_object=img_object)

        # Crop
        crop_bbox = BoundingBox(
            x_st=self.crop_x_st,
            x_ed=self.crop_x_ed,
            y_st=0,
            y_ed=self.y_crop_st
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
            x_st=self.crop_x_st,
            x_ed=self.crop_x_ed,
            y_st=self.y_crop_ed,
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

        update_height_value = int(self.crop_size / 2)
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


class HousingCam4CropperC(HousingCam4Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam4CropperC, self).__init__(img_object, defect_list)

    def crop(self):
        up_img_object = self._crop_up_part(img_object=self.img_object)
        down_img_object = self._crop_down_part(img_object=self.img_object)

        up_square_img_list = self._divide_squares(img_object=up_img_object)
        down_square_img_list = self._divide_squares(img_object=down_img_object)

        return up_square_img_list + down_square_img_list
