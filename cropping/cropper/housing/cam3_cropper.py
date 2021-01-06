import os

from cropping.cam_properties import cam3_properties as c3p
from cropping.cropper.base import HousingCropper
from cropping.cropper.crop_helper import RingCropHelper, CircleCropHelper
from cropping.cropper.figure import *
from domain.image import Coordinate, BoundingBox, EOPImage, Defect
from properties import CROPPING_PROPERTIES


class HousingCam3Cropper(HousingCropper):

    AREA = c3p.CAM3_AREA
    GRID_NUM = 17
    MASK_IMG_PATH = CROPPING_PROPERTIES.HOUSING_CAM3_MASK_IMG_PATH
    BOUNDARY_MASK_IMG_PATH = CROPPING_PROPERTIES.HOUSING_CAM3_BOUNDARY_MASK_IMG_PATH
    AREA_MASK_IMG_DIR_PATH = CROPPING_PROPERTIES.HOUSING_CAM3_AREA_MASK_IMG_HOME_DIR_PATH

    def __init__(self, img_object: EOPImage, defect_list: list, is_cropped_with_mask=True):
        super(HousingCam3Cropper, self).__init__(img_object, defect_list)
        self.mask = self.get_mask(mask_img_path=self.MASK_IMG_PATH)
        self.boundary_mask = self.get_mask(mask_img_path=self.BOUNDARY_MASK_IMG_PATH)

        if is_cropped_with_mask:
            self._crop_with_mask()

    @classmethod
    def get_mask(cls, mask_img_path):
        return HousingCropper.get_mask(mask_img_path=mask_img_path)

    @classmethod
    def get_area_with_coord(cls, coord: Coordinate):
        area_list = list()

        """ A """
        crop_variable = c3p.CROP_A_VARIABLE
        if coord.is_in_margin(img_center_coord=crop_variable.CENTER_COORD,
                              margin_min=crop_variable.SMALL_RAD,
                              margin_max=crop_variable.BIG_RAD):
            area_list.append(c3p.CAM3_AREA.A)

        """ C_inner_ring_inside """
        crop_variable = c3p.CROP_C_INNER_RING_INSIDE_ENTIRE_VARIABLE
        if coord.is_in_margin(img_center_coord=crop_variable.CENTER_COORD,
                              margin_min=0,
                              margin_max=crop_variable.RAD):
            area_list.append(c3p.CAM3_AREA.C_INNER_RING_INSIDE)

        """ C_inner_ring_outside """
        crop_variable = c3p.CROP_C_INNER_RING_OUTSIDE_ENTIRE_VARIABLE
        if coord.is_in_margin(img_center_coord=crop_variable.CENTER_COORD,
                              margin_min=crop_variable.SMALL_RAD,
                              margin_max=crop_variable.BIG_RAD + 100):  # 안전하게 + 100 (경계선에 걸친 coord 존재)
            area_list.append(c3p.CAM3_AREA.C_INNER_RING_OUTSIDE)

        """ C_outer """
        crop_variable = c3p.CROP_C_OUTER_ENTIRE_VARIABLE
        if coord.is_in_margin(img_center_coord=crop_variable.CENTER_COORD,
                              margin_min=crop_variable.RAD,
                              margin_max=2000):
            area_list.append(c3p.CAM3_AREA.C_OUTER)

        """ D """
        crop_variable = c3p.CROP_D_RING_ENTIRE_VARIABLE
        if coord.is_in_margin(img_center_coord=crop_variable.CENTER_COORD,
                              margin_min=crop_variable.SMALL_RAD - 100,  # 안전하게 - 100 (경계선에 걸친 coord 존재)
                              margin_max=crop_variable.BIG_RAD):
            area_list.append(c3p.CAM3_AREA.D)

        if area_list:
            return area_list[0]
        else:
            return None

    def crop(self, **properties):
        pass


class HousingCam3CropperA(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperA, self).__init__(img_object, defect_list)

    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate, angle: int):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)
        crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)

        # Checking defect
        for defect in self.defect_list:
            # Crop A 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=small_rad, margin_max=big_rad):
                for crop_rotate_ring_piece in crop_rotate_ring_piece_list:
                    ring_crop_helper.relocate_defect(
                        img_object=crop_rotate_ring_piece,
                        defect=defect
                    )

        return crop_rotate_ring_piece_list


class HousingCam3CropperARingEntire(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperARingEntire, self).__init__(img_object, defect_list)

    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)

        ring_img_object = ring_crop_helper.cropped_img_object

        # Checking defect
        for defect in self.defect_list:
            # Crop A 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=small_rad,
                                               margin_max=big_rad):
                ring_crop_helper.relocate_defect(
                    img_object=ring_img_object,
                    defect=defect
                )

        return [ring_img_object]


class HousingCam3CropperACDRing(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperACDRing, self).__init__(img_object, defect_list)

    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate, angle: int):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)
        crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)

        # Checking defect
        for defect in self.defect_list:
            # Crop C inner ring 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=small_rad, margin_max=big_rad):
                for crop_rotate_ring_piece in crop_rotate_ring_piece_list:
                    ring_crop_helper.relocate_defect(
                        img_object=crop_rotate_ring_piece,
                        defect=defect
                    )

        return crop_rotate_ring_piece_list


class HousingCam3CropperCInnerRing(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperCInnerRing, self).__init__(img_object, defect_list)

    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate, angle: int):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)
        crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)

        # Checking defect
        for defect in self.defect_list:
            # Crop C inner ring 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=small_rad, margin_max=big_rad):
                for crop_rotate_ring_piece in crop_rotate_ring_piece_list:
                    ring_crop_helper.relocate_defect(
                        img_object=crop_rotate_ring_piece,
                        defect=defect
                    )

        return crop_rotate_ring_piece_list


class HousingCam3CropperCInnerRingEntire(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperCInnerRingEntire, self).__init__(img_object, defect_list)

    def crop(self, rad: int, center_coord: Coordinate):
        circle_crop_helper = CircleCropHelper(img_object=self.img_object, center_coord=center_coord, rad=rad)
        circle_crop_helper.crop_circle(inverse=False)
        cropped_img_object = circle_crop_helper.cropped_img_object

        # Checking defect
        for defect in self.defect_list:
            # Crop C inner ring 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=0, margin_max=rad):
                circle_crop_helper.relocate_defect(
                    img_object=cropped_img_object,
                    defect=defect
                )

        return [cropped_img_object]


class HousingCam3CropperCInnerRingInsideEntire(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperCInnerRingInsideEntire, self).__init__(img_object, defect_list)

    def crop(self, rad: int, center_coord: Coordinate):
        circle_crop_helper = CircleCropHelper(img_object=self.img_object, center_coord=center_coord, rad=rad)
        circle_crop_helper.crop_circle(inverse=False)
        cropped_img_object = circle_crop_helper.cropped_img_object

        # Checking defect
        for defect in self.defect_list:
            # Crop C inner ring 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=0, margin_max=rad):
                circle_crop_helper.relocate_defect(
                    img_object=cropped_img_object,
                    defect=defect
                )

        return [cropped_img_object]


class HousingCam3CropperCInnerRingOutsideEntire(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperCInnerRingOutsideEntire, self).__init__(img_object, defect_list)

    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)

        cropped_img_object = ring_crop_helper.cropped_img_object

        # Checking defect
        for defect in self.defect_list:
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=small_rad, margin_max=big_rad):
                ring_crop_helper.relocate_defect(
                    img_object=cropped_img_object,
                    defect=defect
                )

        return [cropped_img_object]


class HousingCam3CropperDRing(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperDRing, self).__init__(img_object, defect_list)

    """
    @TODO
    이 부분 결함이 비대칭 이미지에도 매칭될 수 있게끔 수정하기
    """
    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate, angle: int):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)
        crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)

        # Checking defect
        for defect in self.defect_list:
            # Crop C inner ring 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=small_rad, margin_max=big_rad):
                for crop_rotate_ring_piece in crop_rotate_ring_piece_list:
                    ring_crop_helper.relocate_defect(
                        img_object=crop_rotate_ring_piece,
                        defect=defect
                    )

        return crop_rotate_ring_piece_list


class HousingCam3CropperDRingEntire(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperDRingEntire, self).__init__(img_object, defect_list)

    """
    @TODO
    이 부분 결함이 비대칭 이미지에도 매칭될 수 있게끔 수정하기
    """
    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)
        cropped_img_object = ring_crop_helper.cropped_img_object

        # Checking defect
        for defect in self.defect_list:
            # Crop A 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=small_rad,
                                               margin_max=big_rad):
                ring_crop_helper.relocate_defect(
                    img_object=cropped_img_object,
                    defect=defect
                )

        return [cropped_img_object]

class HousingCam3CropperCOuterRing(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperCOuterRing, self).__init__(img_object, defect_list)

    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate, angle: int,
                          index_list: list = None):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)
        crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)

        if index_list is None:
            index_list = [i for i in range(len(crop_rotate_ring_piece_list))]

        # Checking defect
        for defect in self.defect_list:
            # Crop D 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=small_rad, margin_max=big_rad):
                for crop_rotate_ring_piece in crop_rotate_ring_piece_list:
                    ring_crop_helper.relocate_defect(
                        img_object=crop_rotate_ring_piece,
                        defect=defect
                    )

        # 해당하는 index 의 이미지만 반환
        return [crop_rotate_ring_piece_list[i] for i in index_list]


class HousingCam3CropperCOuterEntire(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperCOuterEntire, self).__init__(img_object, defect_list)

    def crop(self, rad: int, center_coord: Coordinate):
        circle_crop_helper = CircleCropHelper(
            img_object=self.img_object,
            center_coord=center_coord,
            rad=rad
        )
        circle_crop_helper.crop_circle(inverse=True)
        cropped_img_object = circle_crop_helper.cropped_img_object

        # Checking defect
        for defect in self.defect_list:
            defect: Defect

            # Crop D 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=rad, margin_max=2200):
                circle_crop_helper.relocate_defect(
                    img_object=cropped_img_object,
                    defect=defect
                )

        return [cropped_img_object]


class HousingCam3CropperCRectangleWithRing(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperCRectangleWithRing, self).__init__(img_object, defect_list)

    def crop(self, bbox_list: list, angle_list: list, boundary_circle_center_coord: Coordinate,
                                   boundary_circle_rad: int):
        """
        @TODO
        이 부분 뺄 것
        """
        circle_crop_helper = CircleCropHelper(img_object=self.img_object, center_coord=Coordinate(1205, 1062), rad=757)
        circle_crop_helper.crop_circle(inverse=True)
        outer_circle_img_object = circle_crop_helper.cropped_img_object

        rectangle_crop_helper = RectangleCropHelper(img_object=outer_circle_img_object)
        rectangle_img_list = list()

        for bbox, angle in zip(bbox_list, angle_list):
            bbox: BoundingBox
            rectangle_crop_helper.crop_rectangle(bbox=bbox)
            cropped_img_object = rectangle_crop_helper.cropped_img_object

            # Checking Defect
            for defect in self.defect_list:
                defect: Defect

                # Crop D 에서 defect 가 있을 때
                if defect.is_defect_in_cropped_img(img_center_coord=boundary_circle_center_coord,
                                                   margin_min=boundary_circle_rad, margin_max=2200):
                    if defect.is_defect_in_bounding_box(bbox=bbox):
                        rectangle_crop_helper.relocate_defect(
                            img_object=cropped_img_object,
                            defect=defect
                        )

            # Rotate with angle
            cropped_img_object_width = cropped_img_object.img.shape[1]
            cropped_img_object_height = cropped_img_object.img.shape[0]
            cropped_img_object_center_coord = Coordinate(
                x=cropped_img_object_width // 2,
                y=cropped_img_object_height // 2
            )

            rotate_cropped_img = cv2.warpAffine(
                cropped_img_object.img,
                cv2.getRotationMatrix2D(cropped_img_object_center_coord.value, angle, scale=1),
                (cropped_img_object_width, cropped_img_object_height)
            )

            cropped_img_object.img = rotate_cropped_img

            rectangle_img_list.append(cropped_img_object)

        return rectangle_img_list


class HousingCam3CropperCRectangleTop(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperCRectangleTop, self).__init__(img_object, defect_list)

    def crop(self, bbox_list: list, boundary_circle_center_coord: Coordinate, boundary_circle_rad: int,
                             flip_index: int = 0):
        """
        @TODO
        이 부분 뺄 것
        """
        circle_crop_helper = CircleCropHelper(img_object=self.img_object, center_coord=Coordinate(1205, 1062), rad=757)
        circle_crop_helper.crop_circle(inverse=True)
        outer_circle_img_object = circle_crop_helper.cropped_img_object

        rectangle_crop_helper = RectangleCropHelper(img_object=outer_circle_img_object)
        rectangle_img_list = list()

        for idx, bbox in enumerate(bbox_list):
            rectangle_crop_helper.crop_rectangle(bbox=bbox)
            cropped_img_object = rectangle_crop_helper.cropped_img_object

            # Checking defect
            for defect in self.defect_list:
                defect: Defect

                # Crop C 에서 defect 가 있을 때
                if defect.is_defect_in_cropped_img(img_center_coord=boundary_circle_center_coord,
                                                   margin_min=boundary_circle_rad, margin_max=2200):
                    if defect.is_defect_in_bounding_box(bbox=bbox):
                        rectangle_crop_helper.relocate_defect(
                            img_object=cropped_img_object,
                            defect=defect
                        )

            # Flip the image
            if idx == flip_index:
                flipped_cropped_img = cv2.flip(cropped_img_object.img, 1)  # 1 은 좌우 반전
                cropped_img_object.img = flipped_cropped_img

            rectangle_img_list.append(cropped_img_object)

        return rectangle_img_list


class HousingCam3CropperCRectangleRightDown(HousingCam3Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperCRectangleRightDown, self).__init__(img_object, defect_list)

    def crop(self, bbox: BoundingBox, boundary_circle_center_coord: Coordinate,
                                    boundary_circle_rad: int):
        """
        @TODO
        이 부분 뺄 것
        """
        circle_crop_helper = CircleCropHelper(img_object=self.img_object, center_coord=Coordinate(1205, 1062), rad=757)
        circle_crop_helper.crop_circle(inverse=True)
        outer_circle_img_object = circle_crop_helper.cropped_img_object

        rectangle_crop_helper = RectangleCropHelper(img_object=outer_circle_img_object)
        rectangle_img_list = list()

        rectangle_crop_helper.crop_rectangle(bbox=bbox)
        cropped_img_object = rectangle_crop_helper.cropped_img_object

        # Checking defect
        for defect in self.defect_list:
            defect: Defect

            # Crop C 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=boundary_circle_center_coord,
                                               margin_min=boundary_circle_rad, margin_max=2200):
                if defect.is_defect_in_bounding_box(bbox=bbox):
                    rectangle_crop_helper.relocate_defect(
                        img_object=cropped_img_object,
                        defect=defect
                    )
        rectangle_img_list.append(cropped_img_object)

        return rectangle_img_list


class HousingCam3CropperGrid(HousingCam3Cropper):

    MASK_IMG = HousingCam3Cropper.get_mask(mask_img_path=HousingCam3Cropper.MASK_IMG_PATH)
    BOUNDARY_MASK_IMG = HousingCam3Cropper.get_mask(mask_img_path=HousingCam3Cropper.BOUNDARY_MASK_IMG_PATH)

    AREA_MASK_IMG_DICT = {
        area.name: {f"grid_{i + 1}": HousingCam3Cropper.get_mask(
            mask_img_path=os.path.join(HousingCam3Cropper.AREA_MASK_IMG_DIR_PATH, area.name, f"grid_{i + 1}.png")
        ) for i in range(HousingCam3Cropper.GRID_NUM)}
        for area in c3p.CAM3_AREA.get_list()
    }

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam3CropperGrid, self).__init__(img_object, defect_list)
        self.cropper_square = CropperSquare(self.img_object, self.defect_list)

    def crop(self, bbox_list):
        return self.cropper_square.crop(bbox_list=bbox_list)

    @classmethod
    def crop_grid(cls, bbox_list, is_boundary_mask):
        if is_boundary_mask:
            # grid_mask_img_path = HousingCam3Cropper.BOUNDARY_MASK_IMG_PATH
            mask = cls.BOUNDARY_MASK_IMG
        else:
            # grid_mask_img_path = HousingCam3Cropper.MASK_IMG_PATH
            mask = cls.MASK_IMG

        # mask = cls.get_mask(mask_img_path=grid_mask_img_path)  # @TODO masking 부분 수정해야 할 것
        # import matplotlib.pyplot as plt
        # plt.imsave("./test.jpg", mask)
        cropper = cls(img_object=EOPImage(img=mask), defect_list=[])

        return cropper.crop(bbox_list=bbox_list)
