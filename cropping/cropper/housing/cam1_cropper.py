import os

from cropping.cam_properties import cam1_properties as c1p
from cropping.cropper.base import HousingCropper
from cropping.cropper.crop_helper import RingCropHelper, CircleCropHelper
from cropping.cropper.figure import *
from domain.image import Coordinate, BoundingBox, EOPImage, Defect
from properties import CROPPING_PROPERTIES


class HousingCam1Cropper(HousingCropper):

    AREA = c1p.CAM1_AREA
    GRID_NUM = 15
    MASK_IMG_PATH = CROPPING_PROPERTIES.HOUSING_CAM1_MASK_IMG_PATH
    BOUNDARY_MASK_IMG_PATH = CROPPING_PROPERTIES.HOUSING_CAM1_BOUNDARY_MASK_IMG_PATH
    AREA_MASK_IMG_DIR_PATH = CROPPING_PROPERTIES.HOUSING_CAM1_AREA_MASK_IMG_HOME_DIR_PATH

    def __init__(self, img_object: EOPImage, defect_list: list, is_cropped_with_mask=True):
        super(HousingCam1Cropper, self).__init__(img_object, defect_list)
        self.mask = self.get_mask(mask_img_path=self.MASK_IMG_PATH)
        self.boundary_mask = self.get_mask(mask_img_path=self.BOUNDARY_MASK_IMG_PATH)

        if is_cropped_with_mask:
            self._crop_with_mask()

    @classmethod
    def get_mask(cls, mask_img_path):
        if os.path.isfile(mask_img_path):
            return HousingCropper.get_mask(mask_img_path=mask_img_path)
        else:
            print(f"There is not exist mask image file: {mask_img_path}")  # @TODO 여기 logger 로 바꾸기
            return None

    @classmethod
    def get_area_with_coord(cls, coord: Coordinate):
        area_list = list()

        """ A """
        crop_variable = c1p.CROP_A_VARIABLE
        if coord.is_in_margin(img_center_coord=crop_variable.CENTER_COORD,
                              margin_min=crop_variable.SMALL_RAD,
                              margin_max=crop_variable.BIG_RAD):
            area_list.append(c1p.CAM1_AREA.A)

        """ C """
        crop_variable = c1p.CROP_C_ENTIRE_VARIABLE
        if coord.is_in_margin(img_center_coord=crop_variable.CENTER_COORD,
                              margin_min=0,
                              margin_max=crop_variable.RAD):
            area_list.append(c1p.CAM1_AREA.C)

        """ D_inner """
        crop_variable = c1p.CROP_D_INNER_RING_VARIABLE

        # Crop C 바깥에 위치한 경우
        _is_first_circle_outside = coord.is_in_margin(
            img_center_coord=crop_variable.INNER_CIRCLE_CENTER_COORD,
            margin_min=crop_variable.INNER_CIRCLE_RAD,
            margin_max=2000  # Temp
        )
        # Crop D 외곽 반지름 안에 위치한 경우
        _is_second_circle_inside = coord.is_in_margin(
            img_center_coord=crop_variable.OUTER_CIRCLE_CENTER_COORD,
            margin_min=1,  # Temp
            margin_max=crop_variable.OUTER_CIRCLE_RAD
        )
        if _is_first_circle_outside and _is_second_circle_inside:
            area_list.append(c1p.CAM1_AREA.D_INNER)

        """ D_outer """
        crop_variable = c1p.CROP_D_OUTER_ENTIRE_VARIABLE
        if coord.is_in_margin(img_center_coord=crop_variable.CENTER_COORD,
                              margin_min=crop_variable.RAD,
                              margin_max=2000):
            area_list.append(c1p.CAM1_AREA.D_OUTER)

        if area_list:
            return area_list[0]
        else:
            return None

    def crop(self, **properties):
        pass


class HousingCam1CropperARingEntire(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperARingEntire, self).__init__(img_object, defect_list)

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


class HousingCam1CropperDInnerRingEntire(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperDInnerRingEntire, self).__init__(img_object, defect_list)

    def crop(self, inner_circle_rad: int, inner_circle_center_coord: Coordinate, outer_circle_rad: int,
             outer_circle_center_coord: Coordinate, ring_small_rad: int):
        inner_circle_crop_helper = CircleCropHelper(
            img_object=self.img_object,
            center_coord=inner_circle_center_coord,
            rad=inner_circle_rad
        )
        inner_circle_crop_mask = inner_circle_crop_helper.get_circle_mask(inverse=True)
        outer_circle_crop_helper = CircleCropHelper(
            img_object=self.img_object,
            center_coord=outer_circle_center_coord,
            rad=outer_circle_rad
        )
        outer_circle_crop_mask = outer_circle_crop_helper.get_circle_mask(inverse=False)
        crop_D_inner_ring_mask = inner_circle_crop_mask * outer_circle_crop_mask

        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=outer_circle_rad,
                                          small_rad=ring_small_rad)

        # Switching image
        img_original = self.img_object.img
        self.img_object.img = self.img_object.img * crop_D_inner_ring_mask
        ring_crop_helper.crop_ring(center_coord=outer_circle_center_coord)
        ring_img_object = ring_crop_helper.cropped_img_object
        # crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)
        self.img_object.img = img_original

        # Checking defect
        for defect in self.defect_list:
            # Crop C 바깥에 위치한 경우
            is_first_circle_outside = defect.is_defect_in_cropped_img(
                img_center_coord=inner_circle_center_coord,
                margin_min=inner_circle_rad,
                margin_max=2000  # Temp
            )
            # Crop D 외곽 반지름 안에 위치한 경우
            is_second_circle_inside = defect.is_defect_in_cropped_img(
                img_center_coord=outer_circle_center_coord,
                margin_min=1,  # Temp
                margin_max=outer_circle_rad
            )
            if is_first_circle_outside and is_second_circle_inside:
                ring_crop_helper.relocate_defect(
                    img_object=ring_img_object,
                    defect=defect
                )

        return [ring_img_object]


class HousingCam1CropperDOuterEntire(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperDOuterEntire, self).__init__(img_object, defect_list)

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


class HousingCam1CropperA(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperA, self).__init__(img_object, defect_list)

    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate, angle: int):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)
        crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)

        # Checking defect
        for defect in self.defect_list:
            # Crop A 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=small_rad,
                                               margin_max=big_rad):
                for crop_rotate_ring_piece in crop_rotate_ring_piece_list:
                    ring_crop_helper.relocate_defect(
                        img_object=crop_rotate_ring_piece,
                        defect=defect
                    )

        return crop_rotate_ring_piece_list


class HousingCam1CropperCEntire(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperCEntire, self).__init__(img_object, defect_list)

    def crop(self, rad: int, center_coord: Coordinate):
        circle_crop_helper = CircleCropHelper(img_object=self.img_object, center_coord=center_coord, rad=rad)
        circle_crop_helper.crop_circle(inverse=False)
        cropped_img_object = circle_crop_helper.cropped_img_object

        for defect in self.defect_list:
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=0, margin_max=rad):
                circle_crop_helper.relocate_defect(
                    img_object=cropped_img_object,
                    defect=defect
                )

        return [cropped_img_object]


class HousingCam1CropperC(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperC, self).__init__(img_object, defect_list)

    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate, angle: int):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)
        crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)

        # Checking defect
        for defect in self.defect_list:
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=small_rad, margin_max=big_rad):
                for crop_rotate_ring_piece in crop_rotate_ring_piece_list:
                    ring_crop_helper.relocate_defect(
                        img_object=crop_rotate_ring_piece,
                        defect=defect
                    )

        return crop_rotate_ring_piece_list


class HousingCam1CropperDRing(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperDRing, self).__init__(img_object, defect_list)

    def crop(self, inner_circle_rad: int, inner_circle_center_coord: Coordinate,
             outer_circle_rad: int, outer_circle_center_coord: Coordinate, ring_small_rad: int, angle: int):
        inner_circle_crop_helper = CircleCropHelper(
            img_object=self.img_object,
            center_coord=inner_circle_center_coord,
            rad=inner_circle_rad
        )
        inner_circle_crop_mask = inner_circle_crop_helper.get_circle_mask(inverse=True)
        outer_circle_crop_helper = CircleCropHelper(
            img_object=self.img_object,
            center_coord=outer_circle_center_coord,
            rad=outer_circle_rad
        )
        outer_circle_crop_mask = outer_circle_crop_helper.get_circle_mask(inverse=False)
        crop_ring_mask = inner_circle_crop_mask * outer_circle_crop_mask

        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=outer_circle_rad,
                                          small_rad=ring_small_rad)

        # Switching image
        img_original = self.img_object.img
        self.img_object.img = self.img_object.img * crop_ring_mask
        ring_crop_helper.crop_ring(center_coord=outer_circle_center_coord)
        crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)
        self.img_object.img = img_original

        # Checking defect
        for defect in self.defect_list:
            # Crop C 바깥에 위치한 경우
            is_first_circle_outside = defect.is_defect_in_cropped_img(
                img_center_coord=inner_circle_center_coord,
                margin_min=inner_circle_rad,
                margin_max=2000  # Temp
            )
            # Crop D 외곽 반지름 안에 위치한 경우
            is_second_circle_inside = defect.is_defect_in_cropped_img(
                img_center_coord=outer_circle_center_coord,
                margin_min=0,  # Temp
                margin_max=outer_circle_rad
            )
            if is_first_circle_outside and is_second_circle_inside:
                for crop_rotate_ring_piece in crop_rotate_ring_piece_list:
                    ring_crop_helper.relocate_defect(
                        img_object=crop_rotate_ring_piece,
                        defect=defect
                    )

        return crop_rotate_ring_piece_list


class HousingCam1CropperDInnerRing(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperDInnerRing, self).__init__(img_object, defect_list)

    def crop(self, inner_circle_rad: int, inner_circle_center_coord: Coordinate, outer_circle_rad: int,
             outer_circle_center_coord: Coordinate, ring_small_rad: int, angle: int):
        inner_circle_crop_helper = CircleCropHelper(
            img_object=self.img_object,
            center_coord=inner_circle_center_coord,
            rad=inner_circle_rad
        )
        inner_circle_crop_mask = inner_circle_crop_helper.get_circle_mask(inverse=True)
        outer_circle_crop_helper = CircleCropHelper(
            img_object=self.img_object,
            center_coord=outer_circle_center_coord,
            rad=outer_circle_rad
        )
        outer_circle_crop_mask = outer_circle_crop_helper.get_circle_mask(inverse=False)
        crop_ring_mask = inner_circle_crop_mask * outer_circle_crop_mask

        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=outer_circle_rad,
                                          small_rad=ring_small_rad)

        # Switching image
        img_original = self.img_object.img
        self.img_object.img = self.img_object.img * crop_ring_mask
        ring_crop_helper.crop_ring(center_coord=outer_circle_center_coord)
        crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)
        self.img_object.img = img_original

        # Checking defect
        for defect in self.defect_list:
            # Crop C 바깥에 위치한 경우
            is_first_circle_outside = defect.is_defect_in_cropped_img(
                img_center_coord=inner_circle_center_coord,
                margin_min=inner_circle_rad,
                margin_max=2000  # Temp
            )
            # Crop D 외곽 반지름 안에 위치한 경우
            is_second_circle_inside = defect.is_defect_in_cropped_img(
                img_center_coord=outer_circle_center_coord,
                margin_min=1,  # Temp
                margin_max=outer_circle_rad
            )
            if is_first_circle_outside and is_second_circle_inside:
                for crop_rotate_ring_piece in crop_rotate_ring_piece_list:
                    ring_crop_helper.relocate_defect(
                        img_object=crop_rotate_ring_piece,
                        defect=defect
                    )

        return crop_rotate_ring_piece_list


class HousingCam1CropperDOuterRing(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperDOuterRing, self).__init__(img_object, defect_list)

    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate, angle: int,
                          index_list: list = None):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)
        crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)

        if index_list is None:
            index_list = [i for i in range(len(crop_rotate_ring_piece_list))]

        """
        @TODO
        결과값을 return 해서 리스트를 계속 추가하는 것이 어떨까 ?
        add_child 가 아니라 ... 
        """
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


class HousingCam1CropperDOuterRingInside(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperDOuterRingInside, self).__init__(img_object, defect_list)

    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate, angle: int):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)
        crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)

        # Checking defect
        for defect in self.defect_list:
            # Crop A 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=small_rad,
                                               margin_max=big_rad):
                for crop_rotate_ring_piece in crop_rotate_ring_piece_list:
                    ring_crop_helper.relocate_defect(
                        img_object=crop_rotate_ring_piece,
                        defect=defect
                    )

        return crop_rotate_ring_piece_list


class HousingCam1CropperDOuterRingOutside(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperDOuterRingOutside, self).__init__(img_object, defect_list)

    def crop(self, big_rad: int, small_rad: int, center_coord: Coordinate, angle: int, index_list: list=None):
        ring_crop_helper = RingCropHelper(img_object=self.img_object, big_rad=big_rad, small_rad=small_rad)
        ring_crop_helper.crop_ring(center_coord=center_coord)
        crop_rotate_ring_piece_list = ring_crop_helper.get_crop_rotate_ring_piece_list(angle=angle)

        if index_list is None:
            index_list = [i for i in range(len(crop_rotate_ring_piece_list))]

        # Checking defect
        for defect in self.defect_list:
            # Crop A 에서 defect 가 있을 때
            if defect.is_defect_in_cropped_img(img_center_coord=center_coord, margin_min=small_rad,
                                               margin_max=big_rad):
                for crop_rotate_ring_piece in crop_rotate_ring_piece_list:
                    ring_crop_helper.relocate_defect(
                        img_object=crop_rotate_ring_piece,
                        defect=defect
                    )

        # 해당하는 index 의 이미지만 반환
        return [crop_rotate_ring_piece_list[i] for i in index_list]


class HousingCam1CropperDOuterRectangleWithRing(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperDOuterRectangleWithRing, self).__init__(img_object, defect_list)

    def crop(self, bbox_list: list, angle_list: list,
             boundary_circle_center_coord: Coordinate, boundary_circle_rad: int):
        """
        @TODO
        이 부분 뺄 것
        """
        circle_crop_helper = CircleCropHelper(img_object=self.img_object, center_coord=Coordinate(1250, 985), rad=715)
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


class HousingCam1CropperDOuterRectangleTop(HousingCam1Cropper):

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperDOuterRectangleTop, self).__init__(img_object, defect_list)

    def crop(self, bbox_list: list, boundary_circle_center_coord: Coordinate,
             boundary_circle_rad: int, flip_index: int = 0):
        """
        @TODO
        이 부분 뺄 것
        """
        circle_crop_helper = CircleCropHelper(img_object=self.img_object, center_coord=Coordinate(1250, 985), rad=715)
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

                # Crop D 에서 defect 가 있을 때
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


class HousingCam1CropperGrid(HousingCam1Cropper):

    MASK_IMG = HousingCam1Cropper.get_mask(mask_img_path=HousingCam1Cropper.MASK_IMG_PATH)
    BOUNDARY_MASK_IMG = HousingCam1Cropper.get_mask(mask_img_path=HousingCam1Cropper.BOUNDARY_MASK_IMG_PATH)

    AREA_MASK_IMG_DICT = {
        area.name: {f"grid_{i + 1}": HousingCam1Cropper.get_mask(
            mask_img_path=os.path.join(HousingCam1Cropper.AREA_MASK_IMG_DIR_PATH, area.name, f"grid_{i + 1}.png")
        ) for i in range(HousingCam1Cropper.GRID_NUM)}
        for area in c1p.CAM1_AREA.get_list()
    }

    def __init__(self, img_object: EOPImage, defect_list: list):
        super(HousingCam1CropperGrid, self).__init__(img_object, defect_list)
        self.cropper_square = CropperSquare(self.img_object, self.defect_list)

    def crop(self, bbox_list):
        return self.cropper_square.crop(bbox_list=bbox_list)

    @classmethod
    def crop_grid(cls, bbox_list, is_boundary_mask):
        if is_boundary_mask:
            # grid_mask_img_path = HousingCam1Cropper.BOUNDARY_MASK_IMG_PATH
            mask = cls.BOUNDARY_MASK_IMG_PATH
        else:
            # grid_mask_img_path = HousingCam1Cropper.MASK_IMG_PATH
            mask = cls.MASK_IMG

        # mask = cls.get_mask(mask_img_path=grid_mask_img_path)
        # import matplotlib.pyplot as plt
        # plt.imsave("./test.jpg", mask)
        cropper = cls(img_object=EOPImage(img=mask), defect_list=[])

        return cropper.crop(bbox_list=bbox_list)
