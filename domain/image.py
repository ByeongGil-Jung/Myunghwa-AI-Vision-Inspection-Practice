import math
from typing import List

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from domain.base import ImageDomain


class Area(ImageDomain):

    def __init__(self, name):
        super(Area, self).__init__()
        self.name = name
        self.category = self._get_category()

    def __repr__(self):
        return self.name

    def _get_category(self):
        area_name = self.name
        category = ""

        if "A" in area_name:
            category = "A"
        elif "C" in area_name:
            category = "C"
        elif "D" in area_name:
            category = "D"

        return category


class Coordinate(ImageDomain):

    def __init__(self, x: int=None, y: int=None, area: Area =None):
        super(Coordinate, self).__init__()
        self.x = x
        self.y = y
        self.area = area

    def __call__(self, *args, **kwargs):
        return self.x, self.y

    def __repr__(self):
        return f"({self.x}, {self.y})"

    @property
    def value(self) -> tuple:
        return self.x, self.y

    def _get_properties(self, product, cam_number, crop_part):
        from cropping.cam_properties import cam1_properties as c1p
        from cropping.cam_properties import cam2_properties as c2p
        from cropping.cam_properties import cam3_properties as c3p
        from cropping.cam_properties import cam4_properties as c4p

        properties = None

        if product == "housing":
            if cam_number == 1:
                if crop_part == "grid_1":
                    properties = c1p.CROP_GRID_1_VARIABLE
                elif crop_part == "grid_2":
                    properties = c1p.CROP_GRID_2_VARIABLE
                elif crop_part == "grid_3":
                    properties = c1p.CROP_GRID_3_VARIABLE
                elif crop_part == "grid_4":
                    properties = c1p.CROP_GRID_4_VARIABLE
                elif crop_part == "grid_5":
                    properties = c1p.CROP_GRID_5_VARIABLE
                elif crop_part == "grid_6":
                    properties = c1p.CROP_GRID_6_VARIABLE
                elif crop_part == "grid_7":
                    properties = c1p.CROP_GRID_7_VARIABLE
                elif crop_part == "grid_8":
                    properties = c1p.CROP_GRID_8_VARIABLE
                elif crop_part == "grid_9":
                    properties = c1p.CROP_GRID_9_VARIABLE
                elif crop_part == "grid_10":
                    properties = c1p.CROP_GRID_10_VARIABLE
                elif crop_part == "grid_11":
                    properties = c1p.CROP_GRID_11_VARIABLE
                elif crop_part == "grid_12":
                    properties = c1p.CROP_GRID_12_VARIABLE
                elif crop_part == "grid_13":
                    properties = c1p.CROP_GRID_13_VARIABLE
                elif crop_part == "grid_14":
                    properties = c1p.CROP_GRID_14_VARIABLE
                elif crop_part == "grid_15":
                    properties = c1p.CROP_GRID_15_VARIABLE
            elif cam_number == 2:
                if crop_part == "grid_1":
                    properties = c2p.CROP_GRID_1_VARIABLE
                elif crop_part == "grid_2":
                    properties = c2p.CROP_GRID_2_VARIABLE
                elif crop_part == "grid_3":
                    properties = c2p.CROP_GRID_3_VARIABLE
                elif crop_part == "grid_4":
                    properties = c2p.CROP_GRID_4_VARIABLE
                elif crop_part == "grid_5":
                    properties = c2p.CROP_GRID_5_VARIABLE
                elif crop_part == "grid_6":
                    properties = c2p.CROP_GRID_6_VARIABLE
                elif crop_part == "grid_7":
                    properties = c2p.CROP_GRID_7_VARIABLE
                elif crop_part == "grid_8":
                    properties = c2p.CROP_GRID_8_VARIABLE
                elif crop_part == "grid_9":
                    properties = c2p.CROP_GRID_9_VARIABLE
                elif crop_part == "grid_10":
                    properties = c2p.CROP_GRID_10_VARIABLE
                elif crop_part == "grid_11":
                    properties = c2p.CROP_GRID_11_VARIABLE
                elif crop_part == "grid_12":
                    properties = c2p.CROP_GRID_12_VARIABLE
                elif crop_part == "grid_13":
                    properties = c2p.CROP_GRID_13_VARIABLE
                elif crop_part == "grid_14":
                    properties = c2p.CROP_GRID_14_VARIABLE
                elif crop_part == "grid_15":
                    properties = c2p.CROP_GRID_15_VARIABLE
                elif crop_part == "grid_16":
                    properties = c2p.CROP_GRID_16_VARIABLE
            elif cam_number == 3:
                if crop_part == "grid_1":
                    properties = c3p.CROP_GRID_1_VARIABLE
                elif crop_part == "grid_2":
                    properties = c3p.CROP_GRID_2_VARIABLE
                elif crop_part == "grid_3":
                    properties = c3p.CROP_GRID_3_VARIABLE
                elif crop_part == "grid_4":
                    properties = c3p.CROP_GRID_4_VARIABLE
                elif crop_part == "grid_5":
                    properties = c3p.CROP_GRID_5_VARIABLE
                elif crop_part == "grid_6":
                    properties = c3p.CROP_GRID_6_VARIABLE
                elif crop_part == "grid_7":
                    properties = c3p.CROP_GRID_7_VARIABLE
                elif crop_part == "grid_8":
                    properties = c3p.CROP_GRID_8_VARIABLE
                elif crop_part == "grid_9":
                    properties = c3p.CROP_GRID_9_VARIABLE
                elif crop_part == "grid_10":
                    properties = c3p.CROP_GRID_10_VARIABLE
                elif crop_part == "grid_11":
                    properties = c3p.CROP_GRID_11_VARIABLE
                elif crop_part == "grid_12":
                    properties = c3p.CROP_GRID_12_VARIABLE
                elif crop_part == "grid_13":
                    properties = c3p.CROP_GRID_13_VARIABLE
                elif crop_part == "grid_14":
                    properties = c3p.CROP_GRID_14_VARIABLE
                elif crop_part == "grid_15":
                    properties = c3p.CROP_GRID_15_VARIABLE
                elif crop_part == "grid_16":
                    properties = c3p.CROP_GRID_16_VARIABLE
                elif crop_part == "grid_17":
                    properties = c3p.CROP_GRID_17_VARIABLE
            elif cam_number == 4:
                if crop_part == "grid_1":
                    properties = c3p.CROP_GRID_1_VARIABLE
        elif product == "cover":
            pass

        return properties

    def get_rotate_coord(self, center_coord, angle: int):
        radian = math.pi / 180
        center_x = center_coord.x
        center_y = center_coord.y

        rotate_x = (self.x - center_x) * math.cos(angle * radian) - (self.y - center_y) * math.sin(angle * radian) + center_x
        rotate_y = (self.x - center_x) * math.sin(angle * radian) + (self.y - center_y) * math.cos(angle * radian) + center_y

        return Coordinate(int(rotate_x), int(rotate_y))

    def get_grid_to_original_coord(self, product, cam_number, current_grid, resize_ratio=2.0):
        crop_properties = self._get_properties(product=product, cam_number=cam_number, crop_part=current_grid)
        original_x = int(self.x)
        original_y = int(self.y)

        if resize_ratio != 1.0:
            original_x = int((self.x * resize_ratio) + crop_properties.LEFT_UP_COORDINATE.x)
            original_y = int((self.y * resize_ratio) + crop_properties.LEFT_UP_COORDINATE.y)

        return Coordinate(x=original_x, y=original_y)

    def get_grid_to_original_coord_with_abst_bbox(self, abst_bbox, resize_ratio=2.0):
        original_x = int(self.x)
        original_y = int(self.y)

        if resize_ratio != 1.0:
            original_x = int((self.x * resize_ratio) + abst_bbox.x_st)
            original_y = int((self.y * resize_ratio) + abst_bbox.y_st)

        return Coordinate(x=original_x, y=original_y)

    def is_in_margin(self, img_center_coord, margin_min: int, margin_max: int) -> bool:
        img_center_x = img_center_coord.x
        img_center_y = img_center_coord.y

        distance = math.sqrt(((img_center_x - self.x) ** 2) + ((img_center_y - self.y) ** 2))

        return True if margin_max > distance >= margin_min else False

    def get_distance(self, coord):
        return np.sqrt(np.square(coord.x - self.x) + np.square(coord.y - self.y))


class BoundingBox(ImageDomain):

    def __init__(self, **kwargs):
        super(BoundingBox, self).__init__()

        if len(kwargs) == 0:
            self.x_st: int = None
            self.x_ed: int = None
            self.y_st: int = None
            self.y_ed: int = None
            self.left_up: Coordinate = None
            self.right_up: Coordinate = None
            self.left_down: Coordinate = None
            self.right_down: Coordinate = None
            self.center: Coordinate = None
        elif len(kwargs) != 4:
            raise ValueError("Please put 4 values (Type: int || Coordinate)")
        else:
            self.arg_type = type(next(iter(kwargs.values())))

            if self.arg_type is int:
                self.init_with_st_ed(kwargs['x_st'], kwargs['x_ed'], kwargs['y_st'], kwargs['y_ed'])
            elif self.arg_type is Coordinate:
                self.init_with_coord(kwargs['left_up'], kwargs['right_up'], kwargs['left_down'], kwargs['right_down'])
            else:
                raise ValueError("The type value is must be 'int' or 'Coordinate'")

    def __repr__(self):
        return f"x_st: {self.x_st} \n" \
               f"x_ed: {self.x_ed} \n" \
               f"y_st: {self.y_st} \n" \
               f"y_ed: {self.y_ed} \n" \
               f"left_up: {self.left_up} \n" \
               f"right_up: {self.right_up} \n" \
               f"left_down: {self.left_down} \n" \
               f"right_down: {self.right_down} \n" \
               f"center: {self.center}"

    def init_with_st_ed(self, x_st: int = 0, x_ed: int = 0, y_st: int = 0, y_ed: int = 0):
        self.x_st = x_st
        self.x_ed = x_ed
        self.y_st = y_st
        self.y_ed = y_ed
        self.left_up: Coordinate = Coordinate(self.x_st, self.y_st)
        self.right_up: Coordinate = Coordinate(self.x_ed, self.y_st)
        self.left_down: Coordinate = Coordinate(self.x_st, self.y_ed)
        self.right_down: Coordinate = Coordinate(self.x_ed, self.y_ed)
        self.__calculate_center_coord()

    def init_with_coord(self, left_up: Coordinate, right_up: Coordinate, left_down: Coordinate, right_down: Coordinate):
        self.x_st = left_up.x
        self.x_ed = right_down.x
        self.y_st = left_up.y
        self.y_ed = right_down.y
        self.left_up = left_up
        self.right_up = right_up
        self.left_down = left_down
        self.right_down = right_down
        self.__calculate_center_coord()

    def get_rotate_bounding_box(self, center_coord: Coordinate, angle: int):
        rotate_left_up = self.left_up.get_rotate_coord(center_coord=center_coord, angle=angle)
        rotate_right_up = self.right_up.get_rotate_coord(center_coord=center_coord, angle=angle)
        rotate_left_down = self.left_down.get_rotate_coord(center_coord=center_coord, angle=angle)
        rotate_right_down = self.right_down.get_rotate_coord(center_coord=center_coord, angle=angle)
        rotate_center = self.center.get_rotate_coord(center_coord=center_coord, angle=angle)

        rotate_bbox = BoundingBox(
            left_up=rotate_left_up,
            right_up=rotate_right_up,
            left_down=rotate_left_down,
            right_down=rotate_right_down
        )
        rotate_bbox.center = rotate_center

        return rotate_bbox

    def calculate_with_ratio(self, ratio: float):
        def __calculate(point, _ratio):
            if _ratio == 0:
                return 0
            return int(point * (1 / (_ratio / 3)))

        self.init_with_st_ed(
            x_st=__calculate(self.x_st, ratio),
            x_ed=__calculate(self.x_ed, ratio),
            y_st=__calculate(self.y_st, ratio),
            y_ed=__calculate(self.y_ed, ratio)
        )

    def __calculate_center_coord(self):
        self.center = Coordinate((self.x_st + self.x_ed) // 2, (self.y_st + self.y_ed) // 2)


class EOPImage(ImageDomain):

    def __init__(self, img, abst_bbox: BoundingBox =None, defect_list=None):
        super(EOPImage, self).__init__()

        self.base_img = self
        self.parent: EOPImage = None
        self.abst_bbox = None
        self._child_list: List[EOPImage] = list()
        self._sibling_num: int = 0 if not self.child_list else -1

        self.img = self._get_converted_img(img)

        self.defect_list = list() if defect_list is None else defect_list

        self._set_abst_bbox(abst_bbox)

    def __call__(self):
        return self

    def __repr__(self):
        return f"Absolute Bounding Box : {self.abst_bbox}\n" \
               f"Defect List : {self.defect_list}"

    @property
    def child_list(self):
        return self._child_list

    def exist_child_list(self):
        return True if self.child_list else False

    @property
    def sibling_list(self):
        return self.parent.child_list

    def get_child(self, child_idx):
        return self.child_list[child_idx]

    def get_all_child(self):
        all_child_list = list()

        while not self.exist_child_list():
            all_child_list += self.child_list

            for child in all_child_list:
                child: EOPImage
                child.get_all_child()

        return all_child_list

    def add_child(self, eop_img):
        eop_img.parent = self
        eop_img.base_img = self.base_img
        eop_img._sibling_num = len(self.child_list)
        self.child_list.append(eop_img)

    def show_img(self, figsize=(15, 7)):
        img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)

        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.show()


    def _get_converted_img(self, img):
        # If img is path of the image
        converted_img = None

        # Image path
        if isinstance(img, str):
            with open(img, "rb") as f:
                img = bytearray(f.read())
                img = np.asarray(img, dtype=np.uint8)
            converted_img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

            # If gray image
            if len(converted_img.shape) != 2:
                converted_img = cv2.cvtColor(converted_img, cv2.COLOR_BGR2GRAY)

            converted_img = converted_img.astype(np.float32)

        # Numpy array
        elif isinstance(img, np.ndarray):
            converted_img = img
            converted_img = converted_img.astype(np.float32)
        else:
            raise ValueError("Image path must be not None")

        return converted_img

    def _set_abst_bbox(self, abst_bbox):
        if abst_bbox is None:
            img_height = self.img.shape[0]
            img_width = self.img.shape[1]

            self.abst_bbox = BoundingBox(
                x_st=0,
                x_ed=img_width - 1,
                y_st=0,
                y_ed=img_height - 1
            )
        else:
            self.abst_bbox = abst_bbox


class Defect(ImageDomain):

    def __init__(self, abst_bbox: BoundingBox, category: int, location_img: EOPImage):
        super(Defect, self).__init__()

        self.abst_bbox = abst_bbox
        self.category = category
        """
        @TODO
        이거 수정 할 것.
        location_img 는 하나가 아니라 여러 개가 있을 수 있음 
        """
        self.location_img = location_img

    def __repr__(self):
        return self.abst_bbox.__repr__()

    @property
    def center_x(self):
        return self.abst_bbox.center.x

    @property
    def center_y(self):
        return self.abst_bbox.center.y

    @property
    def center(self):
        return self.center_x, self.center_y

    def is_defect_in_cropped_img(self, img_center_coord: Coordinate, margin_min: int, margin_max: int) -> bool:
        img_center_x = img_center_coord.x
        img_center_y = img_center_coord.y

        distance = math.sqrt(((img_center_x - self.center_x) ** 2) + ((img_center_y - self.center_y) ** 2))

        return True if margin_max > distance >= margin_min else False

    def is_defect_in_bounding_box(self, bbox: BoundingBox):
        coord_x_list = sorted([bbox.left_up.x, bbox.right_up.x, bbox.right_down.x, bbox.left_down.x])
        coord_y_list = sorted([bbox.left_up.y, bbox.right_up.y, bbox.right_down.y, bbox.left_down.y])

        is_defect_center_x_in_bbox = coord_x_list[0] <= self.center_x <= coord_x_list[-1]
        is_defect_center_y_in_bbox = coord_y_list[0] <= self.center_y <= coord_y_list[-1]

        return is_defect_center_x_in_bbox and is_defect_center_y_in_bbox


class Contour(ImageDomain):

    def __init__(self, contour, contour_area_threshold=5, is_approximated_polygon=False):
        super(Contour, self).__init__()

        if is_approximated_polygon:
            epsilon = cv2.arcLength(contour, closed=True) * 0.01
            approximated_polygon = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
            contour = approximated_polygon

        self.contour = contour

        # Property
        self.moment_list = cv2.moments(self.contour)
        self.area_size = self.moment_list['m00']

        if self.area_size >= contour_area_threshold:
            self.center_x = int(self.moment_list['m10'] / self.moment_list['m00'])
            self.center_y = int(self.moment_list['m01'] / self.moment_list['m00'])

            rect = cv2.minAreaRect(contour)
            box_list = cv2.boxPoints(rect)
            box_list = np.int0(box_list)

            self.box_list = box_list

            # self.ellipse = cv2.fitEllipse(contour)
        else:
            self.center_x = -1
            self.center_y = -1

        self.center = Coordinate(x=self.center_x, y=self.center_y)

    def __repr__(self):
        return self.center.__repr__()

    # Check if center coordinate is in difference image (만약 무게중심이 ae img 밖에 존재하면 예외)
    def is_in_difference_img(self, difference_img):
        is_in_difference_img = False

        if 0 <= self.center_y < difference_img.shape[0] and 0 <= self.center_x < difference_img.shape[1]:
            is_in_difference_img = True

        return is_in_difference_img