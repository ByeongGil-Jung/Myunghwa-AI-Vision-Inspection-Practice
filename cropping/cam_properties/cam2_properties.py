from dataclasses import dataclass

from domain.image import Area, BoundingBox, Coordinate


@dataclass
class Cam2Area:
    C_INNER = Area("C_inner")
    C_OUTER = Area("C_outer")

    @classmethod
    def get_list(cls):
        return [cls.C_INNER, cls.C_OUTER]

    @classmethod
    def get_list_dict(cls):
        return {area.name: list() for area in cls.get_list()}


@dataclass
class VariableHousingCam2CropCRectangleSide:
    BBOX_LIST = [
        BoundingBox(x_st=402, x_ed=402 + 740, y_st=126, y_ed=126 + 926),
        BoundingBox(x_st=2658, x_ed=2658 + 740, y_st=126, y_ed=126 + 926)
    ]
    FLIP_INDEX = 0


@dataclass
class VariableHousingCam2CropCRectangleMiddle:
    ENTIRE_X_ST = 402 + 740
    ENTIRE_X_ED = 2658
    ENTIRE_Y_ST = 404
    ENTIRE_Y_ED = 126 + 926

    FLIP_INDEX = 0

    __HALF_X = (ENTIRE_X_ST + ENTIRE_X_ED) // 2
    BBOX_LIST = [
        BoundingBox(x_st=ENTIRE_X_ST, x_ed=__HALF_X, y_st=ENTIRE_Y_ST, y_ed=ENTIRE_Y_ED),
        BoundingBox(x_st=__HALF_X, x_ed=ENTIRE_X_ED, y_st=ENTIRE_Y_ST, y_ed=ENTIRE_Y_ED)
    ]


@dataclass
class VariableHousingCam2CropCInnerEntire:
    BBOX_LIST = [
        BoundingBox(x_st=1142, x_ed=1142 + 1516, y_st=404, y_ed=404 + 444)
    ]
    FLIP_INDEX = 0


@dataclass
class VariableHousingCam2CropCOuterEntire:
    BBOX_LIST = [
        BoundingBox(x_st=402, x_ed=402 + 2996, y_st=126, y_ed=126 + 920)
    ]
    FLIP_INDEX = 0


"""
Grid
"""
@dataclass
class VariableHousingCam2GridEntire:
    BBOX_LIST = [
        # Outer left
        BoundingBox(x_st=382, x_ed=383 + 512, y_st=116, y_ed=116 + 512),
        BoundingBox(x_st=648, x_ed=648 + 512, y_st=116, y_ed=116 + 512),
        BoundingBox(x_st=382, x_ed=383 + 512, y_st=550, y_ed=550 + 512),
        BoundingBox(x_st=648, x_ed=648 + 512, y_st=550, y_ed=550 + 512),

        # Outer right
        BoundingBox(x_st=2636, x_ed=2636 + 512, y_st=116, y_ed=116 + 512),
        BoundingBox(x_st=2901, x_ed=2901 + 512, y_st=116, y_ed=116 + 512),
        BoundingBox(x_st=2636, x_ed=2636 + 512, y_st=550, y_ed=550 + 512),
        BoundingBox(x_st=2901, x_ed=2901 + 512, y_st=550, y_ed=550 + 512),

        # Inner
        BoundingBox(x_st=1064, x_ed=1064 + 512, y_st=390, y_ed=390 + 512),
        BoundingBox(x_st=1384, x_ed=1384 + 512, y_st=390, y_ed=390 + 512),
        BoundingBox(x_st=1896, x_ed=1896 + 512, y_st=390, y_ed=390 + 512),
        BoundingBox(x_st=2408, x_ed=2408 + 512, y_st=390, y_ed=390 + 512),
        BoundingBox(x_st=1064, x_ed=1064 + 512, y_st=550, y_ed=550 + 512),
        BoundingBox(x_st=1384, x_ed=1384 + 512, y_st=550, y_ed=550 + 512),
        BoundingBox(x_st=1896, x_ed=1896 + 512, y_st=550, y_ed=550 + 512),
        BoundingBox(x_st=2408, x_ed=2408 + 512, y_st=550, y_ed=550 + 512)
    ]


@dataclass
class VariableHousingCam2Grid1:
    BBOX_LIST = [
        BoundingBox(x_st=382, x_ed=383 + 512, y_st=116, y_ed=116 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid2:
    BBOX_LIST = [
        BoundingBox(x_st=648, x_ed=648 + 512, y_st=116, y_ed=116 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid3:
    BBOX_LIST = [
        BoundingBox(x_st=382, x_ed=383 + 512, y_st=550, y_ed=550 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid4:
    BBOX_LIST = [
        BoundingBox(x_st=648, x_ed=648 + 512, y_st=550, y_ed=550 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid5:
    BBOX_LIST = [
        BoundingBox(x_st=2636, x_ed=2636 + 512, y_st=116, y_ed=116 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid6:
    BBOX_LIST = [
        BoundingBox(x_st=2901, x_ed=2901 + 512, y_st=116, y_ed=116 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid7:
    BBOX_LIST = [
        BoundingBox(x_st=2636, x_ed=2636 + 512, y_st=550, y_ed=550 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid8:
    BBOX_LIST = [
        BoundingBox(x_st=2901, x_ed=2901 + 512, y_st=550, y_ed=550 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid9:
    BBOX_LIST = [
        BoundingBox(x_st=1064, x_ed=1064 + 512, y_st=390, y_ed=390 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid10:
    BBOX_LIST = [
        BoundingBox(x_st=1384, x_ed=1384 + 512, y_st=390, y_ed=390 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid11:
    BBOX_LIST = [
        BoundingBox(x_st=1896, x_ed=1896 + 512, y_st=390, y_ed=390 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid12:
    BBOX_LIST = [
        BoundingBox(x_st=2408, x_ed=2408 + 512, y_st=390, y_ed=390 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid13:
    BBOX_LIST = [
        BoundingBox(x_st=1064, x_ed=1064 + 512, y_st=550, y_ed=550 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid14:
    BBOX_LIST = [
        BoundingBox(x_st=1384, x_ed=1384 + 512, y_st=550, y_ed=550 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid15:
    BBOX_LIST = [
        BoundingBox(x_st=1896, x_ed=1896 + 512, y_st=550, y_ed=550 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


@dataclass
class VariableHousingCam2Grid16:
    BBOX_LIST = [
        BoundingBox(x_st=2408, x_ed=2408 + 512, y_st=550, y_ed=550 + 512)
    ]
    LEFT_UP_COORDINATE = Coordinate(x=BBOX_LIST[0].x_st, y=BBOX_LIST[0].y_st)


CAM2_AREA = Cam2Area()

CROP_C_RECTANGLE_SIDE_VARIABLE = VariableHousingCam2CropCRectangleSide()
CROP_C_RECTANGLE_MIDDLE_VARIABLE = VariableHousingCam2CropCRectangleMiddle()

CROP_C_INNER_ENTIRE_VARIABLE = VariableHousingCam2CropCInnerEntire()
CROP_C_OUTER_ENTIRE_VARIABLE = VariableHousingCam2CropCOuterEntire()

# Grid Test
CROP_GRID_ENTIRE_VARIABLE = VariableHousingCam2GridEntire()

CROP_GRID_1_VARIABLE = VariableHousingCam2Grid1()
CROP_GRID_2_VARIABLE = VariableHousingCam2Grid2()
CROP_GRID_3_VARIABLE = VariableHousingCam2Grid3()
CROP_GRID_4_VARIABLE = VariableHousingCam2Grid4()

CROP_GRID_5_VARIABLE = VariableHousingCam2Grid5()
CROP_GRID_6_VARIABLE = VariableHousingCam2Grid6()
CROP_GRID_7_VARIABLE = VariableHousingCam2Grid7()
CROP_GRID_8_VARIABLE = VariableHousingCam2Grid8()

CROP_GRID_9_VARIABLE = VariableHousingCam2Grid9()
CROP_GRID_10_VARIABLE = VariableHousingCam2Grid10()
CROP_GRID_11_VARIABLE = VariableHousingCam2Grid11()
CROP_GRID_12_VARIABLE = VariableHousingCam2Grid12()
CROP_GRID_13_VARIABLE = VariableHousingCam2Grid13()
CROP_GRID_14_VARIABLE = VariableHousingCam2Grid14()
CROP_GRID_15_VARIABLE = VariableHousingCam2Grid15()
CROP_GRID_16_VARIABLE = VariableHousingCam2Grid16()
