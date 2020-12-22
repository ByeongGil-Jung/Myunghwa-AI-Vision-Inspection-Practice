from dataclasses import dataclass

from domain.image import Area, Coordinate, BoundingBox


@dataclass
class Cam1Area:
    A = Area("A")
    C = Area("C")
    D_INNER = Area("D_inner")
    D_OUTER = Area("D_outer")

    @classmethod
    def get_list(cls):
        return [cls.A, cls.C, cls.D_INNER, cls.D_OUTER]

    @classmethod
    def get_list_dict(cls):
        return {area.name: list() for area in cls.get_list()}


@dataclass
class VariableHousingCam1CropA:
    # BIG_RAD = 710
    # SMALL_RAD = 641
    # CENTER_COORD = Coordinate(1255, 987)
    BIG_RAD = 705
    SMALL_RAD = 639
    CENTER_COORD = Coordinate(1254, 980)
    ANGLE = 30


@dataclass
class VariableHousingCam1CropCEntire:
    RAD = 463
    CENTER_COORD = Coordinate(1235, 1029)


@dataclass
class VariableHousingCam1CropC:
    BIG_RAD = 463
    SMALL_RAD = 103
    CENTER_COORD = Coordinate(1235, 1029)
    ANGLE = 90


@dataclass
class VariableHousingCam1CropDRing:
    INNER_CIRCLE_RAD = 463
    INNER_CIRCLE_CENTER_COORD = Coordinate(1235, 1029)
    OUTER_CIRCLE_RAD = 808
    OUTER_CIRCLE_CENTER_COORD = Coordinate(1232, 993)
    RING_SMALL_RAD = 430
    ANGLE = 60


@dataclass
class VariableHousingCam1CropDInnerRing:
    INNER_CIRCLE_RAD = 463
    INNER_CIRCLE_CENTER_COORD = Coordinate(1235, 1029)
    OUTER_CIRCLE_RAD = 645
    OUTER_CIRCLE_CENTER_COORD = Coordinate(1254, 980)
    RING_SMALL_RAD = 401
    ANGLE = 60


@dataclass
class VariableHousingCam1CropDOuterRing:
    BIG_RAD = 843
    SMALL_RAD = 715
    CENTER_COORD = Coordinate(1250, 987)
    ANGLE = 30
    INDEX_LIST = [2, 4, 5, 6, 7, 8, 10]


@dataclass
class VariableHousingCam1CropDOuterEntire:
    RAD = 705
    CENTER_COORD = Coordinate(1254, 980)


# Test
@dataclass
class VariableHousingCam1CropDOuterRingInside:
    BIG_RAD = 782
    SMALL_RAD = 705
    CENTER_COORD = Coordinate(1249, 975)
    ANGLE = 30


# Test
@dataclass
class VariableHousingCam1CropDOuterRingOutside:
    BIG_RAD = 825
    SMALL_RAD = 782
    CENTER_COORD = Coordinate(1249, 979)
    ANGLE = 15
    INDEX_LIST = [0, 1, 13, 14, 15, 16, 17, 18, 22, 23]


@dataclass
class VariableHousingCam1CropDRing:
    INNER_CIRCLE_RAD = 463
    INNER_CIRCLE_CENTER_COORD = Coordinate(1235, 1029)
    OUTER_CIRCLE_RAD = 808
    OUTER_CIRCLE_CENTER_COORD = Coordinate(1232, 993)
    RING_SMALL_RAD = 430
    ANGLE = 60


@dataclass
class VariableHousingCam1CropDOuterRectangleWithRing:
    BBOX_LIST = [
        BoundingBox(x_st=1822, x_ed=1822 + 500, y_st=780, y_ed=780 + 500),
        BoundingBox(x_st=850, x_ed=850 + 500, y_st=1646, y_ed=1646 + 500),
        BoundingBox(x_st=72, x_ed=72 + 500, y_st=766, y_ed=766 + 500)
    ]
    ANGLE_LIST = [0, 90, 180]
    BOUNDARY_CIRCLE_CENTER_COORD = Coordinate(1250, 987)
    BOUNDARY_CIRCLE_RAD = 710


@dataclass
class VariableHousingCam1CropDOuterRectangleTop:
    ENTIRE_X_ST = 598
    ENTIRE_X_ED = 598 + 1184
    ENTIRE_Y_ST = 132
    ENTIRE_Y_ED = 132 + 368

    BOUNDARY_CIRCLE_CENTER_COORD = Coordinate(1250, 987)
    BOUNDARY_CIRCLE_RAD = 710
    FLIP_INDEX = 0

    __HALF_X = (ENTIRE_X_ST + ENTIRE_X_ED) // 2
    BBOX_LIST = [
        BoundingBox(x_st=ENTIRE_X_ST, x_ed=__HALF_X, y_st=ENTIRE_Y_ST, y_ed=ENTIRE_Y_ED),
        BoundingBox(x_st=__HALF_X, x_ed=ENTIRE_X_ED, y_st=ENTIRE_Y_ST, y_ed=ENTIRE_Y_ED)
    ]


"""
Grid
"""
@dataclass
class VariableHousingCam1GridEntire:
    BBOX_LIST = [
        # Square 1st
        BoundingBox(x_st=471, x_ed=471 + 512, y_st=150, y_ed=150 + 512),
        BoundingBox(x_st=965, x_ed=965 + 512, y_st=150, y_ed=150 + 512),
        BoundingBox(x_st=1459, x_ed=1459 + 512, y_st=150, y_ed=150 + 512),
        # Square 2nd
        BoundingBox(x_st=471, x_ed=471 + 512, y_st=644, y_ed=644 + 512),
        BoundingBox(x_st=965, x_ed=965 + 512, y_st=644, y_ed=644 + 512),
        BoundingBox(x_st=1459, x_ed=1459 + 512, y_st=644, y_ed=644 + 512),
        # Square 3rd
        BoundingBox(x_st=471, x_ed=471 + 512, y_st=1138, y_ed=1138 + 512),
        BoundingBox(x_st=965, x_ed=965 + 512, y_st=1138, y_ed=1138 + 512),
        BoundingBox(x_st=1459, x_ed=1459 + 512, y_st=1138, y_ed=1138 + 512),
        # Left
        BoundingBox(x_st=27, x_ed=27 + 512, y_st=577, y_ed=577 + 512),
        BoundingBox(x_st=27, x_ed=27 + 512, y_st=993, y_ed=993 + 512),
        # # Right
        BoundingBox(x_st=1791, x_ed=1791 + 512, y_st=547, y_ed=547 + 512),
        BoundingBox(x_st=1791, x_ed=1791 + 512, y_st=949, y_ed=949 + 512),
        # # Down
        BoundingBox(x_st=641, x_ed=641 + 512, y_st=1629, y_ed=1629 + 512),
        BoundingBox(x_st=1125, x_ed=1125 + 512, y_st=1629, y_ed=1629 + 512),
    ]


@dataclass
class VariableHousingCam1Grid1:
    LEFT_UP_COORDINATE = Coordinate(x=471, y=150)
    BBOX_LIST = [
        BoundingBox(x_st=471, x_ed=471 + 512, y_st=150, y_ed=150 + 512)
    ]

@dataclass
class VariableHousingCam1Grid2:
    LEFT_UP_COORDINATE = Coordinate(x=965, y=150)
    BBOX_LIST = [
        BoundingBox(x_st=965, x_ed=965 + 512, y_st=150, y_ed=150 + 512)
    ]

@dataclass
class VariableHousingCam1Grid3:
    LEFT_UP_COORDINATE = Coordinate(x=1459, y=150)
    BBOX_LIST = [
        BoundingBox(x_st=1459, x_ed=1459 + 512, y_st=150, y_ed=150 + 512)
    ]


@dataclass
class VariableHousingCam1Grid4:
    LEFT_UP_COORDINATE = Coordinate(x=471, y=644)
    BBOX_LIST = [
        BoundingBox(x_st=471, x_ed=471 + 512, y_st=644, y_ed=644 + 512)
    ]

@dataclass
class VariableHousingCam1Grid5:
    LEFT_UP_COORDINATE = Coordinate(x=965, y=644)
    BBOX_LIST = [
        BoundingBox(x_st=965, x_ed=965 + 512, y_st=644, y_ed=644 + 512)
    ]

@dataclass
class VariableHousingCam1Grid6:
    LEFT_UP_COORDINATE = Coordinate(x=1459, y=644)
    BBOX_LIST = [
        BoundingBox(x_st=1459, x_ed=1459 + 512, y_st=644, y_ed=644 + 512)
    ]


@dataclass
class VariableHousingCam1Grid7:
    LEFT_UP_COORDINATE = Coordinate(x=471, y=1138)
    BBOX_LIST = [
        BoundingBox(x_st=471, x_ed=471 + 512, y_st=1138, y_ed=1138 + 512)
    ]

@dataclass
class VariableHousingCam1Grid8:
    LEFT_UP_COORDINATE = Coordinate(x=965, y=1138)
    BBOX_LIST = [
        BoundingBox(x_st=965, x_ed=965 + 512, y_st=1138, y_ed=1138 + 512)
    ]

@dataclass
class VariableHousingCam1Grid9:
    LEFT_UP_COORDINATE = Coordinate(x=1459, y=1138)
    BBOX_LIST = [
        BoundingBox(x_st=1459, x_ed=1459 + 512, y_st=1138, y_ed=1138 + 512)
    ]


@dataclass
class VariableHousingCam1Grid10:
    LEFT_UP_COORDINATE = Coordinate(x=27, y=577)
    BBOX_LIST = [
        BoundingBox(x_st=27, x_ed=27 + 512, y_st=577, y_ed=577 + 512)
    ]

@dataclass
class VariableHousingCam1Grid11:
    LEFT_UP_COORDINATE = Coordinate(x=27, y=993)
    BBOX_LIST = [
        BoundingBox(x_st=27, x_ed=27 + 512, y_st=993, y_ed=993 + 512)
    ]


@dataclass
class VariableHousingCam1Grid12:
    LEFT_UP_COORDINATE = Coordinate(x=1791, y=547)
    BBOX_LIST = [
        BoundingBox(x_st=1791, x_ed=1791 + 512, y_st=547, y_ed=547 + 512)
    ]

@dataclass
class VariableHousingCam1Grid13:
    LEFT_UP_COORDINATE = Coordinate(x=1791, y=949)
    BBOX_LIST = [
        BoundingBox(x_st=1791, x_ed=1791 + 512, y_st=949, y_ed=949 + 512)
    ]


@dataclass
class VariableHousingCam1Grid14:
    LEFT_UP_COORDINATE = Coordinate(x=641, y=1629)
    BBOX_LIST = [
        BoundingBox(x_st=641, x_ed=641 + 512, y_st=1629, y_ed=1629 + 512)
    ]

@dataclass
class VariableHousingCam1Grid15:
    LEFT_UP_COORDINATE = Coordinate(x=1125, y=1629)
    BBOX_LIST = [
        BoundingBox(x_st=1125, x_ed=1125 + 512, y_st=1629, y_ed=1629 + 512)
    ]


CAM1_AREA = Cam1Area()

CROP_A_VARIABLE = VariableHousingCam1CropA()
CROP_C_VARIABLE = VariableHousingCam1CropC()
CROP_D_INNER_RING_VARIABLE = VariableHousingCam1CropDInnerRing()
CROP_D_OUTER_RING_VARIABLE = VariableHousingCam1CropDOuterRing()

# Test
CROP_C_ENTIRE_VARIABLE = VariableHousingCam1CropCEntire()
CROP_D_RING_VARIABLE = VariableHousingCam1CropDRing()
CROP_D_OUTER_ENTIRE_VARIABLE = VariableHousingCam1CropDOuterEntire()
CROP_D_OUTER_RING_INSIDE_VARIABLE = VariableHousingCam1CropDOuterRingInside()
CROP_D_OUTER_RING_OUTSIDE_VARIABLE = VariableHousingCam1CropDOuterRingOutside()

CROP_D_OUTER_RECTANGLE_WITH_RING_VARIABLE = VariableHousingCam1CropDOuterRectangleWithRing()
CROP_D_OUTER_RECTANGLE_TOP_VARIABLE = VariableHousingCam1CropDOuterRectangleTop()

# Grid Test
CROP_GRID_ENTIRE_VARIABLE = VariableHousingCam1GridEntire()

CROP_GRID_1_VARIABLE = VariableHousingCam1Grid1()
CROP_GRID_2_VARIABLE = VariableHousingCam1Grid2()
CROP_GRID_3_VARIABLE = VariableHousingCam1Grid3()

CROP_GRID_4_VARIABLE = VariableHousingCam1Grid4()
CROP_GRID_5_VARIABLE = VariableHousingCam1Grid5()
CROP_GRID_6_VARIABLE = VariableHousingCam1Grid6()

CROP_GRID_7_VARIABLE = VariableHousingCam1Grid7()
CROP_GRID_8_VARIABLE = VariableHousingCam1Grid8()
CROP_GRID_9_VARIABLE = VariableHousingCam1Grid9()

CROP_GRID_10_VARIABLE = VariableHousingCam1Grid10()
CROP_GRID_11_VARIABLE = VariableHousingCam1Grid11()

CROP_GRID_12_VARIABLE = VariableHousingCam1Grid12()
CROP_GRID_13_VARIABLE = VariableHousingCam1Grid13()

CROP_GRID_14_VARIABLE = VariableHousingCam1Grid14()
CROP_GRID_15_VARIABLE = VariableHousingCam1Grid15()
