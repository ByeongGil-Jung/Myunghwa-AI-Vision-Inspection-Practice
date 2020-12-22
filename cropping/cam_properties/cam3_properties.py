from dataclasses import dataclass

from domain.image import Area, Coordinate, BoundingBox


@dataclass
class Cam3Area:
    A = Area("A")
    C_INNER_RING_INSIDE = Area("C_inner_ring_inside")
    C_INNER_RING_OUTSIDE = Area("C_inner_ring_outside")
    C_OUTER = Area("C_outer")
    D = Area("D")

    @classmethod
    def get_list(cls):
        return [cls.A, cls.C_INNER_RING_INSIDE, cls.C_INNER_RING_OUTSIDE, cls.C_OUTER, cls.D]

    @classmethod
    def get_list_dict(cls):
        return {area.name: list() for area in cls.get_list()}


@dataclass
class VariableHousingCam3CropACDRing:
    BIG_RAD = 835
    SMALL_RAD = 618
    CENTER_COORD = Coordinate(1210, 1059)
    ANGLE = 60


@dataclass
class VariableHousingCam3CropCInnerRing:
    BIG_RAD = 335
    SMALL_RAD = 145
    CENTER_COORD = Coordinate(1204, 1060)
    ANGLE = 60


# Test
@dataclass
class VariableHousingCam3CropCInnerRingEntire:
    RAD = 335
    CENTER_COORD = Coordinate(1206, 1059)


@dataclass
class VariableHousingCam3CropCInnerRingInsideEntire:
    RAD = 286
    CENTER_COORD = Coordinate(1206, 1059)


@dataclass
class VariableHousingCam3CropCInnerRingOutsideEntire:
    SMALL_RAD = 286
    BIG_RAD = 335
    CENTER_COORD = Coordinate(1206, 1059)


@dataclass
class VariableHousingCam3CropDRing:
    BIG_RAD = 702
    SMALL_RAD = 618
    CENTER_COORD = Coordinate(1210, 1062)
    ANGLE = 30


@dataclass
class VariableHousingCam3CropDRingEntire:
    BIG_RAD = 702
    SMALL_RAD = 618
    CENTER_COORD = Coordinate(1210, 1062)


@dataclass
class VariableHousingCam3CropA:
    BIG_RAD = 759
    SMALL_RAD = 696
    CENTER_COORD = Coordinate(1211, 1065)
    ANGLE = 30


@dataclass
class VariableHousingCam3CropARingEntire:
    BIG_RAD = 759
    SMALL_RAD = 696
    CENTER_COORD = Coordinate(1211, 1065)


@dataclass
class VariableHousingCam3CropCOuterEntire:
    RAD = 759
    CENTER_COORD = Coordinate(1211, 1065)


@dataclass
class VariableHousingCam3CropCOuterRing:
    BIG_RAD = 821
    SMALL_RAD = 757
    CENTER_COORD = Coordinate(1205, 1062)
    ANGLE = 30
    INDEX_LIST = [2, 4, 5, 6, 7, 8, 10]


@dataclass
class VariableHousingCam3CropCRectangleWithRing:
    BBOX_LIST = [
        BoundingBox(x_st=1842, x_ed=1842 + 476, y_st=824, y_ed=824 + 476),
        BoundingBox(x_st=1049, x_ed=1049 + 476, y_st=1707, y_ed=1707 + 476),
        BoundingBox(x_st=79, x_ed=79 + 476, y_st=831, y_ed=831 + 476)
    ]
    ANGLE_LIST = [0, 90, 180]
    BOUNDARY_CIRCLE_CENTER_COORD = Coordinate(1205, 1062)
    BOUNDARY_CIRCLE_RAD = 757


@dataclass
class VariableHousingCam3CropCRectangleTop:
    ENTIRE_X_ST = 386
    ENTIRE_X_ED = 1561 + 461
    ENTIRE_Y_ST = 182
    ENTIRE_Y_ED = 182 + 492

    __HALF_X = (ENTIRE_X_ST + ENTIRE_X_ED) // 2

    BBOX_LIST = [
        BoundingBox(x_st=ENTIRE_X_ST, x_ed=__HALF_X, y_st=ENTIRE_Y_ST, y_ed=ENTIRE_Y_ED),
        BoundingBox(x_st=__HALF_X, x_ed=ENTIRE_X_ED, y_st=ENTIRE_Y_ST, y_ed=ENTIRE_Y_ED)
    ]
    BOUNDARY_CIRCLE_CENTER_COORD = Coordinate(1205, 1062)
    BOUNDARY_CIRCLE_RAD = 757
    FLIP_INDEX = 0


# Old
@dataclass
class VariableHousingCam3CropCRectangleTopSide:
    BBOX_LIST = [
        BoundingBox(x_st=386, x_ed=386 + 461, y_st=182, y_ed=182 + 492),
        BoundingBox(x_st=1561, x_ed=1561 + 461, y_st=182, y_ed=182 + 492)
    ]
    BOUNDARY_CIRCLE_CENTER_COORD = Coordinate(1205, 1062)
    BOUNDARY_CIRCLE_RAD = 757
    FLIP_INDEX = 0


# Old
@dataclass
class VariableHousingCam3CropCRectangleTopMiddle:
    ENTIRE_X_ST = 837
    ENTIRE_X_ED = 837 + 736
    ENTIRE_Y_ST = 304
    ENTIRE_Y_ED = 304 + 95

    __HALF_X = (ENTIRE_X_ST + ENTIRE_X_ED) // 2

    BBOX_LIST = [
        BoundingBox(x_st=ENTIRE_X_ST, x_ed=__HALF_X, y_st=ENTIRE_Y_ST, y_ed=ENTIRE_Y_ED),
        BoundingBox(x_st=__HALF_X, x_ed=ENTIRE_X_ED, y_st=ENTIRE_Y_ST, y_ed=ENTIRE_Y_ED)
    ]
    BOUNDARY_CIRCLE_CENTER_COORD = Coordinate(1205, 1062)
    BOUNDARY_CIRCLE_RAD = 757
    FLIP_INDEX = 0


@dataclass
class VariableHousingCam3CropCRectangleRightDown:
    BBOX = BoundingBox(x_st=1525, x_ed=1525 + 435, y_st=1520, y_ed=1520 + 513)
    BOUNDARY_CIRCLE_CENTER_COORD = Coordinate(1205, 1062)
    BOUNDARY_CIRCLE_RAD = 757


"""
Grid
"""
@dataclass
class VariableHousingCam3GridEntire:
    BBOX_LIST = [
        # 1st layer
        BoundingBox(x_st=354, x_ed=354 + 512, y_st=178, y_ed=178 + 512),
        BoundingBox(x_st=673, x_ed=673 + 512, y_st=293, y_ed=293 + 512),
        BoundingBox(x_st=1140, x_ed=1140 + 512, y_st=293, y_ed=293 + 512),
        BoundingBox(x_st=1526, x_ed=1526 + 512, y_st=178, y_ed=178 + 512),

        # 2nd layer
        BoundingBox(x_st=116, x_ed=116 + 512, y_st=614, y_ed=614 + 512),
        BoundingBox(x_st=574, x_ed=574 + 512, y_st=614, y_ed=614 + 512),
        BoundingBox(x_st=965, x_ed=965 + 512, y_st=753, y_ed=753 + 512),
        BoundingBox(x_st=1345, x_ed=1345 + 512, y_st=614, y_ed=614 + 512),
        BoundingBox(x_st=1806, x_ed=1806 + 512, y_st=614, y_ed=614 + 512),

        # 3rd layer
        BoundingBox(x_st=116, x_ed=116 + 512, y_st=1038, y_ed=1038 + 512),
        BoundingBox(x_st=574, x_ed=574 + 512, y_st=1038, y_ed=1038 + 512),
        BoundingBox(x_st=965, x_ed=965 + 512, y_st=1190, y_ed=1190 + 512),
        BoundingBox(x_st=1345, x_ed=1345 + 512, y_st=1038, y_ed=1038 + 512),
        BoundingBox(x_st=1806, x_ed=1806 + 512, y_st=1038, y_ed=1038 + 512),

        # 4th layer
        BoundingBox(x_st=541, x_ed=541 + 512, y_st=1517, y_ed=1517 + 512),
        BoundingBox(x_st=997, x_ed=997 + 512, y_st=1649, y_ed=1649 + 512),
        BoundingBox(x_st=1455, x_ed=1455 + 512, y_st=1517, y_ed=1517 + 512)
    ]


@dataclass
class VariableHousingCam3Grid1:
    LEFT_UP_COORDINATE = Coordinate(x=354, y=178)
    BBOX_LIST = [
        BoundingBox(x_st=354, x_ed=354 + 512, y_st=178, y_ed=178 + 512)
    ]

@dataclass
class VariableHousingCam3Grid2:
    LEFT_UP_COORDINATE = Coordinate(x=673, y=293)
    BBOX_LIST = [
        BoundingBox(x_st=673, x_ed=673 + 512, y_st=293, y_ed=293 + 512)
    ]

@dataclass
class VariableHousingCam3Grid3:
    LEFT_UP_COORDINATE = Coordinate(x=1140, y=293)
    BBOX_LIST = [
        BoundingBox(x_st=1140, x_ed=1140 + 512, y_st=293, y_ed=293 + 512)
    ]

@dataclass
class VariableHousingCam3Grid4:
    LEFT_UP_COORDINATE = Coordinate(x=1526, y=178)
    BBOX_LIST = [
        BoundingBox(x_st=1526, x_ed=1526 + 512, y_st=178, y_ed=178 + 512)
    ]


@dataclass
class VariableHousingCam3Grid5:
    LEFT_UP_COORDINATE = Coordinate(x=116, y=614)
    BBOX_LIST = [
        BoundingBox(x_st=116, x_ed=116 + 512, y_st=614, y_ed=614 + 512)
    ]

@dataclass
class VariableHousingCam3Grid6:
    LEFT_UP_COORDINATE = Coordinate(x=574, y=614)
    BBOX_LIST = [
        BoundingBox(x_st=574, x_ed=574 + 512, y_st=614, y_ed=614 + 512)
    ]

@dataclass
class VariableHousingCam3Grid7:
    LEFT_UP_COORDINATE = Coordinate(x=965, y=753)
    BBOX_LIST = [
        BoundingBox(x_st=965, x_ed=965 + 512, y_st=753, y_ed=753 + 512)
    ]

@dataclass
class VariableHousingCam3Grid8:
    LEFT_UP_COORDINATE = Coordinate(x=1345, y=614)
    BBOX_LIST = [
        BoundingBox(x_st=1345, x_ed=1345 + 512, y_st=614, y_ed=614 + 512)
    ]

@dataclass
class VariableHousingCam3Grid9:
    LEFT_UP_COORDINATE = Coordinate(x=1806, y=614)
    BBOX_LIST = [
        BoundingBox(x_st=1806, x_ed=1806 + 512, y_st=614, y_ed=614 + 512)
    ]


@dataclass
class VariableHousingCam3Grid10:
    LEFT_UP_COORDINATE = Coordinate(x=116, y=1038)
    BBOX_LIST = [
        BoundingBox(x_st=116, x_ed=116 + 512, y_st=1038, y_ed=1038 + 512)
    ]

@dataclass
class VariableHousingCam3Grid11:
    LEFT_UP_COORDINATE = Coordinate(x=574, y=1038)
    BBOX_LIST = [
        BoundingBox(x_st=574, x_ed=574 + 512, y_st=1038, y_ed=1038 + 512)
    ]

@dataclass
class VariableHousingCam3Grid12:
    LEFT_UP_COORDINATE = Coordinate(x=965, y=1190)
    BBOX_LIST = [
        BoundingBox(x_st=965, x_ed=965 + 512, y_st=1190, y_ed=1190 + 512)
    ]

@dataclass
class VariableHousingCam3Grid13:
    LEFT_UP_COORDINATE = Coordinate(x=1345, y=1038)
    BBOX_LIST = [
        BoundingBox(x_st=1345, x_ed=1345 + 512, y_st=1038, y_ed=1038 + 512)
    ]

@dataclass
class VariableHousingCam3Grid14:
    LEFT_UP_COORDINATE = Coordinate(x=1806, y=1038)
    BBOX_LIST = [
        BoundingBox(x_st=1806, x_ed=1806 + 512, y_st=1038, y_ed=1038 + 512)
    ]


@dataclass
class VariableHousingCam3Grid15:
    LEFT_UP_COORDINATE = Coordinate(x=541, y=1517)
    BBOX_LIST = [
        BoundingBox(x_st=541, x_ed=541 + 512, y_st=1517, y_ed=1517 + 512)
    ]

@dataclass
class VariableHousingCam3Grid16:
    LEFT_UP_COORDINATE = Coordinate(x=997, y=1649)
    BBOX_LIST = [
        BoundingBox(x_st=997, x_ed=997 + 512, y_st=1649, y_ed=1649 + 512)
    ]

@dataclass
class VariableHousingCam3Grid17:
    LEFT_UP_COORDINATE = Coordinate(x=1455, y=1517)
    BBOX_LIST = [
        BoundingBox(x_st=1455, x_ed=1455 + 512, y_st=1517, y_ed=1517 + 512)
    ]


CAM3_AREA = Cam3Area()

# Test
CROP_ACD_RING_VARIABLE = VariableHousingCam3CropACDRing()

CROP_C_INNER_RING_VARIABLE = VariableHousingCam3CropCInnerRing()
CROP_C_INNER_RING_ENTIRE_VARIABLE = VariableHousingCam3CropCInnerRingEntire()
CROP_C_INNER_RING_INSIDE_ENTIRE_VARIABLE = VariableHousingCam3CropCInnerRingInsideEntire()
CROP_C_INNER_RING_OUTSIDE_ENTIRE_VARIABLE = VariableHousingCam3CropCInnerRingOutsideEntire()
CROP_D_RING_VARIABLE = VariableHousingCam3CropDRing()
CROP_D_RING_ENTIRE_VARIABLE = VariableHousingCam3CropDRingEntire()
CROP_A_VARIABLE = VariableHousingCam3CropA()
CROP_A_RING_ENTIRE_VARIABLE = VariableHousingCam3CropARingEntire()
CROP_C_OUTER_ENTIRE_VARIABLE = VariableHousingCam3CropCOuterEntire()
CROP_C_OUTER_RING_VARIABLE = VariableHousingCam3CropCOuterRing()
CROP_C_RECTANGLE_WITH_RING_VARIABLE = VariableHousingCam3CropCRectangleWithRing()
CROP_C_RECTANGLE_TOP_VARIABLE = VariableHousingCam3CropCRectangleTop()
CROP_C_RECTANGLE_TOP_SIDE_VARIABLE = VariableHousingCam3CropCRectangleTopSide()
CROP_C_RECTANGLE_TOP_MIDDLE_VARIABLE = VariableHousingCam3CropCRectangleTopMiddle()
CROP_C_RECTANGLE_RIGHT_DOWN_VARIABLE = VariableHousingCam3CropCRectangleRightDown()

# Grid Test
CROP_GRID_ENTIRE_VARIABLE = VariableHousingCam3GridEntire()

CROP_GRID_1_VARIABLE = VariableHousingCam3Grid1()
CROP_GRID_2_VARIABLE = VariableHousingCam3Grid2()
CROP_GRID_3_VARIABLE = VariableHousingCam3Grid3()
CROP_GRID_4_VARIABLE = VariableHousingCam3Grid4()

CROP_GRID_5_VARIABLE = VariableHousingCam3Grid5()
CROP_GRID_6_VARIABLE = VariableHousingCam3Grid6()
CROP_GRID_7_VARIABLE = VariableHousingCam3Grid7()
CROP_GRID_8_VARIABLE = VariableHousingCam3Grid8()
CROP_GRID_9_VARIABLE = VariableHousingCam3Grid9()

CROP_GRID_10_VARIABLE = VariableHousingCam3Grid10()
CROP_GRID_11_VARIABLE = VariableHousingCam3Grid11()
CROP_GRID_12_VARIABLE = VariableHousingCam3Grid12()
CROP_GRID_13_VARIABLE = VariableHousingCam3Grid13()
CROP_GRID_14_VARIABLE = VariableHousingCam3Grid14()

CROP_GRID_15_VARIABLE = VariableHousingCam3Grid15()
CROP_GRID_16_VARIABLE = VariableHousingCam3Grid16()
CROP_GRID_17_VARIABLE = VariableHousingCam3Grid17()
