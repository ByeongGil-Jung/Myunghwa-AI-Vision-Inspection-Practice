from dataclasses import dataclass

from domain.image import Area, BoundingBox, Coordinate


@dataclass
class Cam4Area:
    C = Area("C")

    @classmethod
    def get_list(cls):
        return [cls.C]

    @classmethod
    def get_list_dict(cls):
        return {area.name: list() for area in cls.get_list()}


@dataclass
class VariableHousingCam4CEntire:
    BBOX = BoundingBox(x_st=40, x_ed=40 + 128, y_st=0, y_ed=4230)


@dataclass
class VariableHousingCam4Grid1:
    FIRST_CROP_X_ST = 40
    CROP_SIZE = 128
    FIRST_CROP_Y_ST = 1579
    CROP_EMPTY_SPACE_Y_GAP = 622
    CROP_Y_GAP = (128 / 4 * 3)  # crop size / 4 * 3

    LEFT_UP_COORDINATE = Coordinate(x=40, y=0)


CAM4_AREA = Cam4Area()

CROP_C_ENTIRE_VARIABLE = VariableHousingCam4CEntire()

# Grid
CROP_GRID_1_VARIABLE = VariableHousingCam4Grid1()
