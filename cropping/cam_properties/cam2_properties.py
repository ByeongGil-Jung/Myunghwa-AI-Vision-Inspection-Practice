from dataclasses import dataclass

from domain.image import BoundingBox


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


CROP_C_RECTANGLE_SIDE_VARIABLE = VariableHousingCam2CropCRectangleSide()
CROP_C_RECTANGLE_MIDDLE_VARIABLE = VariableHousingCam2CropCRectangleMiddle()
