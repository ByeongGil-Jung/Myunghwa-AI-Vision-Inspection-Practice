import sys
from properties import CROPPING_PROPERTIES

sys.path.append(CROPPING_PROPERTIES.HOME_MODULE_PATH)

from pathlib import Path
import os
import shutil

from cropping.cam_properties import cam2_properties as c2p
from cropping.cropper.housing.cam1_cropper import *
from cropping.cropper.housing.cam2_cropper import *
from cropping.cropper.housing.cam3_cropper import *
from cropping.cropper.housing.cam4_cropper import *
# from cropping.utils import get_cropping_properties, get_cropper
from domain.metadata import ImageMetadata, ModelMetadata


class CropperFactory(object):

    @classmethod
    def get(cls, image_metadata: ImageMetadata, img_object: EOPImage, defect_list: list, is_mask=False):
        product = image_metadata.product
        cam_number = image_metadata.cam_number
        crop_part = image_metadata.crop_part

        cropper: CropperBase = None

        if product == "housing":
            if cam_number == 1:
                if crop_part == "C":
                    cropper = HousingCam1CropperC(img_object, defect_list)
                elif crop_part == "D_inner_ring":
                    cropper = HousingCam1CropperDInnerRing(img_object, defect_list)
                elif crop_part == "A":
                    cropper = HousingCam1CropperA(img_object, defect_list)
                elif crop_part == "D_outer_ring":
                    cropper = HousingCam1CropperDOuterRing(img_object, defect_list)
                elif crop_part == "D_outer_rectangle_with_ring":
                    cropper = HousingCam1CropperDOuterRectangleWithRing(img_object, defect_list)
                elif crop_part == "D_outer_rectangle_top":
                    cropper = HousingCam1CropperDOuterRectangleTop(img_object, defect_list)

                # For test
                elif crop_part == "C_entire":
                    cropper = HousingCam1CropperCEntire(img_object, defect_list)
                elif crop_part == "D_ring":
                    cropper = HousingCam1CropperDRing(img_object, defect_list)
                elif crop_part == "A_ring_entire":
                    cropper = HousingCam1CropperARingEntire(img_object, defect_list)
                elif crop_part == "D_inner_ring_entire":
                    cropper = HousingCam1CropperDInnerRingEntire(img_object, defect_list)
                elif crop_part == "D_outer_entire":
                    cropper = HousingCam1CropperDOuterEntire(img_object, defect_list)
                elif crop_part == "D_outer_ring_inside":
                    cropper = HousingCam1CropperDOuterRingInside(img_object, defect_list)
                elif crop_part == "D_outer_ring_outside":
                    cropper = HousingCam1CropperDOuterRingOutside(img_object, defect_list)

                # Grid
                elif crop_part.split("_")[0] == "grid":
                    if is_mask:
                        cropper = HousingCam1CropperGrid
                    else:
                        cropper = HousingCam1CropperGrid(img_object, defect_list)
            elif cam_number == 2:
                if crop_part == "C_rectangle_side":
                    cropper = HousingCam2CropperC(img_object, defect_list)
                elif crop_part == "C_rectangle_middle":
                    cropper = HousingCam2CropperC(img_object, defect_list)
            elif cam_number == 3:
                if crop_part == "C_inner_ring":
                    cropper = HousingCam3CropperCInnerRing(img_object, defect_list)
                # Test
                elif crop_part == "C_inner_ring_entire":
                    cropper = HousingCam3CropperCInnerRingEntire(img_object, defect_list)
                elif crop_part == "C_inner_ring_inside_entire":
                    cropper = HousingCam3CropperCInnerRingInsideEntire(img_object, defect_list)
                elif crop_part == "C_inner_ring_outside_entire":
                    cropper = HousingCam3CropperCInnerRingOutsideEntire(img_object, defect_list)
                elif crop_part == "ACD_ring":
                    cropper = HousingCam3CropperACDRing(img_object, defect_list)

                elif crop_part == "D_ring":
                    cropper = HousingCam3CropperDRing(img_object, defect_list)
                elif crop_part == "D_ring_entire":
                    cropper = HousingCam3CropperDRingEntire(img_object, defect_list)
                elif crop_part == "A":
                    cropper = HousingCam3CropperA(img_object, defect_list)
                elif crop_part == "A_ring_entire":
                    cropper = HousingCam3CropperARingEntire(img_object, defect_list)
                elif crop_part == "C_outer_entire":
                    cropper = HousingCam3CropperCOuterEntire(img_object, defect_list)
                elif crop_part == "C_outer_ring":
                    cropper = HousingCam3CropperCOuterRing(img_object, defect_list)
                elif crop_part == "C_rectangle_with_ring":
                    cropper = HousingCam3CropperCRectangleWithRing(img_object, defect_list)
                elif crop_part == "C_rectangle_top":
                    cropper = HousingCam3CropperCRectangleTop(img_object, defect_list)
                elif crop_part == "C_rectangle_top_side":
                    cropper = HousingCam3CropperCRectangleTop(img_object, defect_list)
                elif crop_part == "C_rectangle_top_middle":
                    cropper = HousingCam3CropperCRectangleTop(img_object, defect_list)
                elif crop_part == "C_rectangle_right_down":
                    cropper = HousingCam3CropperCRectangleRightDown(img_object, defect_list)

                # Grid
                elif crop_part.split("_")[0] == "grid":
                    if is_mask:
                        cropper = HousingCam3CropperGrid
                    else:
                        cropper = HousingCam3CropperGrid(img_object, defect_list)
            elif cam_number == 4:
                if crop_part == "C":
                    cropper = HousingCam4CropperC(img_object, defect_list)
        elif product == "cover":
            if cam_number == 1:
                pass
            elif cam_number == 2:
                pass
            elif cam_number == 3:
                pass
            elif cam_number == 4:
                pass

        return cropper

    @classmethod
    def get_properties(cls, image_metadata: ImageMetadata):
        product = image_metadata.product
        cam_number = image_metadata.cam_number
        crop_part = image_metadata.crop_part

        properties: dict = dict()

        if product == "housing":
            if cam_number == 1:
                if crop_part == "C":
                    properties = dict(
                        big_rad=c1p.CROP_C_VARIABLE.BIG_RAD,
                        small_rad=c1p.CROP_C_VARIABLE.SMALL_RAD,
                        center_coord=c1p.CROP_C_VARIABLE.CENTER_COORD,
                        angle=c1p.CROP_C_VARIABLE.ANGLE
                    )
                ###
                # Test
                elif crop_part == "C_entire":
                    properties = dict(
                        rad=c1p.CROP_C_ENTIRE_VARIABLE.RAD,
                        center_coord=c1p.CROP_C_ENTIRE_VARIABLE.CENTER_COORD
                    )
                elif crop_part == "D_ring":
                    properties = dict(
                        inner_circle_rad=c1p.CROP_D_RING_VARIABLE.INNER_CIRCLE_RAD,
                        inner_circle_center_coord=c1p.CROP_D_RING_VARIABLE.INNER_CIRCLE_CENTER_COORD,
                        outer_circle_rad=c1p.CROP_D_RING_VARIABLE.OUTER_CIRCLE_RAD,
                        outer_circle_center_coord=c1p.CROP_D_RING_VARIABLE.OUTER_CIRCLE_CENTER_COORD,
                        ring_small_rad=c1p.CROP_D_RING_VARIABLE.RING_SMALL_RAD,
                        angle=c1p.CROP_D_RING_VARIABLE.ANGLE
                    )
                elif crop_part == "A_ring_entire":
                    properties = dict(
                        big_rad=c1p.CROP_A_VARIABLE.BIG_RAD,
                        small_rad=c1p.CROP_A_VARIABLE.SMALL_RAD,
                        center_coord=c1p.CROP_A_VARIABLE.CENTER_COORD
                    )
                elif crop_part == "D_inner_ring_entire":
                    properties = dict(
                        inner_circle_rad=c1p.CROP_D_INNER_RING_VARIABLE.INNER_CIRCLE_RAD,
                        inner_circle_center_coord=c1p.CROP_D_INNER_RING_VARIABLE.INNER_CIRCLE_CENTER_COORD,
                        outer_circle_rad=c1p.CROP_D_INNER_RING_VARIABLE.OUTER_CIRCLE_RAD,
                        outer_circle_center_coord=c1p.CROP_D_INNER_RING_VARIABLE.OUTER_CIRCLE_CENTER_COORD,
                        ring_small_rad=c1p.CROP_D_INNER_RING_VARIABLE.RING_SMALL_RAD
                    )
                elif crop_part == "D_outer_ring_inside":
                    properties = dict(
                        big_rad=c1p.CROP_D_OUTER_RING_INSIDE_VARIABLE.BIG_RAD,
                        small_rad=c1p.CROP_D_OUTER_RING_INSIDE_VARIABLE.SMALL_RAD,
                        center_coord=c1p.CROP_D_OUTER_RING_INSIDE_VARIABLE.CENTER_COORD,
                        angle=c1p.CROP_D_OUTER_RING_INSIDE_VARIABLE.ANGLE
                    )
                elif crop_part == "D_outer_ring_outside":
                    properties = dict(
                        big_rad=c1p.CROP_D_OUTER_RING_OUTSIDE_VARIABLE.BIG_RAD,
                        small_rad=c1p.CROP_D_OUTER_RING_OUTSIDE_VARIABLE.SMALL_RAD,
                        center_coord=c1p.CROP_D_OUTER_RING_OUTSIDE_VARIABLE.CENTER_COORD,
                        angle=c1p.CROP_D_OUTER_RING_OUTSIDE_VARIABLE.ANGLE,
                        index_list=c1p.CROP_D_OUTER_RING_OUTSIDE_VARIABLE.INDEX_LIST
                    )
                ###
                elif crop_part == "D_inner_ring":
                    properties = dict(
                        inner_circle_rad=c1p.CROP_D_INNER_RING_VARIABLE.INNER_CIRCLE_RAD,
                        inner_circle_center_coord=c1p.CROP_D_INNER_RING_VARIABLE.INNER_CIRCLE_CENTER_COORD,
                        outer_circle_rad=c1p.CROP_D_INNER_RING_VARIABLE.OUTER_CIRCLE_RAD,
                        outer_circle_center_coord=c1p.CROP_D_INNER_RING_VARIABLE.OUTER_CIRCLE_CENTER_COORD,
                        ring_small_rad=c1p.CROP_D_INNER_RING_VARIABLE.RING_SMALL_RAD,
                        angle=c1p.CROP_D_INNER_RING_VARIABLE.ANGLE
                    )
                elif crop_part == "A":
                    properties = dict(
                        big_rad=c1p.CROP_A_VARIABLE.BIG_RAD,
                        small_rad=c1p.CROP_A_VARIABLE.SMALL_RAD,
                        center_coord=c1p.CROP_A_VARIABLE.CENTER_COORD,
                        angle=c1p.CROP_A_VARIABLE.ANGLE
                    )
                elif crop_part == "D_outer_ring":
                    properties = dict(
                        big_rad=c1p.CROP_D_OUTER_RING_VARIABLE.BIG_RAD,
                        small_rad=c1p.CROP_D_OUTER_RING_VARIABLE.SMALL_RAD,
                        center_coord=c1p.CROP_D_OUTER_RING_VARIABLE.CENTER_COORD,
                        angle=c1p.CROP_D_OUTER_RING_VARIABLE.ANGLE,
                        index_list=c1p.CROP_D_OUTER_RING_VARIABLE.INDEX_LIST
                    )
                elif crop_part == "D_outer_entire":
                    properties = dict(
                        rad=c1p.CROP_D_OUTER_ENTIRE_VARIABLE.RAD,
                        center_coord=c1p.CROP_D_OUTER_ENTIRE_VARIABLE.CENTER_COORD
                    )
                elif crop_part == "D_outer_rectangle_with_ring":
                    properties = dict(
                        bbox_list=c1p.CROP_D_OUTER_RECTANGLE_WITH_RING_VARIABLE.BBOX_LIST,
                        angle_list=c1p.CROP_D_OUTER_RECTANGLE_WITH_RING_VARIABLE.ANGLE_LIST,
                        boundary_circle_center_coord=c1p.CROP_D_OUTER_RECTANGLE_WITH_RING_VARIABLE.BOUNDARY_CIRCLE_CENTER_COORD,
                        boundary_circle_rad=c1p.CROP_D_OUTER_RECTANGLE_WITH_RING_VARIABLE.BOUNDARY_CIRCLE_RAD
                    )
                elif crop_part == "D_outer_rectangle_top":
                    properties = dict(
                        bbox_list=c1p.CROP_D_OUTER_RECTANGLE_TOP_VARIABLE.BBOX_LIST,
                        boundary_circle_center_coord=c1p.CROP_D_OUTER_RECTANGLE_TOP_VARIABLE.BOUNDARY_CIRCLE_CENTER_COORD,
                        boundary_circle_rad=c1p.CROP_D_OUTER_RECTANGLE_TOP_VARIABLE.BOUNDARY_CIRCLE_RAD,
                        flip_index=c1p.CROP_D_OUTER_RECTANGLE_TOP_VARIABLE.FLIP_INDEX
                    )

                # Grid
                elif crop_part == "grid_entire":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_ENTIRE_VARIABLE.BBOX_LIST
                    )

                elif crop_part == "grid_1":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_1_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_2":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_2_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_3":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_3_VARIABLE.BBOX_LIST
                    )

                elif crop_part == "grid_4":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_4_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_5":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_5_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_6":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_6_VARIABLE.BBOX_LIST
                    )

                elif crop_part == "grid_7":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_7_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_8":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_8_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_9":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_9_VARIABLE.BBOX_LIST
                    )

                elif crop_part == "grid_10":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_10_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_11":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_11_VARIABLE.BBOX_LIST
                    )

                elif crop_part == "grid_12":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_12_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_13":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_13_VARIABLE.BBOX_LIST
                    )

                elif crop_part == "grid_14":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_14_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_15":
                    properties = dict(
                        bbox_list=c1p.CROP_GRID_15_VARIABLE.BBOX_LIST
                    )
            elif cam_number == 2:
                if crop_part == "C_rectangle_side":
                    properties = dict(
                        bbox_list=c2p.CROP_C_RECTANGLE_SIDE_VARIABLE.BBOX_LIST,
                        flip_index=c2p.CROP_C_RECTANGLE_SIDE_VARIABLE.FLIP_INDEX
                    )
                elif crop_part == "C_rectangle_middle":
                    properties = dict(
                        bbox_list=c2p.CROP_C_RECTANGLE_MIDDLE_VARIABLE.BBOX_LIST,
                        flip_index=c2p.CROP_C_RECTANGLE_MIDDLE_VARIABLE.FLIP_INDEX
                    )
            elif cam_number == 3:
                if crop_part == "C_inner_ring":
                    properties = dict(
                        big_rad=c3p.CROP_C_INNER_RING_VARIABLE.BIG_RAD,
                        small_rad=c3p.CROP_C_INNER_RING_VARIABLE.SMALL_RAD,
                        center_coord=c3p.CROP_C_INNER_RING_VARIABLE.CENTER_COORD,
                        angle=c3p.CROP_C_INNER_RING_VARIABLE.ANGLE
                    )
                elif crop_part == "C_inner_ring_entire":
                    properties = dict(
                        rad=c3p.CROP_C_INNER_RING_ENTIRE_VARIABLE.RAD,
                        center_coord=c3p.CROP_C_INNER_RING_ENTIRE_VARIABLE.CENTER_COORD
                    )
                elif crop_part == "C_inner_ring_inside_entire":
                    properties = dict(
                        rad=c3p.CROP_C_INNER_RING_INSIDE_ENTIRE_VARIABLE.RAD,
                        center_coord=c3p.CROP_C_INNER_RING_INSIDE_ENTIRE_VARIABLE.CENTER_COORD
                    )
                elif crop_part == "C_inner_ring_outside_entire":
                    properties = dict(
                        big_rad=c3p.CROP_C_INNER_RING_OUTSIDE_ENTIRE_VARIABLE.BIG_RAD,
                        small_rad=c3p.CROP_C_INNER_RING_OUTSIDE_ENTIRE_VARIABLE.SMALL_RAD,
                        center_coord=c3p.CROP_C_INNER_RING_OUTSIDE_ENTIRE_VARIABLE.CENTER_COORD
                    )
                elif crop_part == "ACD_ring":
                    properties = dict(
                        big_rad=c3p.CROP_ACD_RING_VARIABLE.BIG_RAD,
                        small_rad=c3p.CROP_ACD_RING_VARIABLE.SMALL_RAD,
                        center_coord=c3p.CROP_ACD_RING_VARIABLE.CENTER_COORD,
                        angle=c3p.CROP_ACD_RING_VARIABLE.ANGLE
                    )
                elif crop_part == "D_ring":
                    properties = dict(
                        big_rad=c3p.CROP_D_RING_VARIABLE.BIG_RAD,
                        small_rad=c3p.CROP_D_RING_VARIABLE.SMALL_RAD,
                        center_coord=c3p.CROP_D_RING_VARIABLE.CENTER_COORD,
                        angle=c3p.CROP_D_RING_VARIABLE.ANGLE
                    )
                elif crop_part == "D_ring_entire":
                    properties = dict(
                        big_rad=c3p.CROP_D_RING_ENTIRE_VARIABLE.BIG_RAD,
                        small_rad=c3p.CROP_D_RING_ENTIRE_VARIABLE.SMALL_RAD,
                        center_coord=c3p.CROP_D_RING_ENTIRE_VARIABLE.CENTER_COORD
                    )
                elif crop_part == "A":
                    properties = dict(
                        big_rad=c3p.CROP_A_VARIABLE.BIG_RAD,
                        small_rad=c3p.CROP_A_VARIABLE.SMALL_RAD,
                        center_coord=c3p.CROP_A_VARIABLE.CENTER_COORD,
                        angle=c3p.CROP_A_VARIABLE.ANGLE
                    )
                elif crop_part == "A_ring_entire":
                    properties = dict(
                        big_rad=c3p.CROP_A_RING_ENTIRE_VARIABLE.BIG_RAD,
                        small_rad=c3p.CROP_A_RING_ENTIRE_VARIABLE.SMALL_RAD,
                        center_coord=c3p.CROP_A_RING_ENTIRE_VARIABLE.CENTER_COORD
                    )
                elif crop_part == "C_outer_entire":
                    properties = dict(
                        rad=c3p.CROP_C_OUTER_ENTIRE_VARIABLE.RAD,
                        center_coord=c3p.CROP_C_OUTER_ENTIRE_VARIABLE.CENTER_COORD
                    )
                elif crop_part == "C_outer_ring":
                    properties = dict(
                        big_rad=c3p.CROP_C_OUTER_RING_VARIABLE.BIG_RAD,
                        small_rad=c3p.CROP_C_OUTER_RING_VARIABLE.SMALL_RAD,
                        center_coord=c3p.CROP_C_OUTER_RING_VARIABLE.CENTER_COORD,
                        angle=c3p.CROP_C_OUTER_RING_VARIABLE.ANGLE,
                        index_list=c3p.CROP_C_OUTER_RING_VARIABLE.INDEX_LIST
                    )
                elif crop_part == "C_rectangle_with_ring":
                    properties = dict(
                        bbox_list=c3p.CROP_C_RECTANGLE_WITH_RING_VARIABLE.BBOX_LIST,
                        angle_list=c3p.CROP_C_RECTANGLE_WITH_RING_VARIABLE.ANGLE_LIST,
                        boundary_circle_center_coord=c3p.CROP_C_RECTANGLE_WITH_RING_VARIABLE.BOUNDARY_CIRCLE_CENTER_COORD,
                        boundary_circle_rad=c3p.CROP_C_RECTANGLE_WITH_RING_VARIABLE.BOUNDARY_CIRCLE_RAD
                    )
                elif crop_part == "C_rectangle_top":
                    properties = dict(
                        bbox_list=c3p.CROP_C_RECTANGLE_TOP_VARIABLE.BBOX_LIST,
                        boundary_circle_center_coord=c3p.CROP_C_RECTANGLE_TOP_VARIABLE.BOUNDARY_CIRCLE_CENTER_COORD,
                        boundary_circle_rad=c3p.CROP_C_RECTANGLE_TOP_VARIABLE.BOUNDARY_CIRCLE_RAD,
                        flip_index=c3p.CROP_C_RECTANGLE_TOP_VARIABLE.FLIP_INDEX
                    )
                elif crop_part == "C_rectangle_top_side":  # old
                    properties = dict(
                        bbox_list=c3p.CROP_C_RECTANGLE_TOP_SIDE_VARIABLE.BBOX_LIST,
                        boundary_circle_center_coord=c3p.CROP_C_RECTANGLE_TOP_SIDE_VARIABLE.BOUNDARY_CIRCLE_CENTER_COORD,
                        boundary_circle_rad=c3p.CROP_C_RECTANGLE_TOP_SIDE_VARIABLE.BOUNDARY_CIRCLE_RAD,
                        flip_index=c3p.CROP_C_RECTANGLE_TOP_SIDE_VARIABLE.FLIP_INDEX
                    )
                elif crop_part == "C_rectangle_top_middle":  # old
                    properties = dict(
                        bbox_list=c3p.CROP_C_RECTANGLE_TOP_MIDDLE_VARIABLE.BBOX_LIST,
                        boundary_circle_center_coord=c3p.CROP_C_RECTANGLE_TOP_MIDDLE_VARIABLE.BOUNDARY_CIRCLE_CENTER_COORD,
                        boundary_circle_rad=c3p.CROP_C_RECTANGLE_TOP_MIDDLE_VARIABLE.BOUNDARY_CIRCLE_RAD,
                        flip_index=c3p.CROP_C_RECTANGLE_TOP_MIDDLE_VARIABLE.FLIP_INDEX
                    )
                elif crop_part == "C_rectangle_right_down":
                    properties = dict(
                        bbox=c3p.CROP_C_RECTANGLE_RIGHT_DOWN_VARIABLE.BBOX,
                        boundary_circle_center_coord=c3p.CROP_C_RECTANGLE_RIGHT_DOWN_VARIABLE.BOUNDARY_CIRCLE_CENTER_COORD,
                        boundary_circle_rad=c3p.CROP_C_RECTANGLE_RIGHT_DOWN_VARIABLE.BOUNDARY_CIRCLE_RAD
                    )

                # Grid
                elif crop_part == "grid_entire":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_ENTIRE_VARIABLE.BBOX_LIST
                    )

                elif crop_part == "grid_1":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_1_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_2":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_2_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_3":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_3_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_4":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_4_VARIABLE.BBOX_LIST
                    )

                elif crop_part == "grid_5":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_5_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_6":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_6_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_7":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_7_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_8":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_8_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_9":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_9_VARIABLE.BBOX_LIST
                    )

                elif crop_part == "grid_10":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_10_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_11":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_11_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_12":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_12_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_13":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_13_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_14":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_14_VARIABLE.BBOX_LIST
                    )

                elif crop_part == "grid_15":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_15_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_16":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_16_VARIABLE.BBOX_LIST
                    )
                elif crop_part == "grid_17":
                    properties = dict(
                        bbox_list=c3p.CROP_GRID_17_VARIABLE.BBOX_LIST
                    )
            elif cam_number == 4:
                if crop_part == "C":
                    properties = dict()
        elif product == "cover":
            if cam_number == 1:
                pass
            elif cam_number == 2:
                pass
            elif cam_number == 3:
                pass
            elif cam_number == 4:
                pass

        return properties


class CroppingAreaFactory(object):

    HOUSING_CAM2_AREA_LIST = ["C_inner", "C_outer"]
    HOUSING_CAM4_AREA_LIST = ["C"]

    CROPPING_AREA_PIXEL_LOWER_BOUND = 15

    def __init__(self):
        pass

    @classmethod
    def get_area_list(cls, image_metadata: ImageMetadata):
        product = image_metadata.product
        cam_number = image_metadata.cam_number
        crop_part = image_metadata.crop_part
        area_list = list()

        if product == "housing":
            if cam_number == 1:
                area_list = c1p.CAM1_AREA.get_list()
            elif cam_number == 2:
                area_list = CroppingAreaFactory.HOUSING_CAM2_AREA_LIST
            elif cam_number == 3:
                area_list = c3p.CAM3_AREA.get_list()
            elif cam_number == 4:
                area_list = CroppingAreaFactory.HOUSING_CAM4_AREA_LIST

        return area_list

    @classmethod
    def get_area_mask_home_dir_path(cls, product, cam_number):
        area_mask_home_dir_path = ""

        if product == "housing":
            if cam_number == 1:
                area_mask_home_dir_path = CROPPING_PROPERTIES.HOUSING_CAM1_AREA_MASK_IMG_HOME_DIR_PATH
            elif cam_number == 2:
                area_mask_home_dir_path = CROPPING_PROPERTIES.HOUSING_CAM2_AREA_MASK_IMG_HOME_DIR_PATH
            elif cam_number == 3:
                area_mask_home_dir_path = CROPPING_PROPERTIES.HOUSING_CAM3_AREA_MASK_IMG_HOME_DIR_PATH
            elif cam_number == 4:
                area_mask_home_dir_path = CROPPING_PROPERTIES.HOUSING_CAM4_AREA_MASK_IMG_HOME_DIR_PATH

        return area_mask_home_dir_path

    @classmethod
    def get_area_info_dict(cls, image_metadata: ImageMetadata, img_object):
        product = image_metadata.product
        cam_number = image_metadata.cam_number
        crop_part = image_metadata.crop_part
        area_list = cls.get_area_list(image_metadata=image_metadata)
        area_info_dict = {area.name: dict() for area in area_list}

        for area in area_list:
            area_name = area.name
            area_img_object = cls.get_area_img(image_metadata=image_metadata, area=area, img_object=img_object)
            area_img = area_img_object.img
            pos_crop_img = area_img[np.where(area_img > CroppingAreaFactory.CROPPING_AREA_PIXEL_LOWER_BOUND)]

            pos_crop_img_mean = np.mean(pos_crop_img)

            area_info_dict[area_name]["img_object"] = area_img_object
            area_info_dict[area_name]["mean"] = pos_crop_img_mean

        # 예외 및 통합 처리
        if product == "housing":
            if cam_number == 1:
                area_info_dict["D"] = dict()

                D_inner_img = area_info_dict["D_inner"]["img_object"].img
                D_outer_img = area_info_dict["D_outer"]["img_object"].img

                D_inner_pos_img = D_inner_img[np.where(D_inner_img > CroppingAreaFactory.CROPPING_AREA_PIXEL_LOWER_BOUND)]
                D_outer_pos_img = D_outer_img[np.where(D_outer_img > CroppingAreaFactory.CROPPING_AREA_PIXEL_LOWER_BOUND)]

                D_pos_img = np.concatenate([D_inner_pos_img, D_outer_pos_img], axis=0)

                area_info_dict["D"]["img_object"] = None
                area_info_dict["D"]["mean"] = np.mean(D_pos_img)
        
        # img_object 제거
        for area_name in area_info_dict.keys():
            del area_info_dict[area_name]["img_object"]

        return area_info_dict

    @classmethod
    def get_area_grid_mask_img(cls, image_metadata: ImageMetadata, img_object, resize_rate=1.0, is_removed=False, is_saved=False):
        product = image_metadata.product
        cam_number = image_metadata.cam_number
        crop_part = image_metadata.crop_part

        area_list = cls.get_area_list(image_metadata=image_metadata)
        area_mask_home_dir_path = cls.get_area_mask_home_dir_path(product=product, cam_number=cam_number)
        grid_num = CropperBase.get_grid_num(product=product, cam_number=cam_number)
        grid_area_dict = dict()

        if is_removed:
            if os.path.isdir(area_mask_home_dir_path):
                shutil.rmtree(area_mask_home_dir_path)

        for area in area_list:
            grid_area_dict[area.name] = dict()

            area_mask_dir_path = os.path.join(area_mask_home_dir_path, area.name)
            Path(area_mask_dir_path).mkdir(parents=True, exist_ok=True)

            area_img_object = CroppingAreaFactory.get_area_img(
                image_metadata=image_metadata,
                area=area.name,
                img_object=img_object,
                is_preserved_entire_img_shape=True
            )

            for grid_idx in range(grid_num):
                grid_name = f"grid_{grid_idx + 1}"
                model_data = ModelMetadata(image_metadata=image_metadata, version="none_0")

                grid_cropping_properties = CropperFactory.get_properties(image_metadata=image_metadata)
                grid_cropper = CropperFactory.get(image_metadata=image_metadata, img_object=area_img_object, defect_list=list())

                grid_img_object = grid_cropper.crop(**grid_cropping_properties)[0]

                if resize_rate is not None:
                    grid_img_object.img = cv2.resize(
                        grid_img_object.img, dsize=(0, 0), fx=resize_rate, fy=resize_rate, interpolation=cv2.INTER_AREA
                    )

                if is_saved:
                    save_path = os.path.join(area_mask_dir_path, f"{grid_name}.png")
                    plt.imsave(save_path, grid_img_object.img, cmap="gray")

                grid_area_dict[area.name][grid_idx + 1] = grid_img_object

        return grid_area_dict

    @classmethod
    def get_area_img(cls, image_metadata: ImageMetadata, area, img_object: EOPImage, is_preserved_entire_img_shape=False):
        product = image_metadata.product
        cam_number = image_metadata.cam_number
        crop_part = image_metadata.crop_part

        area_name = area.name
        model_data = None
        area_img = None
        is_entire_img_shape = False

        if product == "housing":
            if cam_number == 1:
                if area_name == "A":
                    model_data = ModelMetadata(ImageMetadata(product=product, cam_number=cam_number, crop_part="A_ring_entire"), version="none_0")
                elif area_name == "C":
                    model_data = ModelMetadata(ImageMetadata(product=product, cam_number=cam_number, crop_part="C_entire"), version="none_0")
                elif area_name == "D_inner":
                    model_data = ModelMetadata(ImageMetadata(product=product, cam_number=cam_number, crop_part="D_inner_ring_entire"), version="none_0")
                elif area_name == "D_outer":
                    model_data = ModelMetadata(ImageMetadata(product=product, cam_number=cam_number, crop_part="D_outer_entire"), version="none_0")
                    is_entire_img_shape = True
            elif cam_number == 2:
                if area_name == "C_inner":
                    model_data = ModelMetadata(ImageMetadata(product=product, cam_number=cam_number, crop_part="C_inner_entire"), version="none_0")
                elif area_name == "C_outer":
                    model_data = ModelMetadata(ImageMetadata(product=product, cam_number=cam_number, crop_part="C_outer_entire"), version="none_0")
            elif cam_number == 3:
                if area_name == "A":
                    model_data = ModelMetadata(ImageMetadata(product=product, cam_number=cam_number, crop_part="A_ring_entire"), version="none_0")
                # elif area_name == "C_inner":
                #     model_data = ModelData(product=product, cam_number=cam_number, crop_part="C_inner_ring_entire", version="none_0")
                elif area_name == "C_inner_ring_inside":
                    model_data = ModelMetadata(ImageMetadata(product=product, cam_number=cam_number, crop_part="C_inner_ring_inside_entire"), version="none_0")
                elif area_name == "C_inner_ring_outside":
                    model_data = ModelMetadata(ImageMetadata(product=product, cam_number=cam_number, crop_part="C_inner_ring_outside_entire"), version="none_0")
                elif area_name == "C_outer":
                    model_data = ModelMetadata(ImageMetadata(product=product, cam_number=cam_number, crop_part="C_outer_entire"), version="none_0")
                    is_entire_img_shape = True
                elif area_name == "D":
                    model_data = ModelMetadata(ImageMetadata(product=product, cam_number=cam_number, crop_part="D_ring_entire"), version="none_0")
            elif cam_number == 4:
                if area_name == "C":
                    model_data = ModelMetadata(ImageMetadata(product=product, cam_number=cam_number, crop_part="C_entire"), version="none_0")
        elif product == "cover":
            if cam_number == 1:
                pass
            elif cam_number == 2:
                pass
            elif cam_number == 3:
                pass
            elif cam_number == 4:
                pass

        cropping_properties = CropperFactory.get_properties(image_metadata=image_metadata)
        cropper = CropperFactory.get(image_metadata=image_metadata, img_object=img_object, defect_list=list())
        cropped_img_list = cropper.crop(**cropping_properties)
        cropped_img = cropped_img_list[0]

        if is_preserved_entire_img_shape:
            original_shaped_zeros_img = np.zeros_like(img_object.base_img.img)
            area_cropped_img_abst_bbox = cropped_img.abst_bbox

            if is_entire_img_shape:
                original_shaped_zeros_img = cropped_img.img
            else:
                original_shaped_zeros_img[
                    area_cropped_img_abst_bbox.y_st:area_cropped_img_abst_bbox.y_ed,
                    area_cropped_img_abst_bbox.x_st:area_cropped_img_abst_bbox.x_ed
                ] = cropped_img.img

            area_img = EOPImage(original_shaped_zeros_img)
        else:
            area_img = cropped_img

        return area_img

