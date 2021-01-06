import numpy as np

from domain.base import Domain
from domain.image import EOPImage
from properties import DATASET_PROPERTIES


class EOPData(Domain):

    def __init__(self, serial_number: str, image_path: str, is_NG: bool, cam: int, defect_list: list = None):
        super(EOPData, self).__init__()
        if defect_list is None:
            defect_list = list()

        self.serial_number = serial_number
        self.img_object = EOPImage(img=image_path)
        self.is_NG = is_NG
        self.cam = cam
        self.defect_list = defect_list

    def __repr__(self):
        return f"serial_number: {self.serial_number} \n" \
               f"img_object: {self.img_object} \n" \
               f"is_NG: {self.is_NG} \n" \
               f"cam: {self.cam} \n" \
               f"defect_list: {self.defect_list} \n"


class InputData(object):

    def __init__(self, img_object: EOPImage):
        self.img_object = img_object
        self.img = img_object.img
        self.is_NG = np.zeros(1, dtype=np.float32)
        self.defect_category = np.zeros(DATASET_PROPERTIES.CLASS_NUMBER, dtype=np.float32)

        self.convert_label_to_one_hot()

    def convert_label_to_one_hot(self):
        # defect_list 가 비어있음 -> 결함이 없음
        img_object = self.img_object
        if not img_object.defect_list:
            self.is_NG[0] = 0
            self.defect_category[0] = 1
        else:
            self.is_NG[0] = 1
            for defect in img_object.defect_list:
                self.defect_category[defect.category] = 1
