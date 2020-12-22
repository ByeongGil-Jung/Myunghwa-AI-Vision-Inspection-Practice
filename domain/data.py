from domain.base import Domain
from domain.image import EOPImage


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
