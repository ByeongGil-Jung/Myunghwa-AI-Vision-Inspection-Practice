import cv2
import numpy as np

from domain.image import Coordinate, BoundingBox, EOPImage, Defect


class CropHelper(object):

    def __init__(self, img_object: EOPImage):
        self.img_object = img_object
        self.mask: np.ndarray = None
        self.cropped_img_object: EOPImage = None

    def relocate_defect(self, img_object: EOPImage, defect: Defect):
        # Defect 의 center 좌표가 이미지의 bbox 내에 속해있으면 해당 조각으로 넘김
        if defect.is_defect_in_bounding_box(bbox=img_object.abst_bbox):
            img_object.defect_list.append(defect)
            defect.location_img = img_object


class RingCropHelper(CropHelper):

    def __init__(self, img_object: EOPImage, big_rad=None, small_rad=None):
        super(RingCropHelper, self).__init__(img_object)
        self._big_rad = big_rad
        self._small_rad = small_rad

    @property
    def big_rad(self):
        return self._big_rad

    @big_rad.setter
    def big_rad(self, big_rad):
        if isinstance(big_rad, int):
            self._big_rad = big_rad
        else:
            raise ValueError("It must be int")

    @property
    def small_rad(self):
        return self._small_rad

    @small_rad.setter
    def small_rad(self, small_rad):
        if isinstance(small_rad, int):
            self._small_rad = small_rad
        else:
            raise ValueError("It must be int")

    def get_ring_mask(self, big_rad: int, small_rad: int):
        center_coord = Coordinate(big_rad, big_rad)
        frame_size = (big_rad * 2, big_rad * 2)

        mask_big = np.zeros(frame_size, dtype=np.uint8)
        mask_small = np.zeros(frame_size, dtype=np.uint8)

        cv2.circle(mask_big, center_coord.value, big_rad, (255, 255, 255), thickness=-1)
        cv2.circle(mask_small, center_coord.value, small_rad, (255, 255, 255), thickness=-1)

        return (mask_big - mask_small) / 255

    def crop_ring(self, center_coord: Coordinate):
        center_x = center_coord.x
        center_y = center_coord.y

        crop_abst_bbox = BoundingBox(
            x_st=center_x - self.big_rad,
            x_ed=center_x + self.big_rad,
            y_st=center_y - self.big_rad,
            y_ed=center_y + self.big_rad
        )

        if self.mask is None:
            self.mask = self.get_ring_mask(big_rad=self.big_rad, small_rad=self.small_rad)

        self.cropped_img_object = EOPImage(
            img=self.img_object.img[crop_abst_bbox.y_st: crop_abst_bbox.y_ed,
                crop_abst_bbox.x_st: crop_abst_bbox.x_ed] * self.mask,
            abst_bbox=crop_abst_bbox
        )

        # image object 에 child 로 저장
        self.img_object.add_child(self.cropped_img_object)

        return self.cropped_img_object

    def get_crop_rotate_ring_piece_list(self, angle):
        ring_img = np.copy(self.cropped_img_object.img)
        crop_rotate_ring_piece_list = list()

        abst_center_x = self.cropped_img_object.abst_bbox.center.x
        abst_center_y = self.cropped_img_object.abst_bbox.center.y

        # Crop 할 부분 계산
        """
        @TODO ??
        함수로 뺄 것
        """
        height, width = ring_img.shape
        rad = np.pi / 180

        crop_center_x = width // 2
        crop_center_y = height // 2
        gap_x = self.big_rad * np.sin(angle / 2 * rad)
        gap_y = self.big_rad - self.small_rad * np.cos(angle / 2 * rad)

        # Crop 된 이미지의 잘라낼 좌표
        crop_bbox = BoundingBox(
            x_st=int(crop_center_x - gap_x),
            x_ed=int(crop_center_x + gap_x),
            y_st=0,
            y_ed=int(gap_y)
        )

        # Crop 된 이미지의 절대 좌표
        rotate_crop_abst_bbox = BoundingBox(
            x_st=crop_bbox.x_st,
            x_ed=crop_bbox.x_ed,
            y_st=crop_bbox.y_st,
            y_ed=crop_bbox.y_ed
        )

        rotate_crop_abst_bbox.init_with_st_ed(
            x_st=rotate_crop_abst_bbox.x_st + self.cropped_img_object.abst_bbox.x_st,
            x_ed=rotate_crop_abst_bbox.x_ed + self.cropped_img_object.abst_bbox.x_st,
            y_st=rotate_crop_abst_bbox.y_st + self.cropped_img_object.abst_bbox.y_st,
            y_ed=rotate_crop_abst_bbox.y_ed + self.cropped_img_object.abst_bbox.y_st
        )

        # 회전하면서 Cropping
        for position in range(360 // angle):
            rotate_img = EOPImage(
                img=ring_img[crop_bbox.y_st: crop_bbox.y_ed, crop_bbox.x_st: crop_bbox.x_ed],
                abst_bbox=rotate_crop_abst_bbox
            )

            self.cropped_img_object.add_child(rotate_img)

            # 회전
            ring_img = cv2.warpAffine(
                ring_img,
                cv2.getRotationMatrix2D((crop_center_x, crop_center_y), angle, scale=1),
                (width, height)
            )

            rotate_crop_abst_bbox = rotate_crop_abst_bbox.get_rotate_bounding_box(
                center_coord=Coordinate(abst_center_x, abst_center_y),
                angle=angle
            )

        # 조각들의 image list 반환
        for ring_piece in self.cropped_img_object.child_list:
            crop_rotate_ring_piece_list.append(ring_piece)

        return crop_rotate_ring_piece_list


class CircleCropHelper(CropHelper):

    def __init__(self, img_object: EOPImage, center_coord: Coordinate, rad: int):
        super(CircleCropHelper, self).__init__(img_object)
        self.center_coord = center_coord
        self.rad = rad
        self.circle_abst_bbox = BoundingBox(
            x_st=center_coord.x - rad,
            x_ed=center_coord.x + rad,
            y_st=center_coord.y - rad,
            y_ed=center_coord.y + rad
        )

    def get_circle_mask(self, inverse=False):
        mask = np.zeros_like(self.img_object.img, dtype=np.uint8)
        cv2.circle(mask, self.center_coord.value, self.rad, (255, 255, 255), thickness=-1)

        # 원 내부만 masking
        if not inverse:
            mask = mask / 255
        # 원 외부까지 masking
        else:
            mask = 1 - mask / 255

        return mask

    def crop_circle(self, inverse=False):
        if self.mask is None:
            self.mask = self.get_circle_mask(inverse=inverse)
        cropped_img = np.copy(self.img_object.img)
        cropped_img = cropped_img * self.mask

        # 원 내부만 Cropping
        if not inverse:
            self.cropped_img_object = EOPImage(img=cropped_img[self.circle_abst_bbox.y_st:self.circle_abst_bbox.y_ed,
                                                   self.circle_abst_bbox.x_st:self.circle_abst_bbox.x_ed],
                                               abst_bbox=self.circle_abst_bbox)
        # 원 외부만 Cropping
        else:
            self.cropped_img_object = EOPImage(img=cropped_img, abst_bbox=self.circle_abst_bbox)

        self.img_object.add_child(self.cropped_img_object)


class RectangleCropHelper(CropHelper):

    def __init__(self, img_object: EOPImage):
        super(RectangleCropHelper, self).__init__(img_object)

    def crop_rectangle(self, bbox: BoundingBox):
        self.cropped_img_object = EOPImage(
            img=self.img_object.img[bbox.y_st:bbox.y_ed, bbox.x_st:bbox.x_ed],
            abst_bbox=bbox
        )

        self.img_object.add_child(self.cropped_img_object)
