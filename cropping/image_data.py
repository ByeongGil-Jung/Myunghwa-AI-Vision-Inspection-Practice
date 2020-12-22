import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image

from domain.image import Coordinate, BoundingBox, EOPImage, Defect, Contour


class ImportantPixelContourMapper(object):

    def __init__(self, important_pixel_coord_list, contour_list):
        self.important_pixel_coord_list = important_pixel_coord_list
        self.contour_list = contour_list

    def get_candidate_coord_list(self, distance_margin_min, distance_margin_max, contour_size_margin_min, contour_size_margin_max, contour_area_margin_min, contour_area_margin_max):
        candidate_coord_list = list()

        for important_pixel_coord in self.important_pixel_coord_list:
            important_pixel_coord: Coordinate
            is_candidate = False

            best_distance = None
            best_contour = None

            if "A" in important_pixel_coord.area.name:
                is_candidate = True
            else:

                # 가장 가까운 contour 찾기
                for contour in self.contour_list:
                    contour: Contour

                    distance = important_pixel_coord.get_distance(coord=contour.center)
                    
                    # Distance 예외
                    if not distance_margin_min <= distance < distance_margin_max:
                        continue

                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_contour = contour

                if best_contour:
                    # Contour margin 측정
                    rect_pt_1, rect_pt_2, rect_pt_3, rect_pt_4 = best_contour.box_list
                    rect_length_1 = Coordinate(rect_pt_1[1], rect_pt_1[0]).get_distance(
                        Coordinate(rect_pt_2[1], rect_pt_2[0]))
                    rect_length_2 = Coordinate(rect_pt_1[1], rect_pt_1[0]).get_distance(
                        Coordinate(rect_pt_4[1], rect_pt_4[0]))

                    # Ellipse 측정
                    # (x, y), (MA, ma), angle = best_contour.ellipse
                    # ellipse_area = math.pi * MA * ma
                    rect_area = rect_length_1 * rect_length_2

                    print(best_distance, rect_area, rect_length_1, rect_length_2)

                    if contour_area_margin_min <= rect_area < contour_area_margin_max \
                        and contour_size_margin_min <= rect_length_1 < contour_size_margin_max \
                        and contour_size_margin_min <= rect_length_2 < contour_size_margin_max:
                        is_candidate = True

            # Final
            if is_candidate:
                candidate_coord_list.append(important_pixel_coord)

        return candidate_coord_list


class Drawer(object):

    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    BLUE = (0, 0, 255)
    CYAN = (0, 255, 255)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    ORANGE = (255, 128, 0)
    RED = (255, 0, 0)
    MAGENTA = (255, 0, 255)
    BLACK = (0, 0, 0)

    def __init__(self, img, is_base_img=True):
        if isinstance(img, EOPImage):
            if is_base_img:
                self._set_base_img_with_EOPImage(img.base_img)
            else:
                self._set_base_img_with_EOPImage(img)
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                # img = img.astype(np.uint8)
                self.base_img = img
        elif isinstance(img, torch.Tensor):
            img = img.numpy()
            if len(img.shape) == 2:
                img = img.astype(np.uint8)
                self.base_img = img
            elif len(img.shape) == 3:
                img = img.squeeze(0)
                img = img.astype(np.uint8)
                self.base_img = img

    def draw_point(self, point_coord: Coordinate, point_color=RED, circle_color=BLUE, thickness=5):
        # Draw point
        self.base_img[point_coord.y, point_coord.x] = point_color
        # Draw circle around the point
        cv2.circle(self.base_img, center=point_coord.value, radius=20, color=circle_color, thickness=thickness)

    def draw_bounding_box(self, data, color=BLUE, thickness=5):
        bbox: BoundingBox = None

        if isinstance(data, EOPImage):
            bbox = data.abst_bbox
        elif isinstance(data, BoundingBox):
            bbox = data

        # left_up to right_up
        cv2.line(self.base_img, pt1=bbox.left_up.value, pt2=bbox.right_up.value, color=color, thickness=thickness)
        # right_up to right_down
        cv2.line(self.base_img, pt1=bbox.right_up.value, pt2=bbox.right_down.value, color=color, thickness=thickness)
        # right_down to left_down
        cv2.line(self.base_img, pt1=bbox.right_down.value, pt2=bbox.left_down.value, color=color, thickness=thickness)
        # left_down to left_up
        cv2.line(self.base_img, pt1=bbox.left_down.value, pt2=bbox.left_up.value, color=color, thickness=thickness)

    def draw_bounding_box_list(self, bbox_list: list, color=BLUE, thickness=2):
        for bbox in bbox_list:
            self.draw_bounding_box(bbox, color=color, thickness=thickness)

    def draw_defect(self, defect: Defect, color=RED, thickness=2):
        self.draw_bounding_box(defect.abst_bbox, color=color, thickness=thickness)

    def draw_defect_list(self, defect_list: list, color=RED, thickness=2):
        for defect in defect_list:
            self.draw_defect(defect=defect, color=color, thickness=thickness)

    def draw_text(self, text, location: Coordinate, fontFace, fontScale, text_color=RED, is_background=True, background_color=BLACK, thickness=None):
        x, y = location.value
        # cv2.rectangle(self.base_img, (x - 1, y - 15), (x + 15, y + 15), background_color, -1) # put background
        cv2.putText(self.base_img, text, (x, y), fontFace, fontScale, text_color)

    def show(self, figsize=(15, 7)):
        img = self.base_img

        if not isinstance(self.base_img, np.ndarray):
            img = cv2.cvtColor(self.base_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.show()

    def save(self, save_path, resize_size=None, format='png'):
        img = self.base_img

        if resize_size:
            img = cv2.resize(img, dsize=resize_size)

        if not isinstance(self.base_img, np.ndarray):
            img = cv2.cvtColor(self.base_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        plt.imsave(save_path, img, format=format, cmap='gray')

    def _set_base_img_with_EOPImage(self, base_img: EOPImage):
        base_img = np.copy(base_img.img)
        base_img = base_img.astype(np.uint8)
        self.base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)


class ContourFactory(object):

    def __init__(self, difference_img, cropper):
        self.difference_img = difference_img
        self.cropper = cropper
        self.debug_difference_img = self.difference_img.copy().astype(np.uint8)
        self.debug_difference_img_color = cv2.cvtColor(self.debug_difference_img, cv2.COLOR_GRAY2BGR)

        self.contour_list = list()

    def generate(self, contour_pixel_threshold, contour_area_threshold):
        ret, img_binary = cv2.threshold(self.debug_difference_img, contour_pixel_threshold, 255, 0)
        cv2_contour_list, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = list()

        for cv2_contour in cv2_contour_list:
            contour = Contour(
                contour=cv2_contour,
                contour_area_threshold=contour_area_threshold,
                is_approximated_polygon=False
            )
            contour.center.area = self.cropper.get_area_with_coord(coord=contour.center)

            contour_list.append(contour)

        # contour_list = [Contour(contour=cv2_contour,
        #                         contour_area_threshold=contour_area_threshold,
        #                         is_approximated_polygon=False)
        #                 for cv2_contour in cv2_contour_list]
        #
        # # Set area to center of contour
        # map(lambda contour: contour.center.area = self.cropper.get_area_with_coord(coord=contour.center), contour_list)

        # 면적이 k 이상
        valid_contour_list = [contour for contour in contour_list
                              if contour.is_in_difference_img(difference_img=self.difference_img)]

        self._draw_contour_list(contour_list=valid_contour_list)

        self.contour_list = valid_contour_list

        return valid_contour_list

    def _draw_contour_list(self, contour_list):
        for contour in contour_list:
            # cv2.ellipse(self.debug_difference_img_color, contour.ellipse, (0, 0, 255), 1)
            cv2.drawContours(self.debug_difference_img_color, [contour.contour], 0, (255, 0, 0))
            cv2.drawContours(self.debug_difference_img_color, [contour.box_list], 0, (0, 255, 0), 1)
            cv2.circle(self.debug_difference_img_color, (contour.center_x, contour.center_y), 1, (0, 0, 255), -1)
