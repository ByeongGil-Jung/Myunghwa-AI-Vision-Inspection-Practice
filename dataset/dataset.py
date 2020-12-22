from dataset.base import Dataset

from domain.data import EOPData
from domain.image import BoundingBox, Defect


class EOPDataset(Dataset):

    def __init__(self, data_dict: dict, transform=None):
        super(EOPDataset, self).__init__()
        self.serial_number_list = data_dict['serial_number_list']
        self.image_path_list = data_dict['image_path_list']
        self.is_NG_list = data_dict['is_NG_list']
        self.cam_list = data_dict['cam_list']
        self.defect_category_lists = data_dict['defect_category_list']
        self.x_st_lists = data_dict['x_st_list']
        self.x_ed_lists = data_dict['x_ed_list']
        self.y_st_lists = data_dict['y_st_list']
        self.y_ed_lists = data_dict['y_ed_list']
        self.ratio_lists = data_dict['ratio_list']
        self.transform = transform

    def __getitem__(self, idx):
        eop_data = EOPData(
            serial_number=self.serial_number_list[idx],
            image_path=self.image_path_list[idx],
            is_NG=self.is_NG_list[idx],
            cam=self.cam_list[idx],
            defect_list=None
        )

        defect_list = list()

        # 만약 결함이 있으면 defect 추가
        if eop_data.is_NG:
            for defect_category, x_st, x_ed, y_st, y_ed, ratio in \
                    zip(self.defect_category_lists[idx], self.x_st_lists[idx], self.x_ed_lists[idx],
                        self.y_st_lists[idx], self.y_ed_lists[idx], self.ratio_lists[idx]):
                defect_abst_bbox = BoundingBox(x_st=x_st, x_ed=x_ed, y_st=y_st, y_ed=y_ed)
                defect_abst_bbox.calculate_with_ratio(ratio)

                defect = Defect(
                    abst_bbox=defect_abst_bbox,
                    category=defect_category,
                    location_img=eop_data.img_object
                )
                defect_list.append(defect)

        eop_data.defect_list = defect_list

        return eop_data

    def __len__(self):
        return len(self.image_path_list)
