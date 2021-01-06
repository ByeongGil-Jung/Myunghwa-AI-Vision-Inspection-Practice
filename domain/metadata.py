import os

from domain.base import Domain
from properties import MODEL_PROPERTIES


class Metadata(Domain):

    def __init__(self):
        super(Metadata, self).__init__()


class ImageMetadata(Metadata):

    def __init__(self, product: str, cam_number: int, crop_part: str):
        super(ImageMetadata, self).__init__()
        self.product = product
        self.cam_number = cam_number
        self.crop_part = crop_part


class ModelMetadata(Metadata):

    def __init__(self, image_metadata: ImageMetadata, version: str, ensemble_version_list: list=None):
        super(ModelMetadata, self).__init__()
        self.version = version
        self.image_metadata = image_metadata
        self.model_name = f"{self.image_metadata.product}_{self.image_metadata.cam_number}_{self.image_metadata.crop_part}_{self.version}"

        self.model_file_metadata = ModelFileMetadata(model_name=self.model_name)

        self.ensemble_version_list = ensemble_version_list

        assert len(self.version.split("_")) == 2, "Plz put version with format of 'a_b'"
        self.model_base, self.model_info = self.version.split("_")

        # if not self.ensemble_version_list:
        #     self.model: EOPModelBase = EOPModelBase.create(
        #         product=product,
        #         cam_number=cam_number,
        #         crop_part=crop_part,
        #         version=version
        #     )

    def __repr__(self):
        return self.model_name


class ModelFileMetadata(Metadata):

    def __init__(self, model_name, model_ext=".pt", record_ext=".pkl", plot_ext=".png"):
        super(ModelFileMetadata, self).__init__()
        self.model_name = model_name
        self.model_dir_path = os.path.join(MODEL_PROPERTIES.RESULT_DIRECTORY_PATH, self.model_name)
        self.model_checkpoint_dir_path = os.path.join(self.model_dir_path, "checkpoints")

        self.model_ext = model_ext
        self.record_ext = record_ext
        self.plot_ext = plot_ext
        self.best_model_file_name = f"{self.model_name}_best_model{self.model_ext}"
        self.entire_record_file_name = f"{self.model_name}_entire_record{self.record_ext}"
        self.plot_file_name = f"{self.model_name}_plot{self.plot_ext}"

        self.normalization_variables_file_name = "normalization_variables.json"

    def get_normalization_variables_file_path(self):
        return os.path.join(self.model_dir_path, self.normalization_variables_file_name)

    def get_record_checkpoint_file_path_list(self):
        record_file_path_list = list()

        if os.path.isdir(self.model_checkpoint_dir_path):
            for file_name in os.listdir(self.model_checkpoint_dir_path):
                file_basename, file_ext = os.path.splitext(file_name)

                if file_ext == self.record_ext and file_basename.split("_")[-1].isdigit():
                    record_file_path_list.append(os.path.join(self.model_checkpoint_dir_path, file_name))

        return record_file_path_list

    def get_model_checkpoint_file_path_list(self):
        model_file_path_list = list()

        if os.path.isdir(self.model_checkpoint_dir_path):
            for file_name in os.listdir(self.model_checkpoint_dir_path):
                file_basename, file_ext = os.path.splitext(file_name)

                if file_ext == self.model_ext and file_basename.split("_")[-1].isdigit():
                    model_file_path_list.append(os.path.join(self.model_checkpoint_dir_path, file_name))

        return model_file_path_list

    def create_model_checkpoint_file_name(self, epoch):
        return f"{self.model_name}_epoch_{epoch}{self.model_ext}"

    def create_record_checkpoint_file_name(self, epoch):
        return f"{self.model_name}_record_epoch_{epoch}{self.record_ext}"

    def get_save_model_checkpoint_file_path(self, epoch):
        return os.path.join(self.model_checkpoint_dir_path, self.create_model_checkpoint_file_name(epoch=epoch))

    def get_save_record_checkpoint_file_path(self, epoch):
        return os.path.join(self.model_checkpoint_dir_path, self.create_record_checkpoint_file_name(epoch=epoch))

    def get_best_model_file_path(self):
        return os.path.join(self.model_dir_path, self.best_model_file_name)

    def get_entire_record_file_path(self):
        return os.path.join(self.model_dir_path, self.entire_record_file_name)

    def get_plot_file_path(self):
        return os.path.join(self.model_dir_path, self.plot_file_name)
