from domain.base import Domain


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
        self.image_metadata = image_metadata
        self.version = version
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
        return f"{self.image_metadata.product}_{self.image_metadata.cam_number}_{self.image_metadata.crop_part}_{version}"
