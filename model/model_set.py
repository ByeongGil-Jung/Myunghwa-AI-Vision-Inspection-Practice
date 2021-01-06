from domain.metadata import ModelMetadata, ImageMetadata


class ModelSet(object):

    def __init__(self, name: str, models: dict):
        self.name = name
        self.models = models

    def __repr__(self):
        return f"{self.name}"

    def get_model_name_list(self):
        return [model_name for model_name in self.models.keys()]

    @classmethod
    def housing_cam1_autoencoder_model_set_1(cls):
        return cls(
            name="housing_cam1_autoencoder_model_set_1",
            models={
                "grid_1": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_1"), version="autoencoder_256"),
                "grid_2": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_2"), version="autoencoder_256"),
                "grid_3": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_3"), version="autoencoder_256"),

                "grid_4": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_4"), version="autoencoder_256"),
                "grid_5": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_5"), version="autoencoder_256"),
                "grid_6": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_6"), version="autoencoder_256"),

                "grid_7": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_7"), version="autoencoder_256"),
                "grid_8": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_8"), version="autoencoder_256"),
                "grid_9": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_9"), version="autoencoder_256"),

                "grid_10": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_10"), version="autoencoder_256"),
                "grid_11": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_11"), version="autoencoder_256"),

                "grid_12": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_12"), version="autoencoder_256"),
                "grid_13": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_13"), version="autoencoder_256"),

                "grid_14": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_14"), version="autoencoder_256"),
                "grid_15": ModelMetadata(ImageMetadata(product="housing", cam_number=1, crop_part="grid_15"), version="autoencoder_256")
            }
        )

    @classmethod
    def housing_cam2_autoencoder_model_set_1(cls):
        return cls(
            name="housing_cam2_autoencoder_model_set_1",
            models={
                "grid_1": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_1"), version="autoencoder_256"),
                "grid_2": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_2"), version="autoencoder_256"),
                "grid_3": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_3"), version="autoencoder_256"),
                "grid_4": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_4"), version="autoencoder_256"),

                "grid_5": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_5"), version="autoencoder_256"),
                "grid_6": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_6"), version="autoencoder_256"),
                "grid_7": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_7"), version="autoencoder_256"),
                "grid_8": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_8"), version="autoencoder_256"),

                "grid_9": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_9"), version="autoencoder_256"),
                "grid_10": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_10"), version="autoencoder_256"),
                "grid_11": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_11"), version="autoencoder_256"),
                "grid_12": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_12"), version="autoencoder_256"),
                "grid_13": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_13"), version="autoencoder_256"),
                "grid_14": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_14"), version="autoencoder_256"),
                "grid_15": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_15"), version="autoencoder_256"),
                "grid_16": ModelMetadata(ImageMetadata(product="housing", cam_number=2, crop_part="grid_16"), version="autoencoder_256")
            }
        )

    @classmethod
    def housing_cam3_autoencoder_model_set_1(cls):
        return cls(
            name="housing_cam3_autoencoder_model_set_1",
            models={
                "grid_1": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_1"), version="autoencoder_256"),
                "grid_2": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_2"), version="autoencoder_256"),
                "grid_3": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_3"), version="autoencoder_256"),
                "grid_4": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_4"), version="autoencoder_256"),

                "grid_5": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_5"), version="autoencoder_256"),
                "grid_6": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_6"), version="autoencoder_256"),
                "grid_7": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_7"), version="autoencoder_256"),
                "grid_8": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_8"), version="autoencoder_256"),
                "grid_9": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_9"), version="autoencoder_256"),

                "grid_10": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_10"), version="autoencoder_256"),
                "grid_11": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_11"), version="autoencoder_256"),
                "grid_12": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_12"), version="autoencoder_256"),
                "grid_13": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_13"), version="autoencoder_256"),
                "grid_14": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_14"), version="autoencoder_256"),

                "grid_15": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_15"), version="autoencoder_256"),
                "grid_16": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_16"), version="autoencoder_256"),
                "grid_17": ModelMetadata(ImageMetadata(product="housing", cam_number=3, crop_part="grid_17"), version="autoencoder_256"),
            }
        )

    @classmethod
    def housing_cam4_autoencoder_model_set_1(cls):
        return cls(
            name="housing_cam4_autoencoder_model_set_1",
            models={
                "grid_1": ModelMetadata(ImageMetadata(product="housing", cam_number=4, crop_part="grid_1"), version="autoencoder_256")
            }
        )

    @classmethod
    def housing_cam1_defect_detector_model_set_1(cls):
        return cls(
            name="housing_cam1_defect_detector_model_set_1",
            models={
                "defect_detector_cam_1_resnet_18": ModelMetadata(ImageMetadata(product="defect_detector", cam_number=1, crop_part="defect"), version="resnet_18"),
                "defect_detector_cam_1_resnet_50": ModelMetadata(ImageMetadata(product="defect_detector", cam_number=1, crop_part="defect"), version="resnet_50"),
                "defect_detector_cam_1_custom_01": ModelMetadata(ImageMetadata(product="defect_detector", cam_number=1, crop_part="defect"), version="custom_01"),
            }
        )

    @classmethod
    def housing_cam2_defect_detector_model_set_1(cls):
        return cls(
            name="housing_cam2_defect_detector_model_set_1",
            models={
                "defect_detector_cam_2_resnet_18": ModelMetadata(ImageMetadata(product="defect_detector", cam_number=2, crop_part="defect"), version="resnet_18"),
                "defect_detector_cam_2_resnet_50": ModelMetadata(ImageMetadata(product="defect_detector", cam_number=2, crop_part="defect"), version="resnet_50"),
                "defect_detector_cam_2_custom_01": ModelMetadata(ImageMetadata(product="defect_detector", cam_number=2, crop_part="defect"), version="custom_01"),
           }
        )

    @classmethod
    def housing_cam3_defect_detector_model_set_1(cls):
        return cls(
            name="housing_cam3_defect_detector_model_set_1",
            models={
                "defect_detector_cam_3_resnet_18": ModelMetadata(ImageMetadata(product="defect_detector", cam_number=3, crop_part="defect"), version="resnet_18"),
                "defect_detector_cam_3_resnet_50": ModelMetadata(ImageMetadata(product="defect_detector", cam_number=3, crop_part="defect"), version="resnet_50"),
                "defect_detector_cam_3_custom_01": ModelMetadata(ImageMetadata(product="defect_detector", cam_number=3, crop_part="defect"), version="custom_01")
            }
        )
