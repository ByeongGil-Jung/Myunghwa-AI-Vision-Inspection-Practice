from logger import logger
from model.cnn_custom import CNNModelCustom01
from model.ae_custom import AutoencoderModelCustom01


class ModelFactory(object):

    def __init__(self):
        self.model = None

    @classmethod
    def create(cls, model_name):
        model_factory = cls()

        model = None

        if model_name == "defect_detector_custom_01":
            model = CNNModelCustom01(fc_input=3136)  # Default : 128 x 128
        elif model_name == "autoencoder_256":
            model = AutoencoderModelCustom01()

        # Set
        model_factory.model = model

        logger.info(f"Model selected : '{model_name}'")
        return model_factory
