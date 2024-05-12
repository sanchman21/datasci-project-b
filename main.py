from TrainModel_multimodal import train_model
import utils


# config_path = 'configs\config_original.yaml'
# config = utils.load_config(config_path)
# train_model(config)

# config_path = 'configs\config_multimodal.yaml'
# config = utils.load_config(config_path)
# train_model(config)


config_path = 'configs_normalizaed\config_original.yaml'
config = utils.load_config(config_path)
train_model(config)

config_path = 'configs_normalizaed\config_multimodal.yaml'
config = utils.load_config(config_path)
train_model(config)