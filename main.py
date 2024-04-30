from TrainModel_resnet50 import train_model
import utils

config_path = 'config.yaml'
config = utils.load_config(config_path)
train_model(config)