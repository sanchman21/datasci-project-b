from TrainModel_resnet50 import train_model
import TrainModel_resnet50

config_path = 'config.yaml'
config = TrainModel_resnet50.load_config(config_path)
train_model(config)