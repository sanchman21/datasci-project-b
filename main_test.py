from Puremodel_Trainer import train_model_pure
from Trainer import train_model
import utils
 

if __name__ == '__main__':
    config_path = 'resnet50_model\\neutrophil_resnet50.yaml'
    config = utils.load_config(config_path)
    train_model_pure(config)
