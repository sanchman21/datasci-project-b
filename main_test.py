from Trainer import train_model
import utils

if __name__ == '__main__':

    config_path = 'test_config.yaml'
    config = utils.load_config(config_path)
    train_model(config)
