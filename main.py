from Puremodel_Trainer import train_model_pure
from Trainer import train_model
import utils


if __name__ == '__main__':

    # config_path = 'configs\config_original.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)

    # config_path = 'configs\config_multimodal.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)


    # config_path = 'configs_normalizaed\config_original.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)

    # config_path = 'configs_normalizaed\config_multimodal.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)

    # config_path = 'new_configs\config_multimodal_withNeutrophilImages.yaml'
    # config = utils.load_config(config_path)
    # train_model(config) 

    # config_path = 'new_configs\config_resnet50_withNeutrophilImages.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)

    # config_path = 'configs_random_transfer\config_multimodal_withNeutrophilImages.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)

    # config_path = 'configs_random_transfer\config_resnet50_withNeutrophilImages.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)

    # config_path = 'configs_random_transfer\config_multimodal.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)

    # config_path = 'configs_random_transfer\config_resnet50.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)

    # config_path = 'monocyte_resnet50\monocyte_resnet50.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)
    
    config_path = 'monocyte_resnet50\mneutrophils_resnet50.yaml'
    config = utils.load_config(config_path)
    train_model_pure(config)



    
