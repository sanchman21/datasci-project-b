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

    # config_path = 'configs\configs_random_transfer\config_multimodal.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)

    # config_path = 'configs_random_transfer\config_resnet50.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)

    # config_path = 'monocyte_resnet50\monocyte_resnet50.yaml'
    # config = utils.load_config(config_path)
    # train_model(config)
    
    # config_path = 'resnet50_model\\neutrophil_resnet50.yaml'
    # config_path = utils.convert_path_to_os_specific(config_path)

    # config = utils.load_config(config_path)
    # train_model_pure(config)

    # config_path = '_configs\\neutrophil+monocyte+patientmeta.yaml'
    # config_path = utils.convert_path_to_os_specific(config_path)

    # config = utils.load_config(config_path)
    # train_model(config)

    config_path = '_configs\puremodel-monocyte_reassigned.yaml'
    config_path = utils.convert_path_to_os_specific(config_path)

    config = utils.load_config(config_path)
    train_model_pure(config)

    config_path = '_configs\puremodel-nutrophil.yaml'
    config_path = utils.convert_path_to_os_specific(config_path)

    config = utils.load_config(config_path)
    train_model_pure(config)
