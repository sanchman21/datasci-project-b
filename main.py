from Puremodel_Trainer import train_model_pure
from Trainer import train_model
import utils_zhenzhuo


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

    config_path = '_configs\puremodel-monocyte_reassigned.yaml' # select the config file (yaml) to use
    config_path = utils_zhenzhuo.convert_path_to_os_specific(config_path) # convert the path to os specific
    config = utils_zhenzhuo.load_config(config_path) # load the config file
    train_model_pure(config) # train the model using the config file

    config_path = '_configs\puremodel-nutrophil.yaml' # select the config file (yaml) to use
    config_path = utils_zhenzhuo.convert_path_to_os_specific(config_path) # convert the path to os specific
    config = utils_zhenzhuo.load_config(config_path) # load the config file
    train_model_pure(config) # train the model using the config file
