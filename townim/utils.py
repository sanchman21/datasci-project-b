import torch, numpy as np, random, os

# set seed
def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    Also, need to add "worker_init_fn=np.random.seed(seed)" in dataloader
    # https://discuss.pytorch.org/t/determinism-in-pytorch-across-multiple-files/156269
    # https://stackoverflow.com/questions/65685060/unique-seed-acrossing-multiple-imported-files-with-random-module-python
    """
    print(f"Setting seeds: {seed} ...... ")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=  True
    
def worker_init_fn(worker_id):     
    '''
    this function is for dataloader's worker_init_fn
    '''                                                     
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
# Define a custom dataset class for loading and preprocessing data
def make_weights_for_balanced_classes(labels, device):
    count = torch.bincount(torch.tensor(labels)).to(device)
    print('Count:', count.cpu().detach().numpy())
    
    weight = 1. / count.cpu().detach().numpy()
    print('Data sampling weight:', weight)
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)

    return samples_weight

def convert_path_to_os_specific(path: str) -> str:
    normalized_path = os.path.normpath(path)
    if os.sep == '/':
        return normalized_path.replace('\\', os.sep)
    else:
        return normalized_path.replace('/', os.sep)

def denormalize(image, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(image.device)
    return image * std + mean
