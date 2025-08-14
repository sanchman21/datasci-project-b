import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_mask_channels(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    mask = np.unpackbits(data['mask']).reshape(data['shape'])
    nucleus_mask = mask[:, :, 0]
    cytoplasm_mask = mask[:, :, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(nucleus_mask, cmap='gray')
    ax1.set_title('Nucleus Mask')
    ax1.axis('off')
    
    ax2.imshow(cytoplasm_mask, cmap='gray')
    ax2.set_title('Cytoplasm Mask')
    ax2.axis('off')
    
    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    save_path = os.path.join("../../data/extras", f'{base_name}_masks.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    npy_path = '../../data/labelbox2/images/cmml_1.npy'
    visualize_mask_channels(npy_path)