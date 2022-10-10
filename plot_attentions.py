import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches




def is_vertical(single_head, cutoff=0.9):
    """
    Check if the head is vertical
    """
    assert single_head.shape[0] == single_head.shape[1]
    count = 0
    for i in range(single_head.shape[0]):
        if single_head[i].argmax() in [0,single_head.shape[1]-1]:
            count += 1
    return count >= single_head.shape[0]*cutoff

    

def visualize_all(attn, n_layers=24, n_heads=16):
    vertical_heads_tensor = np.zeros((n_layers, n_heads))
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(48,72), sharex=True, sharey=True, constrained_layout=True)
    
    for i in range(n_layers):
        for j in range(n_heads):
            im = axes[i, j].imshow(attn[i, j], cmap='Blues')
            axes[i, j].axis('off')
            left = plt.xlim()[0]
            right = plt.xlim()[1]
            if is_vertical(attn[i, j]):
                vertical_heads_tensor[i, j] = 1
                axes[i, j].add_patch(
                    patches.Rectangle((left,left), (right-left-0.05),  (right-left+0.05), fill=True, fc=(0.66,0.33,0,0.2) ,edgecolor=(0.66,0.33,0,1), linewidth=0)
                )

    plt.savefig("bert_attention_pretrained.pdf")




if __name__ == "__main__":
    PATH_TO_ATT = None   # need to use your own path here
    if PATH_TO_ATT is not None:
        attention_scores = np.load(PATH_TO_ATT)
        visualize_all(attention_scores)





