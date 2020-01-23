import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img_name = sys.argv[1]
    img = Image.open("Images/" + img_name).convert('RGB')
    img_flip_array = np.array(img)
    for i in range(len(img_flip_array)):
        img_flip_array[i] = 255 - img_flip_array[i]

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].title.set_text('Original Image')
    ax[1].imshow(img_flip_array)
    ax[1].axis('off')
    ax[1].title.set_text('Flipped Image')
    plt.show()
