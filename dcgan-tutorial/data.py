import torch
from torch.utils.data import TensorDataset
from torch.nn.functional import interpolate
from torch.nn import Upsample
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_SIZE = 64  # Images have to get set to 64 x 64 for the architecture

batch_size = 64  # Batch size during training
image_size = 28  # Initial size of the images
num_images = 64 * 64  # Number of images to grab


def get_dataloader(name, num_images=None):
    # Get images
    data = np.load(f"./data/{name}.npy")

    # Section off images
    if num_images is not None:
        data = data[:num_images]

    # Data between 0 and 1
    data = torch.from_numpy(data) / 255
    # Data between -1 and 1
    data = (data - 0.5) / 0.5
    # Reshape image from 1D to 2D
    data = torch.reshape(data, (-1, 1, image_size, image_size))
    # Upscale data to 64 x 64
    data = interpolate(data, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode="bilinear")

    dataset = TensorDataset(data)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    # Name of category to get
    name = "mona_lisa"

    # Get an example tensor
    dataloader = get_dataloader(name)
    raw = next(iter(dataloader))[0][0]  # Get the first image in the batch
    print(f"{raw.shape = }")

    # Show the example tensor
    plt.axis("off")
    plt.title(name)
    plt.imshow(torch.reshape(raw, (OUTPUT_SIZE, OUTPUT_SIZE)))
    plt.show()
