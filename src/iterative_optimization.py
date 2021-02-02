from src.model import IterMPI
from src.utils import Config
import torch
import numpy as np
from pytorch_msssim import ssim
import torch.nn as nn
from src.components import AlphaCompositing
import matplotlib.pyplot as plt
from planes_sampling.utils import load_single_image_and_intrinsics, plot_planes, image_to_tensor, get_extrinsic_after_shift


root = 'data'
scene = 0
frame = 0
epochs = 10000
lr = 0.003
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
M = 6


def fit(model, image):
    image = image_to_tensor(image)
    for epoch in range(epochs):
        optimizer.zero_grad()
        extrinsics = get_extrinsic_after_shift([-25.0, 25.0], [0, 0])
        projection = model(extrinsics)
        mse = loss(image.unsqueeze(0).float(), projection)
        epoch_loss = (1 - ssim(image.unsqueeze(0).float(), projection)) + mse

        print(epoch_loss, mse)
        epoch_loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            plt.imshow(projection[0, 0].permute(1, 2, 0).detach())
            plt.show()
            # plt.imshow(projection[0, 1].permute(1, 2, 0).detach())
            # plt.show()
            plot_planes(model.features.detach(), M, alpha=True)

        features = model.features.detach().numpy()
        np.save('mpi.npy', features)


if __name__ == "__main__":
    config = Config()

    intrinsics, images = load_single_image_and_intrinsics(scene, frame)
    _, H, W, C = images.shape
    print('Images Loaded. shape: {}'.format(images.shape))

    config.mpi_shape = (M, C+1, H, W)
    config.depth_range = np.array([1, 10000])
    config.device = device
    config.extrinsic = None
    config.intrinsic = torch.tensor(intrinsics[-2:]).unsqueeze(0).float()
    print(config.intrinsic.shape)
    config.compositor = AlphaCompositing()

    iter_model = IterMPI(config)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(iter_model.parameters(), lr=lr)

    fit(iter_model, images)
