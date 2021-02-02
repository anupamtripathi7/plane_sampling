import torch
import os
import torch.nn as nn
from . import utils
from .camera import PerspectiveCameras
from .msi import MultiPlaneImage


class BaseModel(nn.Module):
    """A base class for all models.
    Contains methods:
        save: save a model
        load: load a model
        init_optimizer: initialize Adam optimizer
    """
    def __init__(self, name="test0", description=""):
        super().__init__()

        self.name = name
        self.description = description
        self.epochs = 0
        self.optimizer = None
        self.best_loss = 1e10

    def save(self, filepath):
        path, filename = filepath.rsplit("/", 1)

        if not os.path.exists(path):
            os.makedirs(path)

        model_dict = {
            "class": self.__class__.__name__,
            "name": self.name,
            "description": self.description,
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epochs": self.epochs,
            "cfg": self.cfg.__dict__
        }

        torch.save(model_dict, filepath)

    def load(self, model_dict, device="cpu", verbose=True):
        self.load_state_dict(model_dict["model"])
        self.to(device)

        if "optimizer" in model_dict.keys():
            self.init_optimizer()
            self.optimizer.load_state_dict(model_dict["optimizer"])
        self.epochs = model_dict["epochs"]

    def init_optimizer(self, lr=1e-4, betas=(0.9, 0.999)):
        parameters = [params for params in self.parameters() if params.requires_grad]
        self.optimizer = torch.optim.Adam(parameters, lr=lr, betas=betas)


class IterMPI(BaseModel):
    def __init__(self, cfg):
        super().__init__()

        M, C, H, W = cfg.mpi_shape
        self.features = nn.Parameter(torch.zeros((1, M, C, H, W)))
        if hasattr(cfg, 'disparity_planes'):
            self.disparity_planes = cfg.disparity_planes
        else:
            self.disparity_planes = utils.uniform_disparity_planes(cfg.depth_range, M).to(cfg.device)
        self.camera = PerspectiveCameras(cfg.intrinsic, cfg.extrinsic).to(cfg.device)                                   # order of parameters
        self.cfg = cfg
        self.compositor = cfg.compositor

    def forward(self, shift):
        camera = self.camera.translate_in_x(shift)
        features = (torch.tanh(self.features) + 1.0) / 2.0
        mpi = MultiPlaneImage(features, self.disparity_planes, self.camera)
        out = mpi.view(camera, self.cfg.compositor)

        return out
