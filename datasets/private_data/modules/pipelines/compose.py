import torch

from datasets.private_data.utils.functional import initialize_object


class Compose(torch.nn.Module):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        super().__init__()
        assert isinstance(transforms, dict), "transforms must be a dict."
        self.transforms = []
        for key, transform in transforms.items():
            transform = initialize_object(transform)
            self.transforms.append(transform)

    def forward(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
