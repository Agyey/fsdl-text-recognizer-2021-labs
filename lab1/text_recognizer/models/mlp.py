from typing import Any, Dict
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FC1_DIM = 1024
FC2_DIM = 128


class MLP(nn.Module):
    """Simple MLP suitable for recognizing single characters."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dim = np.prod(data_config["input_dims"])
        num_classes = len(data_config["mapping"])

        self.dropout = nn.Dropout(0.5)
        layers = self.args.get("layers", FC1_DIM)
        self.layers = layers
        if layers:
            fcn = (int(FC1_DIM - x * ((FC1_DIM - FC2_DIM)//(layers-1))) for x in range(layers))
            fcl = input_dim
            fcv = []
            for fci in fcn:
                fcv.append(nn.Linear(fcl, fci))
                fcl = fci
            fcv.append(nn.Linear(fcl, num_classes))
            self.fcv = nn.Sequential(*fcv)
        else:
            fc1_dim = self.args.get("fc1", FC1_DIM)
            fc2_dim = self.args.get("fc2", FC2_DIM)

            self.fc1 = nn.Linear(input_dim, fc1_dim)
            self.fc2 = nn.Linear(fc1_dim, fc2_dim)
            self.fc3 = nn.Linear(fc2_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        if self.layers:
            for fci in self.fcv[:-1]:
                x = fci(x)
                x = F.relu(x)
                x = self.dropout(x)
            x = self.fcv[-1](x)
        else:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--layers", type=int, default=None, choices=range(2, 20))
        parser.add_argument("--fc1", type=int, default=1024)
        parser.add_argument("--fc2", type=int, default=128)
        return parser
