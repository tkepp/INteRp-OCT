#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/NILOIDE/Implicit_segmentation/blob/master/model/mlp.py
"""

import torch
from torch import nn
from typing import Tuple, overload
from model.layer_activations import Layer, Sine
import numpy as np

class MLP(nn.Module):
    def __init__(self, coord_size: int, embed_size: int, layer_class: Layer, **kwargs):
        super(MLP, self).__init__()
        hidden_size = kwargs.get("hidden_size")
        input_coord_to_all_layers = kwargs.get("input_coord_to_all_layers")
        num_hidden_layers = kwargs.get("num_hidden_layers")

        hidden_input_size = hidden_size + (coord_size if input_coord_to_all_layers else 0)
        a = [layer_class(coord_size + embed_size, hidden_size, is_first=True, **kwargs)]
        for i in range(num_hidden_layers - 1):
            a.append(layer_class(hidden_input_size, hidden_size, **kwargs))
        self.hid_layers = nn.ModuleList(a)
        self.out_size = hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        x = torch.cat(x, dim=2)
        x = self.hid_layers[0](x)
        if self.hid_layers.__len__() > 1:
            for layer in self.hid_layers[1:]:
                x = layer(x)
        return x


class ResMLP(MLP):

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        x = torch.cat(x, dim=2)
        x = self.hid_layers[0](x)
        if self.hid_layers.__len__() > 1:
            prev_x = x
            for layer in self.hid_layers[1:]:
                x = layer(x) + prev_x
                prev_x = x
        return x


class MLPHiddenCoords(MLP):

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        coord, _ = x
        x = torch.cat(x, dim=2)
        x = self.hid_layers[0](x)
        if self.hid_layers.__len__() > 1:
            for layer in self.hid_layers[1:]:
                x = layer(torch.cat((coord, x), dim=2))
        return x


class ResMLPHiddenCoords(MLP):

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        coord, _ = x
        x = torch.cat(x, dim=2)
        x = self.hid_layers[0](x)
        if self.hid_layers.__len__() > 1:
            prev_x = x
            for layer in self.hid_layers[1:]:
                x = layer(torch.cat((coord, x), dim=2)) + prev_x
                prev_x = x
        return x


class SegmentationHead(nn.Module):
    def __init__(self, input_size, num_classes=4, **kwargs):
        super(SegmentationHead, self).__init__()
        self.seg_layer = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor):
        out = self.seg_layer(x)
        out = nn.functional.softmax(out, dim=2)
        return out

class SegmentationHeadSDM(nn.Module):
    def __init__(self, input_size, **kwargs):
        super(SegmentationHeadSDM, self).__init__()
        self.out_layer = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor):
        out = self.out_layer(x)
        out = torch.tanh(out)
        return out


class ReconstructionHead(nn.Module):
    def __init__(self, input_size, **kwargs):
        super(ReconstructionHead, self).__init__()
        self.out_layer = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor):
        out = self.out_layer(x)
        out = torch.sigmoid(out)
        return out


class ModulatedSiren(nn.Module):
    def __init__(self, coord_size: int, embed_size: int, **kwargs):
        super(ModulatedSiren, self).__init__()
        hidden_size = kwargs.get("hidden_size")
        input_coord_to_all_layers = kwargs.get("input_coord_to_all_layers")
        num_hidden_layers = kwargs.get("num_hidden_layers")

        self.omega_0 = kwargs.get("siren_omega_0", 30.0)

        net_layer_list = [nn.Linear(coord_size, hidden_size)]
        mod_layer_list = []
        for i in range(num_hidden_layers - 1):
            net_layer_list.append(nn.Linear(hidden_size, hidden_size))
            mod_layer_list.append(nn.Linear(embed_size, hidden_size * 2))
        self.hid_layers = nn.ModuleList(net_layer_list)
        self.mod_layers = nn.ModuleList(mod_layer_list)
        self.out_size = hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.reset_params()

    @staticmethod
    def weights_init(model, omega_0=30):
        with torch.no_grad():
            if isinstance(model, nn.Linear):
                hidden_ch = max(model.weight.shape[0], model.weight.shape[1])
                val = np.sqrt(6 / hidden_ch) / omega_0 if model.weight.shape[1] > 4 else 1 / model.weight.shape[1]
                torch.nn.init.uniform_(model.weight, -val, val)
                if model.bias is not None:
                    torch.nn.init.zeros_(model.bias)

    def reset_params(self):
        for i, module in enumerate(self.modules()):
            self.weights_init(module)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        coords, h = x
        out = torch.sin(self.omega_0 * self.hid_layers[0](coords))

        for i in range(1, len(self.hid_layers) - 1):
            phi_psi = self.mod_layers[i - 1](h)
            out = torch.sin(self.omega_0 * (phi_psi[:self.hidden_size] * self.hid_layers[i](out)) + phi_psi[self.hidden_size:])
        return out

class Siren(nn.Module):
    def __init__(self, coord_size: int, embed_size: int, **kwargs):
        super(Siren, self).__init__()
        hidden_size = kwargs.get("hidden_size")
        input_coord_to_all_layers = kwargs.get("input_coord_to_all_layers")
        num_hidden_layers = kwargs.get("num_hidden_layers")

        self.omega_0 = kwargs.get("siren_omega_0", 30.0)

        net_layer_list = [nn.Linear(coord_size+embed_size, hidden_size)]
        for i in range(num_hidden_layers - 1):
            net_layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.hid_layers = nn.ModuleList(net_layer_list)
        self.out_size = hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.reset_params()

    @staticmethod
    def weights_init(model, omega_0=30):
        with torch.no_grad():
            if isinstance(model, nn.Linear):
                hidden_ch = max(model.weight.shape[0], model.weight.shape[1])
                val = np.sqrt(6 / hidden_ch) / omega_0 if model.weight.shape[1] > 4 else 1 / model.weight.shape[1]
                torch.nn.init.uniform_(model.weight, -val, val)
                if model.bias is not None:
                    torch.nn.init.zeros_(model.bias)

    def reset_params(self):
        for i, module in enumerate(self.modules()):
            self.weights_init(module)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        x = torch.cat(x, dim=2)
        out = torch.sin(self.omega_0 * self.hid_layers[0](x))

        for i in range(1, len(self.hid_layers) - 1):
            out = torch.sin(self.omega_0 * self.hid_layers[i](out))
        return out