#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/NILOIDE/Implicit_segmentation/blob/master/model/activation_functions.py
"""

import abc
import torch
from torch import nn
import numpy as np


class Layer(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, **kwargs):
        super(Layer, self).__init__()
        self.dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.in_size = in_size
        self.out_size = out_size
        self.is_first = kwargs.get("is_first", False)

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Relu(Layer):
    def __init__(self, in_size, out_size, **kwargs):
        super(Relu, self).__init__(in_size, out_size, **kwargs)
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Sine(Layer):
    """ See SIREN paper and github. """
    def __init__(self, in_size, out_size, **kwargs):
        super(Sine, self).__init__(in_size, out_size, **kwargs)
        self.omega_0 = kwargs.get("siren_omega_0", 30.0)
        self.linear = nn.Linear(in_size, out_size)
        self.weight_init()
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30

    def forward(self, x):
        x = self.linear(x)
        x = torch.sin(self.omega_0 * x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def weight_init(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_size, 1 / self.in_size)
            else:
                # See supplement Sec. 1.5 for discussion of factor 30
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_size) / self.omega_0, np.sqrt(6 / self.in_size) / self.omega_0)


class ComplexWIRE(Layer):
    '''
        See WIRE paper and github.
        Implicit representation with complex Gabor nonlinearity

        Inputs;
            in_size: Input features
            out_size; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''

    def __init__(self, in_size, out_size, omega0=10.0, sigma0=40.0, trainable=False, **kwargs):
        super().__init__(in_size, out_size, **kwargs)
        self.omega_0 = omega0
        self.scale_0 = sigma0
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)
        self.linear = nn.Linear(in_size, out_size, bias=True, dtype=torch.cfloat)

    def forward(self, x):
        lin = self.linear(x)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        out = torch.exp(1j * omega - scale.abs().square())
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class WIRE(Layer):
    '''
        Implicit representation with Gabor nonlinearity

        Inputs;
            in_size: Input features
            out_size; Output features
            bias: if True, enable bias for the linear operation
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''

    def __init__(self, in_size, out_size, bias=True, **kwargs):
        super().__init__(in_size, out_size, **kwargs)
        self.omega_0 = kwargs.get("wire_omega_0", 30.0)  # Freq
        self.scale_0 = kwargs.get("wire_scale_0", 10.0)
        self.freqs = nn.Linear(in_size, out_size, bias=bias)
        self.scale = nn.Linear(in_size, out_size, bias=bias)
        self.weight_init()
    def forward(self, x):
        omega = self.omega_0 * self.freqs(x)
        scale = self.scale(x) * self.scale_0
        x = torch.cos(omega) * torch.exp(-(scale * scale))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


    def weight_init(self):
        with torch.no_grad():
            if self.is_first:
                self.freqs.weight.uniform_(-1 / self.in_size, 1 / self.in_size)
                self.scale.weight.uniform_(-1 / self.in_size, 1 / self.in_size)
            else:
                # See supplement Sec. 1.5 for discussion of factor 30
                self.freqs.weight.uniform_(-np.sqrt(6 / self.in_size) / self.omega_0, np.sqrt(6 / self.in_size) / self.omega_0)
                self.scale.weight.uniform_(-np.sqrt(6 / self.in_size) / self.omega_0, np.sqrt(6 / self.in_size) / self.omega_0)


class FINER(Layer):
    """ See FINER paper and https://github.com/liuzhen0212/FINER/blob/main/models.py. """
    def __init__(self, in_size, out_size, **kwargs):
        super(FINER, self).__init__(in_size, out_size, **kwargs)
        self.omega_0 = kwargs.get("finer_omega_0", 30.0)
        self.linear = nn.Linear(in_size, out_size, bias=True)
        self.init_weights()
        self.scale_req_grad = kwargs.get("scale_req_grad", False)
        self.first_bias_scale = kwargs.get("finer_scale_0", None)
        if self.first_bias_scale != None:
            self.init_first_bias()

    def forward(self, x):
        x = self.linear(x)
        scale = self.generate_scale(x)
        x = torch.sin(self.omega_0 * scale * x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_size,
                                            1 / self.in_size)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_size) / self.omega_0,
                                            np.sqrt(6 / self.in_size) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
                # print('init fbs', self.first_bias_scale)

    def generate_scale(self, x):
        if self.scale_req_grad:
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale