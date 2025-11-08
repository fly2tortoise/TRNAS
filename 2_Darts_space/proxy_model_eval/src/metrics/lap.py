import sys
import numpy as np
import torch
import torch.nn as nn
from src.utils.utilities import count_parameters


def cal_regular_factor(model, mu, sigma):
    """
    Compute a regularization factor to control model complexity.

    Args:
        model (nn.Module): The target neural network.
        mu (float): Mean reference value for parameter count.
        sigma (float): Variance scaling factor.

    Returns:
        torch.Tensor: Regularization coefficient.
    """
    model_params = torch.as_tensor(count_parameters(model))
    regular_factor = torch.exp(-(torch.pow((model_params - mu), 2) / sigma))
    return regular_factor


class SampleWiseActivationPatterns_noise(object):
    """
    Collects and processes activation patterns with added noise.
    Used to measure activation diversity (SNAP metric).
    """
    def __init__(self, device):
        self.snap = -1
        self.activations = None  # 用于收集激活值
        self.device = device

    @torch.no_grad()
    def collect_activations(self, activations):
        # 收集激活值并转换为二进制 0/1
        n_sample = activations.size()[0]
        n_neuron = activations.size()[1]

        if self.activations is None:
            self.activations = torch.zeros(n_sample, n_neuron).to(self.device)

        self.activations = torch.sign(activations)

    @torch.no_grad()
    def calSNAP(self, regular_factor):

        self.activations = self.activations.T
        self.snap = torch.unique(self.activations, dim=0).size(0)

        del self.activations
        self.activations = None
        torch.cuda.empty_cache()
        return self.snap * regular_factor

class SNAP:
    """
    from SWAP (Sample-Wise Activation Patterns) in ICLR 2024.
    The SNAP metric add noise regularization to SWAP.
    This class measures network expressivity by analyzing binary activation
    patterns across samples, with optional Gaussian perturbations for stability and smoothness.
    """
    def __init__(self, model=None, inputs=None, device='cuda', seed=0, regular=False, mu=None, sigma=None, sap_std=0.05):
        self.model = model
        self.interFeature = []
        self.seed = seed
        self.regular_factor = 1
        self.inputs = inputs
        self.device = device
        self.sap_std = sap_std
        # ==========================
        if regular and mu is not None and sigma is not None:
            self.regular_factor = cal_regular_factor(self.model, mu, sigma).item()
        self.reinit(self.model, self.seed)

    def reinit(self, model=None, seed=None):
        if model is not None:
            self.model = model
            self.register_hook(self.model)
            self.snap = SampleWiseActivationPatterns_noise(self.device)

        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def clear(self):
        self.snap = SampleWiseActivationPatterns_noise(self.device)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for n, m in model.named_modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        """
        Capture ReLU activations during forward pass and apply random smoothing (SNAP noise).
        The noise stabilizes activation diversity across layers by preventing
        saturation and promoting local smoothness.
        """
        if not isinstance(output, torch.Tensor):
            return
        x = output.detach()
        # Relative perturbation (layer/channel-invariant)
        noise = 1.0 + self.sap_std * torch.randn_like(x)  # ~ N(1, σ)
        x = x * noise
        x = torch.relu(x)
        self.interFeature.append(x)

    def forward(self):
        """Compute the SNAP score for the given model and inputs."""
        self.interFeature = []
        with torch.no_grad():
            self.model.forward(self.inputs.to(self.device))
            if len(self.interFeature) == 0:
                return
            activations = torch.cat([f.view(self.inputs.size(0), -1) for f in self.interFeature], 1)
            self.snap.collect_activations(activations)
            return self.snap.calSNAP(self.regular_factor)
