# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:49:10 2021

@author: Jesse Hagenaars
Modified by Guido de Croon for AE4350
"""

from typing import Optional, NamedTuple, Tuple, Any, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib notebook


class SpikeFunction(torch.autograd.Function):
    """
    Spiking function with rectangular gradient.
    Source: https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full
    Implementation: https://github.com/combra-lab/pop-spiking-deep-rl/blob/main/popsan_drl/popsan_td3/popsan.py
    """

    @staticmethod
    def forward(ctx: Any, v: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(v)  # save voltage - thresh for backwards pass
        return v.gt(0.0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        v, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (v.abs() < 0.5).float()  # 0.5 is the width of the rectangle
        return grad_input * spike_pseudo_grad, None  # ensure a tuple is returned
    
# Placeholder for LIF state
class LIFState(NamedTuple):
    z: torch.Tensor
    v: torch.Tensor
    i: torch.Tensor

class LIF(nn.Module):
    """
    Leaky-integrate-and-fire neuron with learnable parameters.
    """

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        # Initialize all parameters randomly as U(0, 1)
        self.i_decay = nn.Parameter(torch.rand(size)) # random decays
        self.v_decay = nn.Parameter(torch.rand(size))
        self.thresh = nn.Parameter(torch.rand(size))
        self.spike = SpikeFunction.apply  # spike function

    def forward(
        self,
        synapse: nn.Module,
        z: torch.Tensor,
        state: Optional[LIFState] = None,
    ) -> Tuple[torch.Tensor, LIFState]:
        # Previous state
        if state is None:
            state = LIFState(
                z=torch.zeros_like(synapse(z)),
                v=torch.zeros_like(synapse(z)),
                i=torch.zeros_like(synapse(z)),
            )
        # Update state
        i = state.i * self.i_decay + synapse(z)
        v = state.v * self.v_decay * (1.0 - state.z) + i
        z = self.spike(v - self.thresh)
        return z, LIFState(z, v, i)
    
class SpikingMLP(nn.Module):
    """
    Spiking network with LIF neuron model.
    """

    def __init__(self, sizes: Sequence[int]):
        super().__init__()
        self.sizes = sizes
        self.spike = SpikeFunction.apply

        # Define layers
        self.synapses = nn.ModuleList()
        self.neurons = nn.ModuleList()
        self.states = []
        # Loop over current (accessible with 'size') and next (accessible with 'sizes[i]') element
        for i, size in enumerate(sizes[:-1], start=1):
            # Parameters of synapses and neurons are randomly initialized
            self.synapses.append(nn.Linear(size, sizes[i], bias=False))
            self.neurons.append(LIF(sizes[i]))
            self.states.append(None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for i, (neuron, synapse) in enumerate(zip(self.neurons, self.synapses)):
            z, self.states[i] = neuron(synapse, z, self.states[i])
        return z

    def reset(self):
        """
        Resetting states when you're done is very important!
        """
        for i, _ in enumerate(self.states):
            self.states[i] = None
            
#sizes = [4, 10, 5]
#snn = SpikingMLP(sizes)

# Data and labels
samples = 10000
x = torch.randint(2, (samples, 2)).float()
y = (x.sum(-1) == 1).float()
print(x[:10])
print(y[:10])
print(f"Class imbalance: {y.sum() / y.shape[0]}")

# Network
sizes = [2,10,1]#[2, 5, 1]
snn = SpikingMLP(sizes)

# Loss function and optimizer
def most_basic_loss_ever(x, y):
    return (x.view(-1) - y.view(-1)).abs().sum()  # ensure both are flat

criterion = most_basic_loss_ever
optimizer = optim.Adam(snn.parameters())

# Batch size of 100, 2 epochs
batch = 100
epochs = 5
losses = []
for e in range(epochs):
    for i in range(samples // batch):

        # Zero the parameter gradients
        for p in snn.parameters():
            p.grad = None
            
        # Reset the network
        snn.reset()

        # Forward + backward + optimize
        y_hat = snn(x[i:i + batch])
        loss = criterion(y_hat, y[i:i + batch])
        loss.backward()  # sum loss over batch
        optimizer.step()

        # Print statistics
        losses.append(loss.item())
        # print(f"[{e + 1}, {i}] loss: {loss.item()}")

print('Finished Training')

plt.figure()
plt.plot(losses)
plt.show()

# Generate test samples:
samples = 1000
x_test = torch.randint(2, (samples, 2)).float()
y_test = (x_test.sum(-1) == 1).float()

snn.reset()
y_hat = snn(x_test)

n_errors = torch.sum(torch.abs(y_test.view(-1) - y_hat.view(-1)))
print(f'Percentage of errors on test set: {(n_errors / samples) * 100.0}')


