# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:36:53 2021

@author: Jesse Hagenaars
Adapted by Guido for the AE4350 course
"""

# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

from typing import Optional, NamedTuple, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib notebook


def spike_function(v: torch.Tensor) -> torch.Tensor:
    return v.gt(0.0).float()  # gt means greater than, which works on tensors

# Placeholder for LIF state
class LIFState(NamedTuple):
    z: torch.Tensor
    v: torch.Tensor
    i: torch.Tensor


# Placeholder for LIF parameters
class LIFParameters(NamedTuple):
    i_decay: torch.Tensor = torch.as_tensor(0.0)
    v_decay: torch.Tensor = torch.as_tensor(0.75)
    thresh: torch.Tensor = torch.as_tensor(0.5)


# Actual LIF function
def lif_neuron(
    i: torch.Tensor,
    state: Optional[LIFState] = None,
    p: LIFParameters = LIFParameters(),
) -> Tuple[torch.Tensor, LIFState]:
    # Previous state
    if state is None:
        state = LIFState(
            z=torch.zeros_like(i),
            v=torch.zeros_like(i),
            i=torch.zeros_like(i),
        )
    # Update state
    i = state.i * p.i_decay + i
    v = state.v * p.v_decay * (1.0 - state.z) + i
    z = spike_function(v - p.thresh)
    return z, LIFState(z, v, i)


# Placeholder for ALIF state
# Change wrt LIF: threshold is also a state!
class ALIFState(NamedTuple):
    z: torch.Tensor
    v: torch.Tensor
    i: torch.Tensor
    t: torch.Tensor


# Placeholder for ALIF parameters
# Change wrt LIF: reset/start value for threshold, decay and addition constants
class ALIFParameters(NamedTuple):
    i_decay: torch.Tensor = torch.as_tensor(0.0)
    v_decay: torch.Tensor = torch.as_tensor(0.75)
    t_decay: torch.Tensor = torch.as_tensor(0.95)
    t_add: torch.Tensor = torch.as_tensor(1.05)
    t_0: torch.Tensor = torch.as_tensor(0.5)


# Actual ALIF function
# Change wrt LIF: additional equation for threshold adaptation
def alif_neuron(
    i: torch.Tensor,
    state: Optional[ALIFState] = None,
    p: ALIFParameters = ALIFParameters(),
) -> Tuple[torch.Tensor, ALIFState]:
    # Previous state
    if state is None:
        state = ALIFState(
            z=torch.zeros_like(i),
            v=torch.zeros_like(i),
            i=torch.zeros_like(i),
            t=torch.ones_like(i) * p.t_0,
        )
    # Update state
    i = state.i * p.i_decay + i
    v = state.v * p.v_decay * (1.0 - state.z) + i
    z = spike_function(v - state.t)
    t = state.t * p.t_decay + p.t_add * z
    return z, ALIFState(z, v, i, t)


# Incoming current: spikes of previous layer * weights
# Let's assume weights of 0.3 and one incoming connection
steps = 100
i = torch.randint(5, (steps,)).eq(2).float() * 0.3

currents = []
states = []
state = None
for step in range(steps):
    _, state = lif_neuron(i[step], state)
    currents.append(i[step].item())  # .item() to convert tensor to float (for 1-element tensor)
    states.append([state.z.item(), state.v.item(), state.i.item()])
    
plt.plot(currents, label="incoming current")
plt.plot(np.array(states)[:, 0], label="neuron spikes")
plt.plot(np.array(states)[:, 1], label="neuron voltage")
plt.plot(np.array(states)[:, 2], label="neuron current")
plt.grid()
plt.legend()
plt.show()


## Plot spike rate as a function of input strength
#steps = 1001  # input values
#duration = 5  # how long to stimulate
## By putting 'steps' in the batch dimension, we can can evaluate all steps in parallel
## It's as if we have 'steps' neurons with the same parameters!
#inp = torch.linspace(0, 10, steps)  # no .view(-1, 1) necessary
#
## Log responses in a tensor
#log_lif = torch.zeros(steps, duration)
#log_alif = torch.zeros(steps, duration)
#state_lif, state_alif = None, None
## Note that we can't parallelize the 'duration' dimension, because the next neuron state depends on the previous
#for d in range(duration):
#    # Only log spikes
#    log_lif[:, d], state_lif = lif_neuron(inp, state_lif)
#    log_alif[:, d], state_alif = alif_neuron(inp, state_alif)
#    
## Now average over 'duration' to get the avg spike rate
## And plot!
#plt.plot(inp.view(-1).numpy(), log_lif.mean(-1).numpy(), label="LIF")
#plt.plot(inp.view(-1).numpy(), log_alif.mean(-1).numpy(), label="ALIF")
#plt.plot(inp.view(-1).numpy(), inp.view(-1).numpy(), label="Input")
#plt.grid()
#plt.legend()
#plt.show()