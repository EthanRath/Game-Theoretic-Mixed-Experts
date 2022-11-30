'''
Snn model library.

Class SNN and SNN2 are same model. But SNN2 forward function returns dict which contains all statevariables.
SNN returns state variables directly.

'''

import torch
import numpy as np
from Defenses.SNN.snn_functions import *

class first_order_low_pass_cell(torch.nn.Module):
    '''
    implement first order low pass filter
    '''
    def __init__(self, input_shape, step_num, batch_size, tau, train_coefficients):
        '''
        :param input_shape: tuple (width hight) or (width, hight, depth) this should be the same as input shape,
        always on to one connection
        :param step_num:
        :param batch_size:
        :param tau:

        '''
        super().__init__()
        self.input_shape = input_shape
        self.step_num = step_num
        self.batch_size = batch_size

        self.tau = torch.full(input_shape, tau)

        self.alpha_1 = torch.nn.Parameter(torch.exp(-1 / self.tau))
        self.alpha_1.requires_grad = train_coefficients

    def forward(self, current_spike, prev_states):
        """
        :param current_spike: [batch, dim0 ,dim1..]
        :param  prev_states: prev_state
        :return:
        """
        prev_t_1 = prev_states

        current_psp = self.alpha_1 * prev_t_1 + current_spike

        psp = current_psp
        new_states = psp

        return psp, new_states

class first_order_low_pass_layer(torch.nn.Module):
    def __init__(self, input_shape, step_num, batch_size, tau, train_coefficients, return_state=True):
        '''
        :param input_shape: tuple (width hight) or (width, hight, depth) this should be the same as input shape,
        always on to one connection
        :param step_num:
        :param batch_size:
        '''
        super().__init__()
        self.input_shape = input_shape
        self.step_num = step_num
        self.batch_size = batch_size
        self.tau = tau
        self.return_state = return_state

        self.first_order_low_pass_cell = first_order_low_pass_cell(input_shape, step_num, batch_size, tau,
                                                    train_coefficients)

    def forward(self, input_spikes, states=None):
        """
        :param current_spike: [batch, dim0 ,dim1..]
        :param  states: prev_psp
        :return:
        """

        if states is None:
            states = self.create_init_states()

        # unbind along last dimension
        inputs = list(input_spikes.unbind(dim=-1))
        spikes = []
        length = len(inputs)

        for i in range(length):
            spike, states = self.first_order_low_pass_cell(inputs[i], states)
            spikes += [spike]
        
        if self.return_state:
            return torch.stack(spikes, dim=-1), states
        else:
            return torch.stack(spikes, dim=-1)

    def create_init_states(self):
        device = self.first_order_low_pass_cell.alpha_1.device
        prev_t_1 = torch.zeros(self.input_shape).to(device)

        init_states = prev_t_1

        return init_states
    
    def allow_train_filter_coefficients(self, option:bool):

        self.first_order_low_pass_cell.alpha_1.requires_grad = option

class axon_cell(torch.nn.Module):
    '''
    implement spike response in axon
    '''
    def __init__(self, input_shape, step_num, batch_size, tau_m, tau_s, train_tau_m, train_tau_s):
        '''
        :param axon_shape: tuple (width hight) or (width, hight, depth) this should be the same as input shape,
        always on to one connection
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param tau_s:
        :param train_tau_m:
        :param train_tau_s:
        '''
        super().__init__()
        self.axon_shape = input_shape
        self.input_shape = input_shape
        self.step_num = step_num
        self.batch_size = batch_size

        self.tau_m = torch.full(input_shape, tau_m)
        self.tau_s = torch.full(input_shape, tau_s)

        # calculate the norm factor to make max spike response to be 1
        eta = torch.tensor(tau_m / tau_s)
        self.v_0 = torch.nn.Parameter(torch.pow(eta, eta / (eta - 1)) / (eta - 1))
        self.v_0.requires_grad = False

        self.decay_m = torch.nn.Parameter(torch.exp(-1 / self.tau_m))
        self.decay_m.requires_grad = train_tau_m

        self.decay_s = torch.nn.Parameter(torch.exp(-1 / self.tau_s))
        self.decay_s.requires_grad = train_tau_s

    def forward(self, current_spike, prev_states):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]
        """
        :param current_spike: [batch, dim0 ,dim1..]
        :param  prev_states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """

        prev_psp_m, prev_psp_s = prev_states

        current_psp_m = prev_psp_m * self.decay_m + current_spike
        current_psp_s = prev_psp_s * self.decay_s + current_spike
        current_psp = current_psp_m - current_psp_s

        psp = current_psp * self.v_0

        new_states = (current_psp_m, current_psp_s)

        return psp, new_states

class axon_layer(torch.nn.Module):
    def __init__(self, input_shape, step_num, batch_size, tau_m, tau_s, train_tau_m, train_tau_s, return_state=True):
        '''

        :param axon_shape: tuple (width hight) or (width, hight, depth) this should be the same as input shape,
        always on to one connection
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param tau_s:
        :param train_tau_m:
        :param train_tau_s:
        '''
        super().__init__()
        self.axon_shape = input_shape
        self.input_shape = input_shape
        self.step_num = step_num
        self.batch_size = batch_size

        self.tau_m = tau_m
        self.tau_s = tau_s

        self.return_state = return_state

        self.axon_cell = axon_cell(input_shape, step_num, batch_size, tau_m, tau_s, train_tau_m, train_tau_s)

    def forward(self, input_spikes, states=None):
        """
        :param current_spike: [batch, dim0 ,dim1..]
        :param  states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """

        if states is None:
            states = self.create_init_states()

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        psps = []
        for i in range(len(inputs)):
            psp, states = self.axon_cell(inputs[i], states)
            psps += [psp]
        
        if self.return_state:
            return torch.stack(psps,dim=-1), states
        else:
            return torch.stack(psps,dim=-1)

    def create_init_states(self):

        device = self.axon_cell.v_0.device
        init_psp_m = torch.zeros(self.input_shape).to(device)
        init_psp_s = torch.zeros(self.input_shape).to(device)

        init_states = (init_psp_m, init_psp_s)

        return init_states

    def named_parameters(self, prefix='', recurse=True):
        '''
        return nothing
        :return:
        '''
        parameter_list = []
        for elem in parameter_list:
            yield elem
    
    def allow_train_filter_coefficients(self, option:bool):

        self.axon_cell.decay_m.requires_grad = option
        self.axon_cell.decay_s.requires_grad = option


class neuron_cell(torch.nn.Module):
    def __init__(self, input_size, neuron_number, step_num, batch_size, tau_m, **kargs):
        '''

        :param input_size: int
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_decay_v:
        :param train_reset_decay:
        :param membrane_filter: True or False
        '''
        super().__init__()

        # default values
        self.options = {
        'train_decay_v'         : False,
        'train_reset_decay'     : False,
        'train_reset_v'         : False, 
        'train_threshold'       : False,
        'train_bias'            : True,
        'membrane_filter'       : False,
        'reset_v'               : 1.0
        # 'input_type'            : 'axon'
        }

        self.options.update(kargs)

        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        # self.input_type = self.options['input_type']
        self.train_threshold = self.options['train_threshold']

        # if self.input_type != 'axon':
        #     # input size: [synapse_number, neuron_number]
        #     # in this case, the input comes from synapse_layer, not axon_layer
        #     # different is that in synapse_layer, each synapse has its own states
        #     # while in axon_layer, each axon has its states
        #     weight = torch.empty([input_size,neuron_number])
        #     torch.nn.init.xavier_uniform(weight)
        #     self.input_type = 'synapse'
        #     self.weight = torch.nn.Parameter(weight)
        # else:
        self.weight = torch.nn.Linear(input_size, neuron_number, bias=self.options['train_bias'])
        self.tau_m = torch.full((neuron_number,), tau_m)

        self.sigma = torch.nn.Parameter(torch.tensor(0.4))
        self.sigma.requires_grad = False

        self.reset_decay = torch.exp(torch.tensor(-1.0/tau_m))
        self.reset_decay = torch.nn.Parameter(torch.full((self.neuron_number,), self.reset_decay))
        self.reset_decay.requires_grad = self.options['train_reset_decay']
        
        self.reset_v = torch.nn.Parameter(torch.full((self.neuron_number,), self.options['reset_v']))
        self.reset_v.requires_grad = self.options['train_reset_v']

        self.decay_v = torch.exp(torch.tensor(-1/tau_m))
        self.decay_v = torch.nn.Parameter(torch.full((self.neuron_number,),self.decay_v))
        self.decay_v.requires_grad = self.options['train_decay_v']

        self.threshold_offset = torch.nn.Parameter(torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1])).sample([neuron_number]).reshape(-1))

        self.threshold_offset.requires_grad = self.options['train_threshold']

        self.enable_threshold = True
        self.membrane_filter = self.options['membrane_filter']

    def forward(self, current_input, prev_states):
        """
        :param current_input: [batch, input_size]
        :param  prev_states: tuple (prev_v, prev_reset)
        :return: spike, news_state
        """
        prev_v, prev_reset = prev_states

        # if self.input_type != 'axon':
        #     # input is synapse, the input shape is [synapse_number, neuron_number]
        #     # the shape of weight is also [synapse_number, neuron_number], each input represent the
        #     # state of the synapse.
        #     # we do element wise multiplication between weight and synapse states. and then sum along
        #     # dimension 1, so we get 1d tensor of shape [neuron_number], each element is a sum of weighted spike
        #     # response of a neuron
        #     weighted_psp = self.weight * current_input
        #     weighted_psp = weighted_psp.sum(dim=1)
        # else:
        weighted_psp = self.weight(current_input)

        if self.membrane_filter:
            current_v = prev_v * self.decay_v + weighted_psp - prev_reset
        else:
            current_v = weighted_psp - prev_reset

        if self.train_threshold:
            current_v = current_v + self.threshold_offset

        if self.enable_threshold:
            threshold_function = threshold.apply
            spike = threshold_function(current_v, self.sigma)
        else:
            spike = current_v.clamp(0.0, 1.0)

        current_reset = prev_reset * self.reset_decay + spike * self.reset_v

        if self.train_threshold:
            current_v = current_v - self.threshold_offset

        new_states = (current_v, current_reset)

        return spike, new_states

class neuron_layer(torch.nn.Module):
    def __init__(self, input_size, neuron_number, step_num, batch_size, tau_m, **kargs):
        '''

        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_decay_v:
        :param train_reset_decay:
        :param train_reset_v:
        :param train_threshold:
        :param train_bias:
        :param membrane_filter:
        '''
        super().__init__()

        # default values
        self.options = {
        'train_decay_v'         : False,
        'train_reset_decay'     : False,
        'train_reset_v'         : False, 
        'train_threshold'       : False,
        'train_bias'            : True,
        'membrane_filter'       : False,
        'reset_v'               : 1.0,
        # 'input_type'            : 'axon'
        'return_state'            : True,
        'synapse_type'          : None
        }

        self.options.update(kargs)

        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        self.return_state = self.options['return_state']

        self.synapse_type = self.options['synapse_type']
        self.synapse_filter = None

        if self.synapse_type == 'none' or self.synapse_type == None:
            pass
        elif self.synapse_type == 'first_order_low_pass':
            self.synapse_filter = first_order_low_pass_layer((input_size,), step_num, batch_size, 
                                        kargs['synapse_tau_s'],
                                        kargs['train_synapse_tau'])
        elif self.synapse_type == 'dual_exp':
            self.synapse_filter = axon_layer((input_size,), step_num, self.batch_size, 
                                        kargs['synapse_tau_m'], 
                                        kargs['synapse_tau_s'], 
                                        kargs['train_synapse_tau'], 
                                        kargs['train_synapse_tau'])
        else:
            raise Exception("unrecognized synapse filter type")

        self.neuron_cell = neuron_cell(input_size, neuron_number, step_num, batch_size, tau_m, **self.options)

    def forward(self, input_spikes, states=None):
        """
        :param input_spikes: [batch, dim0 ,dim1..]
        :param  states: tuple (init_v, init_reset_v)
        :return:
        """

        if self.synapse_filter is not None:
            x,_ = self.synapse_filter(input_spikes)
        else:
            x = input_spikes

        if states is None:
            states = self.create_init_states()

        # unbind along last dimension
        inputs = x.unbind(dim=-1)
        spikes = []
        for i in range(len(inputs)):
            spike, states = self.neuron_cell(inputs[i], states)
            spikes += [spike]
        
        if self.return_state:
            return torch.stack(spikes,dim=-1), states
        else:
            return torch.stack(spikes,dim=-1)

    def create_init_states(self):

        device = self.neuron_cell.reset_decay.device
        init_v = torch.zeros(self.neuron_number).to(device)
        init_reset_v = torch.zeros(self.neuron_number).to(device)

        init_states = (init_v, init_reset_v)

        return init_states

    def named_parameters(self, prefix='', recurse=True):
        '''
        only return weight in neuron cell
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.named_parameters
        :return:
        '''
        for name, param in self.neuron_cell.weight.named_parameters(recurse=recurse):
            yield name, param
    
    def allow_train_filter_coefficients(self, option:bool):

        self.train_decay_v = option
        self.neuron_cell.decay_v.requires_grad = option
        self.neuron_cell.train_decay_v = option


class if_encoder(torch.nn.Module):
    def __init__(self, input_shape, step_num, batch_size, threshold=1.0, reset_mode='soft', return_state=True):
        '''
        Assumes 1 to 1 connection
        :param input_shape: int or tuple[width,hight] or tuple [depth, width, hight]
        :param step_num:
        :param batch_size:
        :param threshold:
        :param reset_mode: soft or hard
        '''

        super().__init__()
        self.input_shape = input_shape
        self.step_num = step_num
        self.batch_size = batch_size
        self.reset_mode = reset_mode
        self.return_state = return_state

        # self.weight = self.weight = torch.nn.Parameter(torch.Tensor(input_size))

        # self.threshold = torch.nn.Parameter(torch.full((self.neuron_number,), 1.0))

        self.threshold = torch.nn.Parameter(torch.tensor(1.0))

        self.threshold.requires_grad = False

    def forward(self, input_spikes, states=None):
        """
        :param input_spikes: [batch, dim0 ,dim1..]
        :param  states: initial states, tuple (v, reset_v)
        :return:
        """

        if states is None:
            states = self.create_init_states()

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        spikes = []

        current_v = states
        for i in range(len(inputs)):

            current_v = current_v + inputs[i]

            spike = current_v.clone()

            spike[spike < self.threshold] = 0.0
            spike[spike >= self.threshold] = 1.0

            if self.reset_mode == 'soft':
                current_v[current_v >= self.threshold] = current_v[current_v >= self.threshold] - self.threshold
            else:
                current_v[current_v >= self.threshold] = 0.0

            spikes += [spike]

        if self.return_state:
            return torch.stack(spikes,dim=-1), current_v
        else:
            return torch.stack(spikes,dim=-1)

    def create_init_states(self):

        device = self.threshold.device

        if isinstance(self.input_shape, int):
            init_v = torch.zeros((self.batch_size, self.input_shape)).to(device)
        elif isinstance(self.input_shape, tuple):
            init_v = torch.zeros((self.batch_size, *self.input_shape)).to(device)

        init_states = init_v

        return init_states

class encoder_layer(torch.nn.Module):
    def __init__(self, input_size, neuron_number, step_num, batch_size, tau_m, gain=None, bias=None, train_gain_bias=True,
                 train_encoder_weight=True, connection_type='full', v_calculate_method=0, return_state=True):
        '''
        TODO: need to support multi-dimension input!
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        '''
        super().__init__()
        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        self.connection_type = connection_type
        self.train_encoder_weight = train_encoder_weight
        self.train_gain_bias = train_gain_bias
        self.return_state = return_state

        # if connection type is 1to1, neuron number should be same as input size
        # discard the neuron_number argument
        if connection_type == '1to1':
            self.neuron_number=input_size

        self.use_merged_bias = True
        if (v_calculate_method == 2) or (v_calculate_method == 3):
            self.use_merged_bias = False

        if self.connection_type == 'full':
            self.weight = torch.nn.Linear(input_size, neuron_number, bias=self.use_merged_bias)
            self.weight.weight.requires_grad = train_encoder_weight
        elif self.connection_type == '1to1':
            self.weight = torch.nn.Parameter(torch.Tensor(input_size))
            torch.nn.init.uniform_(self.weight, -1.0, 1.0)
            self.use_merged_bias = False
            self.weight.requires_grad = self.train_encoder_weight

        if self.use_merged_bias == True:
            if bias is not None:
                # bias is already initialized inside Linear object, if bias is provided, overwrite the bias in Linear
                # object by provided bias
                self.weight.bias = torch.nn.Parameter(torch.Tensor(bias/gain))

            self.weight.bias.requires_grad = train_gain_bias
            #separate bias should be 0, and disable gradient
            self.bias = torch.nn.Parameter(torch.zeros(neuron_number))
            self.bias.requires_grad = False
        else:
            # if use separate bias, the bias in Linear object is none when the Linear object is created,
            # here to create the separate bias
            if bias is not None:
                self.bias = torch.nn.Parameter(torch.Tensor(bias))
            else:
                self.bias = torch.nn.Parameter(torch.Tensor(neuron_number))
                if self.connection_type == '1to1':
                    torch.nn.init.uniform_(self.bias, -1.0, 1.0)
                else:
                    # bias init.
                    # reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
                    # if bias is provided, use provided bias, otherwise initialize use pytorch method
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.weight)
                    bound = 1 / np.sqrt(fan_in)
                    torch.nn.init.uniform_(self.bias, -bound, bound)
            self.bias.requires_grad = train_gain_bias

        # self.gain = torch.full((neuron_number,), gain)
        # self.gain = torch.nn.Parameter(self.gain)

        # init_gain = torch.Tensor(neuron_number)
        # torch.nn.init.uniform_(init_gain, a=gain[0], b=gain[0])
        # self.gain = torch.nn.Parameter(init_gain)

        if gain is not None:
            init_gain = torch.Tensor(gain)
            self.gain = torch.nn.Parameter(init_gain)
        else:
            # is gain is not provided, use 1 as gain
            self.gain = torch.nn.Parameter(torch.ones(neuron_number))

        self.gain.requires_grad = train_gain_bias

        self.tau_m = torch.full((input_size,), tau_m)

        self.sigma = torch.nn.Parameter(torch.tensor(0.4))
        self.sigma.requires_grad = False

        self.reset_decay = torch.exp(torch.tensor(-1.0/tau_m))
        self.reset_decay = torch.nn.Parameter(torch.full((self.neuron_number,),self.reset_decay))
        self.reset_decay.requires_grad = False

        self.reset_v = torch.nn.Parameter(torch.full((self.neuron_number,), 1.0))
        self.reset_v.requires_grad = False

        self.decay_v = torch.exp(torch.tensor(-1/tau_m))
        self.decay_v = torch.nn.Parameter(torch.full((self.neuron_number,),self.decay_v))
        self.decay_v.requires_grad = False

        self.enable_threshold = True

        self.v_calculate_method = v_calculate_method


    def forward(self, input_spikes, states=None):
        """
        :param input_spikes: [batch, dim0 ,dim1..]
        :param  states: initial states, tuple (v, reset_v)
        :return:
        """

        if states is None:
            states = self.create_init_states()

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        spikes = []

        current_v, current_reset = states
        for i in range(len(inputs)):

            if self.connection_type == 'full':
                weighted_psp = self.weight(inputs[i])
            elif self.connection_type == '1to1':
                weighted_psp = self.weight * inputs[i]

            # ref: https://www.nengo.ai/nengo/_modules/nengo/neurons.html#LIFRate.gain_bias

            if self.v_calculate_method == 0:
                current_v = current_v * self.decay_v + (self.gain * weighted_psp) * (1 - self.decay_v)
            elif self.v_calculate_method == 1:
                current_v = current_v * self.decay_v + (self.gain * weighted_psp)
            elif self.v_calculate_method == 2:
                current_v = current_v * self.decay_v + (self.gain * weighted_psp + self.bias) * (1 - self.decay_v)
            elif self.v_calculate_method == 3:
                current_v = current_v * self.decay_v + (self.gain * weighted_psp + self.bias)

            if self.enable_threshold:
                threshold_function = threshold.apply
                spike = threshold_function(current_v, self.sigma)
            else:
                spike = current_v.clamp(0.0, 1.0)

            current_reset = current_reset * self.reset_decay + spike * self.reset_v

            spikes += [spike]

        new_states = (current_v, current_reset)

        if self.return_state:
            return torch.stack(spikes,dim=-1), new_states
        else:
            return torch.stack(spikes,dim=-1)

    def create_init_states(self):

        device = self.reset_decay.device
        init_v = torch.zeros(self.neuron_number).to(device)
        init_reset_v = torch.zeros(self.neuron_number).to(device)

        init_states = (init_v, init_reset_v)

        return init_states

class encoder_layer_2d_1to1(torch.nn.Module):
    def __init__(self, input_shape, step_num, batch_size, tau_m, gain=None, bias=None, train_gain_bias=True,
                 train_encoder_weight=True, v_calculate_method=0, return_state=True):
        '''
        LIF neuron, convert input (3 channel image) to spike trains by 1 to 1 connection
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        '''
        super().__init__()
        self.input_shape = input_shape
        self.step_num = step_num
        self.batch_size = batch_size
        self.train_encoder_weight = train_encoder_weight
        self.train_gain_bias = train_gain_bias
        self.return_state = return_state

        self.weight = torch.nn.Parameter(torch.ones(input_shape))
        torch.nn.init.uniform_(self.weight, -1.0, 1.0)
        self.weight.requires_grad = self.train_encoder_weight

        if bias is not None:
            self.bias = torch.nn.Parameter(torch.full(input_shape, bias))
        else:
            self.bias = torch.nn.Parameter(torch.zeros(input_shape))
            torch.nn.init.uniform_(self.bias, -1.0, 1.0)
        self.bias.requires_grad = train_gain_bias

        if gain is not None:
            init_gain = torch.torch.full(input_shape, gain)
            self.gain = torch.nn.Parameter(init_gain)
        else:
            # is gain is not provided, use 1 as gain
            self.gain = torch.nn.Parameter(torch.ones(input_shape))

        self.gain.requires_grad = train_gain_bias

        self.tau_m = torch.full(input_shape, tau_m)

        self.sigma = torch.nn.Parameter(torch.tensor(0.4),requires_grad = False)

        self.reset_decay = torch.exp(torch.tensor(-1.0/tau_m))
        self.reset_decay = torch.nn.Parameter(torch.full(input_shape,self.reset_decay), requires_grad=False)

        self.reset_v = torch.nn.Parameter(torch.full(input_shape, 1.0), requires_grad=False)

        self.decay_v = torch.exp(torch.tensor(-1/tau_m))
        self.decay_v = torch.nn.Parameter(torch.full(input_shape,self.decay_v), requires_grad=False)

        self.enable_threshold = True

        self.v_calculate_method = v_calculate_method


    def forward(self, input_spikes, states=None):
        """
        :param input_spikes: [batch, dim0 ,dim1..]
        :param  states: initial states, tuple (v, reset_v)
        :return:
        """

        if states is None:
            states = self.create_init_states()

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        spikes = []

        current_v, current_reset = states
        for i in range(len(inputs)):

            weighted_psp = self.weight * inputs[i]

            # ref: https://www.nengo.ai/nengo/_modules/nengo/neurons.html#LIFRate.gain_bias

            if self.v_calculate_method == 0:
                current_v = current_v * self.decay_v + (self.gain * weighted_psp) * (1 - self.decay_v)
            elif self.v_calculate_method == 1:
                current_v = current_v * self.decay_v + (self.gain * weighted_psp)
            elif self.v_calculate_method == 2:
                current_v = current_v * self.decay_v + (self.gain * weighted_psp + self.bias) * (1 - self.decay_v)
            elif self.v_calculate_method == 3:
                current_v = current_v * self.decay_v + (self.gain * weighted_psp + self.bias)

            if self.enable_threshold:
                threshold_function = threshold.apply
                spike = threshold_function(current_v, self.sigma)
            else:
                spike = current_v.clamp(0.0, 1.0)

            current_reset = current_reset * self.reset_decay + spike * self.reset_v

            spikes += [spike]

        new_states = (current_v, current_reset)

        if self.return_state:
            return torch.stack(spikes,dim=-1), new_states
        else:
            return torch.stack(spikes,dim=-1)

    def create_init_states(self):

        device = self.reset_decay.device
        init_v = torch.zeros(self.input_shape).to(device)
        init_reset_v = torch.zeros(self.input_shape).to(device)

        init_states = (init_v, init_reset_v)

        return init_states

class conv2d_cell(torch.nn.Module):
    def __init__(self, h_input, w_input, in_channels, out_channels, kernel_size, stride, padding, dilation, step_num, batch_size,
                 tau_m, **kargs):
        '''
        :param input_size: int
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_decay_v:
        :param train_reset_decay:
        :param membrane_filter: True or False
        '''
        super().__init__()

        # default values
        self.options = {
        'train_decay_v'         : False,
        'train_reset_decay'     : False,
        'train_reset_v'         : False, 
        'train_threshold'       : False,
        'train_bias'            : True,
        'membrane_filter'       : False,
        'reset_v'               : 1.0,
        'input_type'            : 'axon'}

        self.options.update(kargs)

        self.step_num = step_num
        self.batch_size = batch_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.h_input = h_input
        self.w_input =  w_input

        self.train_threshold = self.options['train_threshold']

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias= self.options['train_bias'])

        conv_out_h, conv_out_w = calculate_conv2d_outsize(h_input,w_input,padding,kernel_size,stride)

        #output shape will be (batch, out_channels, conv_out_h, conv_out_w)
        #this is also the shape of neurons and time constants and reset_v
        self.output_shape = (out_channels, conv_out_h, conv_out_w)

        self.sigma = torch.nn.Parameter(torch.tensor(0.4))
        self.sigma.requires_grad = False

        self.reset_decay = torch.exp(torch.tensor(-1.0/tau_m))
        self.reset_decay = torch.nn.Parameter(torch.full(self.output_shape,self.reset_decay))
        self.reset_decay.requires_grad =  self.options['train_reset_decay']

        self.reset_v = torch.nn.Parameter(torch.full(self.output_shape, 1.0))
        self.reset_v.requires_grad =  self.options['train_reset_v']

        self.decay_v = torch.exp(torch.tensor(-1/tau_m))
        self.decay_v = torch.nn.Parameter(torch.full(self.output_shape,self.decay_v))
        self.decay_v.requires_grad =  self.options['train_decay_v']

        self.threshold_offset = torch.nn.Parameter(torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1])).sample(self.output_shape).squeeze())

        self.threshold_offset.requires_grad =  self.options['train_threshold']

        self.enable_threshold = True
        self.membrane_filter =  self.options['membrane_filter']

    def forward(self, current_input, prev_states):
        """
        :param current_input: [batch, input_size]
        :param  prev_states: tuple (prev_v, prev_reset)
        :return:
        """
        prev_v, prev_reset = prev_states

        # if self.input_type != 'axon':
        #     # input is synapse, the input shape is [synapse_nummber, neuron_number]
        #     # the shape of weight is also [synapse_nummber, neuron_number], each input represent the
        #     # state of the synapse.
        #     # we do element wise multiplication between weight and synapse states. and then sum along
        #     # dimension 1, so we get 1d tensor of shape [neuron_number], each element is a sum of weighted spike
        #     # response of a neuron
        #     weighted_psp = self.weight * current_input
        #     weighted_psp = weighted_psp.sum(dim=1)
        # else:
        #     weighted_psp = self.weight(current_input)

        conv2d_out = self.conv(current_input)

        if self.membrane_filter:
            current_v = prev_v * self.decay_v + conv2d_out - prev_reset
        else:
            current_v = conv2d_out - prev_reset

        if self.train_threshold:
            current_v = current_v + self.threshold_offset

        if self.enable_threshold:
            threshold_function = threshold.apply
            spike = threshold_function(current_v, self.sigma)
        else:
            spike = current_v.clamp(0.0, 1.0)
            # print('spike', spike)

        current_reset = prev_reset * self.reset_decay + spike * self.reset_v

        if self.train_threshold:
            current_v = current_v - self.threshold_offset

        new_states = (current_v, current_reset)

        return spike, new_states

class conv2d_layer(torch.nn.Module):
    def __init__(self, h_input, w_input, in_channels, out_channels, kernel_size, stride, padding, dilation, step_num, batch_size,
                 tau_m, **kwargs):
        '''
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_decay_v:
        :param train_reset_decay:
        :param train_reset_v:
        :param train_threshold:
        :param train_bias:
        :param membrane_filter:
        '''
        super().__init__()

        # default values
        self.options = {
        'train_decay_v'         : False,
        'train_reset_decay'     : False,
        'train_reset_v'         : False, 
        'train_threshold'       : False,
        'train_bias'            : True,
        'membrane_filter'       : False,
        'reset_v'               : 1.0,
        'input_type'            : 'axon',
        'return_state'          : True,
        'synapse_type'          : None}

        self.options.update(kwargs)

        self.step_num = step_num
        self.batch_size = batch_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.h_input = h_input
        self.w_input = w_input
        self.return_state = self.options['return_state']

        self.synapse_type = self.options['synapse_type']
        self.synapse_filter = None

        conv_out_h, conv_out_w = calculate_conv2d_outsize(h_input, w_input, padding, kernel_size, stride)
        self.output_shape = (out_channels, conv_out_h, conv_out_w)

        if self.synapse_type == 'none':
            pass
        elif self.synapse_type == 'first_order_low_pass':
            self.synapse_filter = first_order_low_pass_layer((in_channels,h_input,w_input), step_num, batch_size, 
                                        kwargs['synapse_tau_s'],
                                        kwargs['train_synapse_tau'])
        elif self.synapse_type == 'dual_exp':
            self.synapse_filter = axon_layer((in_channels,h_input,w_input), step_num, self.batch_size, 
                                        kwargs['synapse_tau_m'], 
                                        kwargs['synapse_tau_s'], 
                                        kwargs['train_synapse_tau'], 
                                        kwargs['train_synapse_tau'])
        else:
            raise Exception("unrecognized synapse filter type")


        self.conv2d_cell = conv2d_cell(h_input, w_input, in_channels, out_channels, kernel_size, stride, padding, dilation, step_num, batch_size,
                 tau_m, **self.options)

    def forward(self, input_spikes, states=None):
        """
        :param input_spikes: [batch, dim0 ,dim1..,t]
        :param  prev_states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """

        if self.synapse_filter is not None:
            x,_ = self.synapse_filter(input_spikes)
        else:
            x = input_spikes

        if states is None:
            states = self.create_init_states()

        # unbind along last dimension
        inputs = x.unbind(dim=-1)
        spikes = []
        for i in range(len(inputs)):
            spike, states = self.conv2d_cell(inputs[i], states)
            spikes += [spike]
        
        if self.return_state:
            return torch.stack(spikes,dim=-1), states
        else:
            return torch.stack(spikes,dim=-1)

    def create_init_states(self):

        device = self.conv2d_cell.reset_decay.device

        init_v = torch.zeros(self.output_shape).to(device)
        init_reset_v = torch.zeros(self.output_shape).to(device)

        init_states = (init_v, init_reset_v)

        return init_states

    def named_parameters(self, prefix='', recurse=True):
        '''
        only return weight in neuron cell
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.named_parameters
        :return:
        '''
        for name, param in self.conv2d_cell.conv.named_parameters(recurse=recurse):
            yield name, param

class maxpooling2d_layer(torch.nn.Module):
    def __init__(self, h_input, w_input, in_channels, kernel_size, stride, padding, dilation, step_num, batch_size):
        '''
        2d max pooling, input should be the output of axon, it pools the axon's psp
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        '''
        super().__init__()
        # self.input_size = input_size
        # self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        # self.input_type = input_type

        self.maxpooling2d = torch.nn.MaxPool2d(kernel_size,stride,padding,dilation)

        self.output_shape = calculate_maxpooling2d_outsize(h_input, w_input, padding, kernel_size, stride)


    def forward(self, input_psp):
        """
        :param input_spikes: [batch, dim0 ,dim1..,t]
        :param  prev_states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """

        # unbind along last dimension
        inputs = input_psp.unbind(dim=-1)
        pooled_psp = []
        for i in range(len(inputs)):
            psp = self.maxpooling2d(inputs[i])
            pooled_psp += [psp]
        return torch.stack(pooled_psp,dim=-1)

    # def create_init_states(self):
    #
    #     device = self.conv2d_cell.reset_decay.device
    #
    #     init_v = torch.zeros(self.output_shape).to(device)
    #     init_reset_v = torch.zeros(self.output_shape).to(device)
    #
    #     init_states = (init_v, init_reset_v)
    #
    #     return init_states


class neuron_layer_bntt(torch.nn.Module):
    def __init__(self, input_size, neuron_number, step_num, batch_size, tau_m, **kargs):
        '''
        batch norm through time, ref； https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time/blob/main/model.py
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_decay_v:
        :param train_reset_decay:
        :param train_reset_v:
        :param train_threshold:
        :param train_bias:
        :param membrane_filter:
        '''
        super().__init__()

        # default values
        self.options = {
        'train_decay_v'         : False,
        'train_reset_decay'     : False,
        'train_reset_v'         : False, 
        'train_threshold'       : False,
        'train_bias'            : True,
        'membrane_filter'       : False,
        'reset_v'               : 1.0,
        # 'input_type'            : 'axon'
        'return_state'            : True,
        'synapse_type'          : None
        }

        self.options.update(kargs)

        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        self.return_state = self.options['return_state']
        self.enable_threshold = True

        self.synapse_type = self.options['synapse_type']
        self.synapse_filter = None

        if self.synapse_type == 'none':
            pass
        elif self.synapse_type == 'first_order_low_pass':
            self.synapse_filter = first_order_low_pass_layer((input_size,), step_num, batch_size, 
                                        kargs['synapse_tau_s'],
                                        kargs['train_synapse_tau'])
        elif self.synapse_type == 'dual_exp':
            self.synapse_filter = axon_layer((input_size,), step_num, self.batch_size, 
                                        kargs['synapse_tau_m'], 
                                        kargs['synapse_tau_s'], 
                                        kargs['train_synapse_tau'], 
                                        kargs['train_synapse_tau'])
        else:
            raise Exception("unrecognized synapse filter type")

        self.weight = torch.nn.Linear(input_size, neuron_number, bias=self.options['train_bias'])
        self.tau_m = torch.full((neuron_number,), tau_m)

        self.sigma = torch.nn.Parameter(torch.tensor(0.4))
        self.sigma.requires_grad = False

        self.reset_decay = torch.exp(torch.tensor(-1.0/tau_m))
        self.reset_decay = torch.nn.Parameter(torch.full((self.neuron_number,), self.reset_decay))
        self.reset_decay.requires_grad = self.options['train_reset_decay']
        
        self.reset_v = torch.nn.Parameter(torch.full((self.neuron_number,), self.options['reset_v']))
        self.reset_v.requires_grad = self.options['train_reset_v']

        self.decay_v = torch.exp(torch.tensor(-1/tau_m))
        self.decay_v = torch.nn.Parameter(torch.full((self.neuron_number,),self.decay_v))
        self.decay_v.requires_grad = self.options['train_decay_v']

        self.threshold_offset = torch.nn.Parameter(torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1])).sample([neuron_number]).reshape(-1))

        self.threshold_offset.requires_grad = self.options['train_threshold']
        self.train_threshold = self.options['train_threshold']

        self.enable_threshold = True
        self.membrane_filter = self.options['membrane_filter']

        self.bntt = torch.nn.ModuleList([torch.nn.BatchNorm1d(neuron_number, eps=1e-4, momentum=0.1, affine=True) for i in range(self.step_num)])

        for bntt_layer in self.bntt:
            bntt_layer.bias = None

    def forward(self, input_spikes, states=None):
        """
        :param input_spikes: [batch, dim0 ,dim1..]
        :param  states: tuple (init_v, init_reset_v)
        :return:
        """
        
        if self.synapse_filter is not None:
            x,_ = self.synapse_filter(input_spikes)
        else:
            x = input_spikes
    
        if states is None:
            prev_v, prev_reset = self.create_init_states()

        # unbind along last dimension
        inputs = x.unbind(dim=-1)
        spikes = []

        for i in range(len(inputs)):
            
            weighted_psp = self.weight(inputs[i])

            if self.membrane_filter:
                current_v = prev_v * self.decay_v + self.bntt[i](weighted_psp) - prev_reset
            else:
                current_v = weighted_psp - prev_reset

            if self.train_threshold:
                current_v = current_v + self.threshold_offset

            if self.enable_threshold:
                threshold_function = threshold.apply
                spike = threshold_function(current_v, self.sigma)
            else:
                spike = current_v.clamp(0.0, 1.0)

            current_reset = prev_reset * self.reset_decay + spike * self.reset_v

            if self.train_threshold:
                current_v = current_v - self.threshold_offset
            
            spikes += [spike]

            prev_v = current_v
            prev_reset = current_reset

        if self.return_state:
            return torch.stack(spikes,dim=-1), (current_v, current_reset)
        else:
            return torch.stack(spikes,dim=-1)

    def create_init_states(self):

        device = self.reset_decay.device
        init_v = torch.zeros(self.neuron_number).to(device)
        init_reset_v = torch.zeros(self.neuron_number).to(device)

        init_states = (init_v, init_reset_v)

        return init_states

    # def named_parameters(self, prefix='', recurse=True):
    #     '''
    #     only return weight in neuron cell
    #     https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.named_parameters
    #     :return:
    #     '''
    #     for name, param in self.neuron_cell.weight.named_parameters(recurse=recurse):
    #         yield name, param

class conv2d_layer_bntt(torch.nn.Module):
    def __init__(self, h_input, w_input, in_channels, out_channels, kernel_size, stride, padding, dilation, step_num, batch_size,
                 tau_m, **kwargs):
        '''
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_decay_v:
        :param train_reset_decay:
        :param train_reset_v:
        :param train_threshold:
        :param train_bias:
        :param membrane_filter:
        '''
        super().__init__()

        # default values
        self.options = {
        'train_decay_v'         : False,
        'train_reset_decay'     : False,
        'train_reset_v'         : False, 
        'train_threshold'       : False,
        'train_bias'            : True,
        'membrane_filter'       : False,
        'reset_v'               : 1.0,
        'input_type'            : 'axon',
        'return_state'          : True,
        'synapse_type'          : None}

        self.options.update(kwargs)

        self.step_num = step_num
        self.batch_size = batch_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.h_input = h_input
        self.w_input = w_input
        self.return_state = self.options['return_state']

        self.enable_threshold = True
        self.membrane_filter =  self.options['membrane_filter']

        self.synapse_type = self.options['synapse_type']
        self.synapse_filter = None

        conv_out_h, conv_out_w = calculate_conv2d_outsize(h_input, w_input, padding, kernel_size, stride)
        self.output_shape = (out_channels, conv_out_h, conv_out_w)

        if self.synapse_type == 'none':
            pass
        elif self.synapse_type == 'first_order_low_pass':
            self.synapse_filter = first_order_low_pass_layer((in_channels,h_input,w_input), step_num, batch_size, 
                                        kwargs['synapse_tau_s'],
                                        kwargs['train_synapse_tau'])
        elif self.synapse_type == 'dual_exp':
            self.synapse_filter = axon_layer((in_channels,h_input,w_input), step_num, self.batch_size, 
                                        kwargs['synapse_tau_m'], 
                                        kwargs['synapse_tau_s'], 
                                        kwargs['train_synapse_tau'], 
                                        kwargs['train_synapse_tau'])
        else:
            raise Exception("unrecognized synapse filter type")

        self.train_threshold = self.options['train_threshold']

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias= self.options['train_bias'])

        conv_out_h, conv_out_w = calculate_conv2d_outsize(h_input,w_input,padding,kernel_size,stride)

        #output shape will be (batch, out_channels, conv_out_h, conv_out_w)
        #this is also the shape of neurons and time constants and reset_v
        self.output_shape = (out_channels, conv_out_h, conv_out_w)

        self.sigma = torch.nn.Parameter(torch.tensor(0.4))
        self.sigma.requires_grad = False

        self.reset_decay = torch.exp(torch.tensor(-1.0/tau_m))
        self.reset_decay = torch.nn.Parameter(torch.full(self.output_shape,self.reset_decay))
        self.reset_decay.requires_grad =  self.options['train_reset_decay']

        self.reset_v = torch.nn.Parameter(torch.full(self.output_shape, 1.0))
        self.reset_v.requires_grad =  self.options['train_reset_v']

        self.decay_v = torch.exp(torch.tensor(-1/tau_m))
        self.decay_v = torch.nn.Parameter(torch.full(self.output_shape,self.decay_v))
        self.decay_v.requires_grad =  self.options['train_decay_v']

        self.threshold_offset = torch.nn.Parameter(torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1])).sample(self.output_shape).squeeze())

        self.threshold_offset.requires_grad =  self.options['train_threshold']

        self.bntt = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.out_channels, eps=1e-4, momentum=0.1, affine=True) for i in range(self.step_num)])

        for bntt_layer in self.bntt:
            bntt_layer.bias = None

    def forward(self, input_spikes, states=None):
        """
        :param input_spikes: [batch, dim0 ,dim1..,t]
        :param  prev_states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """

        if self.synapse_filter is not None:
            x,_ = self.synapse_filter(input_spikes)
        else:
            x = input_spikes

        if states is None:
            prev_v, prev_reset = self.create_init_states()

        # unbind along last dimension
        inputs = x.unbind(dim=-1)
        spikes = []
        for i in range(len(inputs)):

            conv2d_out = self.conv(inputs[i])

            if self.membrane_filter:
                current_v = prev_v * self.decay_v + self.bntt[i](conv2d_out) - prev_reset
            else:
                current_v = self.bntt[i](conv2d_out) - prev_reset

            if self.train_threshold:
                current_v = current_v + self.threshold_offset

            if self.enable_threshold:
                threshold_function = threshold.apply
                spike = threshold_function(current_v, self.sigma)
            else:
                spike = current_v.clamp(0.0, 1.0)
                # print('spike', spike)

            current_reset = prev_reset * self.reset_decay + spike * self.reset_v

            if self.train_threshold:
                current_v = current_v - self.threshold_offset
            
            prev_v = current_v
            prev_reset = current_reset

            spikes += [spike]
        
        if self.return_state:
            return torch.stack(spikes,dim=-1), (current_v, current_reset)
        else:
            return torch.stack(spikes,dim=-1)

    def create_init_states(self):

        device = self.reset_decay.device

        init_v = torch.zeros(self.output_shape).to(device)
        init_reset_v = torch.zeros(self.output_shape).to(device)

        init_states = (init_v, init_reset_v)

        return init_states

    # def named_parameters(self, prefix='', recurse=True):
    #     '''
    #     only return weight in neuron cell
    #     https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.named_parameters
    #     :return:
    #     '''
    #     for name, param in self.conv2d_cell.conv.named_parameters(recurse=recurse):
    #         yield name, param

