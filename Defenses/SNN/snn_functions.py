
import torch
import numpy as np


class threshold(torch.autograd.Function):
    """
    heaviside step threshold function
    """

    @staticmethod
    def forward(ctx, input, sigma):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        ctx.sigma = sigma

        output = input.clone()

        # gpu: 0.182
        # cpu: 0.143
        # output[output < 1] = 0
        # output[output >= 1] = 1

        # no cuda
        # gpu: 0.157s
        # cpu: 0.137
        output = torch.max(torch.tensor(0.0,device=output.device),torch.sign(output-1.0))

        # use cude
        # output = threshold_cuda.forward(input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # a = time.time()
        input, = ctx.saved_tensors
        sigma = ctx.sigma

        # start = time.time()
        exponent = -torch.pow((1-input), 2)/(2.0*sigma**2)
        exp = torch.exp(exponent)
        # print("exp:", time.time()-start)

        # compute square root of pi and 2
        # sqrt_pi = torch.tensor(3.1415926).sqrt()
        # sqrt_2 = torch.tensor(3.2).sqrt()
        #
        # erfc_grad = exp/(sqrt_pi*sqrt_2*sigma)
        # grad = erfc_grad*grad_output

        # no cuda
        # don't compute square root of pi and 2 everytime
        # this improves 0.013s on gpu and cpu
        # use above implementation, around 0.2s on cpu for 100 steps
        # below take around 0.17s
        # start = time.time()
        # erfc_grad = exp / (1.7724538509055159 * 1.4142135623730951 * sigma)
        erfc_grad = exp / (2.5066282746310007 * sigma)
        grad = erfc_grad * grad_output
        # print(time.time()-start)
        # print(grad.shape)
        # print("all", time.time()-a)

        # cuda
        # erfc_grad = threshold_cuda.backward(input, sigma)
        grad = erfc_grad * grad_output

        return grad, None

class erfc(torch.autograd.Function):
    """
    f(x) = 1/2 * erfc(-x/(sqrt(2)* sigma))
    f'(x) = e^(-x^2/(2*sigma^2)) / (sqrt(2*pi)*sigma)
    sigma = 0.4
    """

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)

        output = input.clone()

        output = torch.max(torch.tensor(0.0,device=output.device),torch.sign(output-1.0))

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors

        exponent = -torch.pow((1-input), 2)/(2.0*0.4**2)
        exp = torch.exp(exponent)

        erfc_grad = exp / (2.5066282746310007 * 0.4)
        grad = erfc_grad * grad_output

        return grad

class threshold_logistic(torch.autograd.Function):
    """
    f(x) = 1/(1+e^(-x))
    f'(x) = e^(-x) / (1+e^(-x))^2
    """

    @staticmethod
    def forward(ctx, input):
        """

        """
        # a asjusts the max value of gradient and sharpness
        # b moves gradient to left (positive) or right (negative)
        a = 4 # set to 4 as it sets max value to 1
        b = -1
        ctx.save_for_backward(input)
        ctx.a = a
        ctx.b = b

        output = input.gt(1.0).float()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """

        """

        input, = ctx.saved_tensors
        a = ctx.a
        b = ctx.b

        x = input
        numerator = torch.exp(1-x)
        denominator = (numerator + 1)**2

        grad = numerator * grad_output / denominator

        return grad

class threshold_slayer(torch.autograd.Function):
    """
    https://arxiv.org/pdf/1810.08646.pdf
    https://github.com/bamsumit/slayerPytorch/blob/master/src/slayer.py
    https://github.com/bamsumit/slayerPytorch/blob/master/example/02_NMNIST_MLP/network.yaml

    f'(x) = 1/alpha exp(-beta * |v - vth|)
    vth = 0
    alpha = 10
    beta = 1
    """

    @staticmethod
    def forward(ctx, input):
        """

        """
        ctx.save_for_backward(input)

        output = input.gt(1.0).float()

        return output

    @staticmethod
    def backward(ctx, grad_output):

        membranePotential, = ctx.saved_tensors

        spikePdf = 1/10 * torch.exp( -torch.abs(membranePotential - 1) / 10)

        return grad_output * spikePdf

class ActFun(torch.autograd.Function):
    """
    Adopted from https://github.com/yjwu17/BP-for-SpikingNN/blob/master/spiking_model.py
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(1.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - 1.0) < 0.5
        return grad_input * temp.float()

class threshold_diet(torch.autograd.Function):
    """
    DIET-SNN: A LOW-LATENCY SPIKING NEURAL NETWORK WITH DIRECT INPUT ENCODING & LEAKAGE AND THRESHOLD OPTIMIZATION
    https://arxiv.org/pdf/2008.03658.pdf
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(1.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        max(0, 1 - |v-1|)
        """
        input, = ctx.saved_tensors
        v_minus_1_abs = (input - 1).abs()
        tmp = torch.max(torch.tensor(0), 1 - v_minus_1_abs)

        grad_input = grad_output.clone()
        return grad_input * tmp

class threshold_fastsigmoid(torch.autograd.Function):
    """
    SuperSpike: Supervised learning in multi-layer spiking neural networks
    https://arxiv.org/pdf/1705.11146.pdf
    f(x) = x/(1+|x|)
    f'(x) = 1 / (|x| + 1)^2
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(1.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        f(x) = 1/(1+|x|)
        f'(x) = 1/ (|x-1|+1)^2
        """
        input, = ctx.saved_tensors
        denominator = (torch.abs(input-1) + 1)**2

        grad_input = grad_output.clone()
        return grad_input / denominator

class threshold_arctan(torch.autograd.Function):
    """
    Incorporating Learnable Membrane Time Constant To Enhance Learning of Spiking Neural Networks 
    https://openaccess.thecvf.com/content/ICCV2021/html/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.html
    f(x) = 1/pi * arctan(pi*(x)) + 1/2
    f'(x) = 1/ ((pi^2 * x^2) + 1)
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(1.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        input, = ctx.saved_tensors
        denominator = 9.8696 * (input-1)**2 + 1

        grad_input = grad_output.clone()
        return grad_input / denominator

def calculate_conv2d_outsize(h_input, w_input, padding, kernel_size, stride, dilation=1):
    '''
    calculate the output size of conv2d
    :param h_input:
    :param w_input:
    :param padding:
    :param dilation:
    :param kernel_size:
    :return:
    '''
    h_output = (h_input + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1
    w_output = (w_input + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1

    return h_output, w_output

def calculate_maxpooling2d_outsize(h_input, w_input, padding, kernel_size, stride, dilation=1):
    '''
    calculate the output size of conv2d
    :param h_input:
    :param w_input:
    :param padding:
    :param dilation:
    :param kernel_size:
    :return:
    '''
    h_output = (h_input + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1
    w_output = (w_input + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1

    return h_output, w_output

"""
def calculate_out_shape(h_input, w_input, kernel_h, kernel_w, stride, padding, dilation=1):
    '''
    calculate the output size of conv2d
    :param h_input:
    :param w_input:
    :param padding:
    :param dilation:
    :param kernel_size:
    :return:
    '''
    h_output = (h_input + 2*padding - dilation*(kernel_h-1) - 1)//stride + 1
    w_output = (w_input + 2*padding - dilation*(kernel_w-1) - 1)//stride + 1

    return h_output, w_output


def calculate_out_shape(input_h, input_w, layer_configs):
    '''
    layers_configs:a list of each layer kernel_size, stride, padding, dilation=1
    '''
    prev_h = input_h
    prev_w = input_w

    output_shapes = []

    for config in layer_configs:

        output_shapes.append([prev_h, prev_w])
        kernel_h, kernel_w, stride, padding, dilation = config
        h,w = calculate_out_shape(prev_h, prev_w, kernel_h, kernel_w, stride, padding, dilation)
        prev_h, prev_w = h, w
    
    return output_shapes
"""

def gain_bias(max_rates, intercepts, tau_m = 4, tau_ref = 0.002):
    """Analytically determine gain, bias.
        max_rate: numpy array
        intercept: numpy array
        nengo default value:
            reference: https://www.nengo.ai/nengo/frontend-api.html?highlight=ensemble#nengo.Ensemble
            max_rate: uniform (200,400)
            intercept: uniform (-1,0.9)
        Assume simulation run 1s, unit of rau_ref is also s
        Therefore max spike rate will be 1/0.002 = 500 hz
        To convert rate with spike probability, simply divide rate by 1000

        ref: https://www.nengo.ai/nengo/_modules/nengo/neurons.html#LIFRate.gain_bias
    """
    # tau_rc = np.exp(-1/tau_m)

    # in nengo, v =  v * exp(-0.001/tau_rc) + I * (1-exp(-0.0001/tau_rc))
    # in our model, v = v*(-1/tau_m) + I * (1-exp(-0.0001/tau_rc))
    # therefore, -1/tau_m = -0.001/tau_rc
    tau_rc = tau_m * 0.001

    # max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
    # intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)

    inv_tau_ref = 1.0 / tau_ref if tau_ref > 0 else np.inf
    if np.any(max_rates > inv_tau_ref):
        raise ValidationError(
            "Max rates must be below the inverse "
            "refractory period (%0.3f)" % inv_tau_ref,
            attr="max_rates",
            obj=self,
        )
    x = 1.0 / (1 - np.exp((tau_ref - (1.0 / max_rates)) / tau_rc))
    gain = (1 - x) / (intercepts - 1.0)
    bias = 1 - gain * intercepts
    return gain, bias

def max_rates_intercepts(gain, bias, tau_m = 4, tau_ref = 0.002):
    """Compute the inverse of gain_bias.

        ref: https://www.nengo.ai/nengo/_modules/nengo/neurons.html#LIFRate.gain_bias
    """

    tau_rc = np.exp(-1/tau_m)
    intercepts = (1 - bias) / gain
    max_rates = 1.0 / (
        tau_ref - tau_rc * np.log1p(1.0 / (gain * (intercepts - 1) - 1))
    )
    if not np.all(np.isfinite(max_rates)):
        warnings.warn(
            "Non-finite values detected in `max_rates`; this "
            "probably means that `gain` was too small."
        )
    return max_rates, intercepts

class SNN_Monitor():
    """
    Record spikes and states
    reference: https://www.kaggle.com/sironghuang/understanding-pytorch-hooks
    """
    def __init__(self, module, max_iteration = 1):

        self.step_num = module.step_num
        self.max_iteration = max_iteration

        self.variable_dict = {}
        self.record = {}

        self.counter  = 0
        self.max_len = max_iteration * self.step_num

        if isinstance(module, dual_exp_iir_layer):
            self.psp_list = []
            self.hook = module.dual_exp_iir_cell.register_forward_hook(self.get_output_dual_exp_iir)

            self.variable_dict['psp'] = self.psp_list

        elif isinstance(module, first_order_low_pass_layer):
            self.psp_list = []
            self.hook = module.first_order_low_pass_cell.register_forward_hook(self.get_output_first_order_low_pass)

            self.variable_dict['psp'] = self.psp_list

        elif isinstance(module, neuron_layer):
            self.hook = module.neuron_cell.register_forward_hook(self.get_output_neuron_layer)
            self.v_list = []
            self.reset_v_list = []
            self.spike_list = []

            self.variable_dict['v'] = self.v_list
            self.variable_dict['reset_v'] = self.reset_v_list
            self.variable_dict['spike'] = self.spike_list

        elif isinstance(module, axon_layer):
            self.hook =module.axon_cell.register_forward_hook(self.get_output_axon_layer)
            self.psp_list = []

            self.variable_dict['psp'] = self.psp_list

    def get_output_dual_exp_iir(self, module, input, output):
        '''

        :param module:
        :param input: a tuple [spike, new state[state(t-1), state(t-2)]]
        :param output: a tuple [psp, new state[psp, state(t-1)]]
        :return:
        '''

        self.counter += 1
        if self.counter > self.max_len:
            return

        self.psp_list.append(output[0])

        if self.counter == self.max_len:
            self.reshape()

    def get_output_first_order_low_pass(self, module, input, output):
        '''

        :param module:
        :param input:
        :param output:
        :return:
        '''

        self.counter += 1
        if self.counter > self.max_len:
            return

        self.psp_list.append(output[0])

        if self.counter == self.max_len:
            self.reshape()

    def get_output_neuron_layer(self, module, input, output):
        '''

        :param module:
        :param input:
        :param output: [spike, [v, reset_v]]
        :return:
        '''

        self.counter += 1
        if self.counter > self.max_len:
            return

        self.spike_list.append(output[0])
        self.v_list.append(output[1][0])
        self.reset_v_list.append(output[1][1])

        if self.counter == self.max_len:
            self.reshape()

    def get_output_axon_layer(self, module, input, output):
        '''

        :param module:
        :param input:
        :param output:
        :return:
        '''

        self.counter += 1
        if self.counter > self.max_len:
            return

        self.psp_list.append(output[0])

        if self.counter == self.max_len:
            self.reshape()

    def reshape(self):

        for key in self.variable_dict:
            temp_list = []
            for element in self.variable_dict[key]:
                temp_list.append(element.detach().cpu().numpy())

            # shape packed [total steps, batch, neuron/synapse]
            packed = np.array(temp_list)

            #shape packed [iterations,step num, batch, neuron/synapse]
            packed = np.reshape(packed, (self.max_iteration, self.step_num, *packed.shape[1:]))
            #shape packed [iterations, batch, step num, neuron/synapse]
            packed = packed.swapaxes(1,2)
            # #shape packed [iterations, batch, neuron/synapse, step num]
            packed = packed.swapaxes(2,3)
            self.record[key] = packed

class voltage_monitor():
    """
    Record spikes and states
    reference: https://www.kaggle.com/sironghuang/understanding-pytorch-hooks
    """
    def __init__(self, module):

        self.step_num = module.step_num

        self.voltage_list = []

        self.enable = True

        if isinstance(module, conv2d_layer):
            self.hook = module.conv2d_cell.register_forward_hook(self.get_voltage_conv2d_layer)
            self.layer_type = 'conv2d'

        elif isinstance(module, neuron_layer):
            self.hook = module.neuron_cell.register_forward_hook(self.get_voltage_neuron_layer)
            self.layer_type = 'dense'

    
    def get_voltage_neuron_layer(self, module, input, output):
        '''
        input: [input spike, [previous v, previous reset]]
        output: [output spike, [current v, current reset]]
        '''

        if self.enable == True:
            self.voltage_list.append(output[1][0])
    
    def get_voltage_conv2d_layer(self, module, input, output):
        '''
        '''

        if self.enable == True:
            self.voltage_list.append(output[1][0])

    def reset(self):

        self.voltage_list = []
    
    def disable(self):

        self.enable = False

    def get_record(self):
        
        if self.layer_type == 'dense':
            temp_list = []
            for item in self.voltage_list:
                temp_list.append(item.detach().cpu().numpy())

            # shape packed [steps, batch, neuron]
            packed = np.array(temp_list)

            # shape packed [batch, steps, neuron]
            packed = packed.swapaxes(0,1)

            # shape packed [batch, neuron, steps]
            packed = packed.swapaxes(1,2)

            return packed


