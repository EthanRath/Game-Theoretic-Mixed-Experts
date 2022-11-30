import torch
from Defenses.SNN.snn_lib import *               # different snn models

#Originally called the "mysnn" class
class SNNBackprop(torch.nn.Module):
    global cfg
    #Configurations for the vgg model
    cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def __init__(self, length, batch_size, tau_m, tau_s, train_synapse_tau, train_neuron_tau, membrane_filter, neuron_model, synapse_type, dropout):

        super().__init__()

        global cfg
        self.length = length
        self.batch_size = batch_size
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.train_synapse_tau = train_synapse_tau
        self.train_neuron_tau = train_neuron_tau
        self.membrane_filter = membrane_filter
        self.neuron_model = neuron_model
        self.synapse_type = synapse_type
        self.dropout = dropout
        
        '''
        layer_configs = [
            # kernel_h, kernel_w, stride, padding, dilation
            [3,3,1,1,1], # conv
            [3,3,1,1,1], # conv
            [2,2,2,0,1], # pooling
            [3,3,1,1,1], # conv
            [3,3,1,1,1], # conv
            [2,2,2,0,1], # pooling
            [3,3,1,1,1], # conv
            [3,3,1,1,1], # conv
            [3,3,1,1,1], # conv
            [2,2,2,0,1], # pooling
            [3,3,1,1,1], # conv
            [3,3,1,1,1], # conv
            [3,3,1,1,1], # conv
            [2,2,2,0,1], # pooling
            [3,3,1,1,1], # conv
            [3,3,1,1,1], # conv
            [3,3,1,1,1], # conv
            [2,2,2,0,1], # pooling
        ]
        '''


        self.layers = torch.nn.ModuleList([])

        if self.neuron_model == 'iir':
            
            synapse_config = {'synapse_type' : self.synapse_type,
                  'synapse_tau_m'       : self.tau_m,
                  'synapse_tau_s'       : self.tau_s,
                  'train_synapse_tau'   : self.train_synapse_tau}

            c,h,w = self.make_layers(cfg['VGG16'], synapse_config)

            # output shape of conv2d_layer [batch, channel, hight, width, length]
            # flatten from 1st dimension to 3rd dimension, get [batch, channel*hight*width, length]  
            self.layers.append(torch.nn.Flatten(1, 3))

            self.layers.append(torch.nn.Dropout(p=self.dropout))

            self.layers.append(neuron_layer_bntt(c*h*w, 512, self.length, self.batch_size, 
                                                self.tau_m, membrane_filter = self.membrane_filter, 
                                                return_state=False, train_decay_v=self.train_neuron_tau, **synapse_config))

            self.layers.append(torch.nn.Dropout(p=self.dropout))

            self.layers.append(neuron_layer_bntt(512, 512, self.length, self.batch_size, 
                                                self.tau_m, membrane_filter = self.membrane_filter, 
                                                return_state=False, train_decay_v=self.train_neuron_tau, **synapse_config))

            self.layers.append(torch.nn.Dropout(p=self.dropout))

            self.layers.append(neuron_layer_bntt(512, 10, self.length, self.batch_size, 
                                                self.tau_m, membrane_filter = self.membrane_filter, 
                                                return_state=False, train_decay_v=self.train_neuron_tau, **synapse_config))
            
        elif neuron_model == 'stbp':

            raise Exception('stbp not implemented')

        else:
            raise Exception('wrong model')
        
        self.output_filter = first_order_low_pass_layer((10,), self.length, self.batch_size, self.tau_s, False)
        self.output_filter.first_order_low_pass_cell.alpha_1.requires_grad = False

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """

        x = inputs
        for layer in self.layers:
            x = layer(x)

        # shape out filtered_output: [batch, 10, length]
        filtered_output,_ = self.output_filter(x)

        # x is output spike train, torch.sum(x, dim=2) is spike count, filtered_output[-1] is filter's output at last time step
        # shape: x: [batch, 10, length], torch.sum(x, dim=2): [batch, 10], filtered_output[...,-1]: [batch, 10]
        #Change by K to make this behave like a CNN and just give output probabilites
        #return x, torch.sum(x, dim=2), filtered_output[...,-1], torch.sum(filtered_output, dim=2)
        return torch.sum(x, dim=2)
    
    def allow_train_filter_coefficients(self, option:bool):

        self.conv1_filter.allow_train_filter_coefficients(option)
        self.conv2_filter.allow_train_filter_coefficients(option)
        self.conv3_filter.allow_train_filter_coefficients(option)
        self.dense1_filter.allow_train_filter_coefficients(option)
        self.dense2_filter.allow_train_filter_coefficients(option)
    
    def switch_threshold(self, enable=True):

        self.conv1.conv2d_cell.enable_threshold = enable
        self.conv2.conv2d_cell.enable_threshold = enable
        self.conv3.conv2d_cell.enable_threshold = enable
        self.conv4.conv2d_cell.enable_threshold = enable

        self.dense1.neuron_cell.enable_threshold = enable
        self.dense2.neuron_cell.enable_threshold = enable
    
    def make_layers(self, configs, synapse_config):
        global cfg #get access to the VGG config variable

        #Hardcoding seems bad, can we fix? -KM
        prev_c, prev_h, prev_w = 3, 32, 32

        for cfg in configs:
            if 'M' == cfg:
                self.layers.append(maxpooling2d_layer(prev_h, prev_w, prev_c, 2, 2, 0, 1, self.length, self.batch_size))
                prev_h, prev_w = calculate_maxpooling2d_outsize(prev_h, prev_w, 0, 2, 2)
            else:
                self.layers.append(conv2d_layer_bntt(h_input=prev_h, w_input=prev_w, in_channels=prev_c, out_channels=cfg, kernel_size=3, stride=1, padding=1, dilation=1, 
                                    step_num=self.length, batch_size=self.batch_size, tau_m=self.tau_m, membrane_filter = self.membrane_filter, **synapse_config)) 
                self.layers[-1].return_state = False
                prev_c, prev_h, prev_w = self.layers[-1].output_shape

        return prev_c, prev_h, prev_w


