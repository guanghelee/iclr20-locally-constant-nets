import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def my_softplus(x, tau=1., threshold=20.):
    truncate_mask = (x > threshold).type(torch.cuda.FloatTensor)
    return truncate_mask * x + (1. - truncate_mask) * (tau * torch.log(1 + torch.exp((1. - truncate_mask) * x / tau)))

def my_softplus_derivative(x, tau=1., threshold=20.):
    truncate_mask = (x > threshold).type(torch.cuda.FloatTensor)
    return truncate_mask + (1. - truncate_mask) / (1 + torch.exp(- (1. - truncate_mask) * x / tau) )

class Net(nn.Module):
    def __init__(self, \
            input_dim, \
            output_dim, \
            hidden_dim, \
            num_layer, \
            num_back_layer, \
            dense = False, \
            drop_type = 'none', \
            net_type = 'locally_constant', \
            approx = 'none'):
        super(Net, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.num_back_layer = num_back_layer
        self.dense = dense
        self.drop_type = drop_type
        self.net_type = net_type
        self.n_neuron = self.hidden_dim * self.num_layer
        self.approx = approx
        
        self.layer = nn.ModuleList()

        self.weights = dict()
        self.biases = dict()
        self.output_constants = dict()
        self.output_weights = dict()
        self.output_biases = dict()

        accu_dim = input_dim
        self.weight_masks = []
        for i in range(self.num_layer):
            self.layer.append(nn.Linear(accu_dim, hidden_dim))
            ite_mask = np.zeros((1, hidden_dim, accu_dim))
            pick = np.random.choice(accu_dim, int(np.sqrt(accu_dim)), replace=False)
            ite_mask[0, 0, pick] = 1
            assert(hidden_dim == 1) 
            ite_mask = torch.tensor(ite_mask.astype(np.float32)).cuda()
            self.weight_masks.append(ite_mask)

            if self.dense:
                accu_dim += hidden_dim
            else:
                accu_dim = hidden_dim

        if self.net_type == 'locally_constant':
            self.backward_layer = nn.ModuleList()
            backward_dim = 256
            cur_dim = self.num_layer * self.hidden_dim * (1 + self.input_dim)
            for i in range(self.num_back_layer):
                self.backward_layer.append(nn.Linear(cur_dim, backward_dim))
                cur_dim = backward_dim
            self.backward_layer.append(nn.Linear(cur_dim, output_dim))

        elif self.net_type == 'locally_linear':
            self.output_fc = nn.Linear(accu_dim, self.output_dim)
        else:
            print('net_type', self.net_type, 'is not supported')
            exit(0)

    def normal_forward(self, init_layer, p=0, training=True):
        assert(self.net_type == 'locally_linear')
        relu_masks = []
        if len(init_layer.shape) == 4:
            bz = init_layer.shape[0]
            init_layer = init_layer.view(bz, -1)
        cur_embed = init_layer
        batch_size = init_layer.shape[0]

        for i in range(self.num_layer):
            if training == False:
                next_embed = self.layer[i](cur_embed)
            else:
                w = self.layer[i].weight
                w = w.view(1, w.shape[0], w.shape[1]).expand(batch_size, -1, -1)

                b = self.layer[i].bias
                b = b.view(1, b.shape[0]).expand(batch_size, -1)

                next_embed = torch.bmm(F.dropout(w, p=p, training=training), cur_embed.unsqueeze(-1)) + b.unsqueeze(-1)
                next_embed = next_embed.squeeze(-1)

            relu_masks.append( (next_embed > 0) )
            next_embed = F.relu(next_embed)
            if self.dense:
                cur_embed = torch.cat((cur_embed, next_embed), 1)
            else:
                cur_embed = next_embed
        return self.output_fc(cur_embed), relu_masks

    def forward(self, init_layer, p=0, training=True, alpha=None, anneal='none'):
        assert(self.net_type == 'locally_constant')
        relu_masks = []
        if len(init_layer.shape) == 4:
            bz = init_layer.shape[0]
            init_layer = init_layer.view(bz, -1)
        # cur_embed is used for forward computation.
        cur_embed = init_layer
        # x is used to compute Jacobian using dynamic programming.
        x = init_layer.unsqueeze(-1)
        
        batch_size = x.shape[0]
        patterns = []
        for i in range(self.num_layer):
            if self.drop_type != 'node_dropconnect':
                next_embed = self.layer[i](cur_embed)
            w = self.layer[i].weight
            w = w.view(1, w.shape[0], w.shape[1]).expand(batch_size, -1, -1)   

            if self.drop_type == 'node_dropconnect':
                b = self.layer[i].bias
                b = b.view(1, b.shape[0]).expand(batch_size, -1)

                next_embed = torch.bmm(F.dropout(w, p=p, training=training), cur_embed.unsqueeze(-1)) + b.unsqueeze(-1)
                next_embed = next_embed.squeeze(-1)
            else:
                pass

            relu_masks.append( (next_embed > 0) )
            if self.approx == 'approx':
                neur_deriv = my_softplus_derivative(next_embed)
                next_embed = my_softplus(next_embed)
            elif anneal == 'interpolation':
                neur_deriv = alpha * (next_embed > 0).type(torch.cuda.FloatTensor) + (1 - alpha) * my_softplus_derivative(next_embed)
                next_embed = alpha * F.relu(next_embed) + (1 - alpha) * my_softplus(next_embed)
            elif anneal == 'none':
                neur_deriv = (next_embed > 0).type(torch.cuda.FloatTensor)
                next_embed = F.relu(next_embed)

            patterns.append(neur_deriv)
            neur_deriv = neur_deriv.unsqueeze(-1)
             
            if i == 0:
                jacobians = neur_deriv * w
                offsets = next_embed - torch.bmm(jacobians, x).squeeze(-1)
            else:
                if self.dense:
                    ite_jacobians = w[:, :, :self.input_dim] + torch.bmm(w[:, :, self.input_dim:], jacobians)
                else:
                    ite_jacobians = torch.bmm(w, jacobians[:, -self.hidden_dim:, :])

                ite_jacobians = neur_deriv * ite_jacobians
                ite_offsets = next_embed - torch.bmm(ite_jacobians, x).squeeze(-1)
                
                jacobians = torch.cat([jacobians, ite_jacobians], dim=1)
                offsets = torch.cat([offsets, ite_offsets], dim=1)

            if self.dense:
                cur_embed = torch.cat((cur_embed, next_embed), 1)
            else:
                cur_embed = next_embed

        leaf_input = torch.cat([jacobians, offsets.unsqueeze(-1)], dim=2).view(batch_size, -1)    
        for i in range(self.num_back_layer):
            leaf_input = F.relu(self.backward_layer[i](leaf_input))
        leaf_output = self.backward_layer[-1](leaf_input)
        
        return leaf_output, relu_masks

        
