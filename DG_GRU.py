import torch.nn as nn
from torch.autograd import Variable
import torch
from torch_geometric.nn import GCNConv, GATConv, GraphConv
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from algorithm_utils import Algorithm, PyTorchUtils


class GATGRUCell(nn.Module, PyTorchUtils):

    def __init__(self, nodes_num, input_dim, hidden_dim, head=1, dropout=0, bias=True, seed: int=0, gpu: int=None):
        super(GATGRUCell, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        self.nodes_num = nodes_num
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.head = head
        self.dropout = dropout
        self.bias = bias

        self.gconv = GATConv(in_channels=self.input_dim + self.hidden_dim,
                             out_channels=3 * self.hidden_dim,
                             heads=self.head,
                             concat=False,
                             dropout=self.dropout,
                             bias=self.bias)

    def forward(self, input_tensor, cur_state, edge_index):
        h_cur = cur_state[0]

        #    h_cur = torch.stack(h_cur)
        combined = torch.cat([input_tensor, h_cur], dim=2)  #将input和hidden在最后一维拼接
        batch = Batch.from_data_list([Data(x=combined[i], edge_index=edge_index) for i in range(combined.shape[0])])

        combined_conv = self.gconv(batch.x, batch.edge_index)
        combined_conv = combined_conv.reshape(combined.shape[0],combined.shape[1],-1)

        cc_r, cc_z, cc_n = torch.split(combined_conv, self.hidden_dim, dim=2)
        r = torch.sigmoid(cc_r)
        z = torch.sigmoid(cc_z)
        n = torch.tanh(cc_n)

        h_next = (1 - z) * n + z * h_cur

        return h_next

    def init_hidden(self, batch_size):
        return self.to_var(Variable(torch.zeros(batch_size, self.nodes_num, self.hidden_dim)))
class GraphGRU(nn.Module, PyTorchUtils):

    def __init__(self, nodes_num, input_dim, hidden_dim, num_layers, head=1, dropout=0, kind='GAT',
                 batch_first=False, bias=True, return_all_layers=True, seed: int=0, gpu: int=None):
        super(GraphGRU, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        head = self._extend_for_multilayer(head, num_layers)
        if not len(hidden_dim) == len(head) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.nodes_num = nodes_num
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.head = head
        self.dropout = dropout
        self.kind = kind
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            if self.kind == 'GAT':
                cell_list.append(GATGRUCell(nodes_num=nodes_num,
                                            input_dim=cur_input_dim,
                                            hidden_dim=self.hidden_dim[i],
                                            head=self.head[i],
                                            dropout=self.dropout,
                                            bias=self.bias,
                                            seed=self.seed,
                                            gpu=self.gpu))
            else:
                raise NotImplementedError()

        self.cell_list = nn.ModuleList(cell_list)


    def forward(self, input_tensor, edge_index, hidden_state=None):
        if hidden_state is not None:
            hidden_state = hidden_state
        else:
            hidden_state = self._init_hidden(input_tensor.size(
                1))
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(0)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx]
            if isinstance(h, list):
                h = h[0]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[t],
                                                 edge_index=edge_index, cur_state=[h])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list
    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param