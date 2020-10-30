import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.nn.Module):

    def __init__(self, consensus_type, dim=1, vnet=None):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None
        self.vnet = vnet

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        elif self.consensus_type == 'vnet':
            num_class = self.shape[2]
            input_var = torch.autograd.Variable(input_tensor)
            input_var = torch.reshape(input_var, (-1, num_class))
            seg_weight = self.vnet(input_var)
            output = input_var * seg_weight
            output = torch.reshape(output, self.shape)
            output = output.sum(dim=self.dim, keepdim=True)      
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input, vnet=None):
        if self.consensus_type == "vnet" and not vnet:
            raise ValueError("Need to pass vnet for Consensus")
        return SegmentConsensus(self.consensus_type, self.dim, vnet)(input)
