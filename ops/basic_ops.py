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
            output = vnet(input_tensor)
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1, vnet=None):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim
        self.vnet = vnet

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim, self.vnet)(input)
