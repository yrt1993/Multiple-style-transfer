import torch
import torch.nn as nn

class GramMatrix(nn.Module):
    '''
    calculate the feature correlations of some layer, i.e. Gram matrix.
    Input is of this format: batch size(1 for this project) * #channnels * H * W.
    So N = 1 * #channels, M = H * W. Each element of Gram Matrix is a sum of M products, 
    so it can be normalized by getting it divided by M.
    
    :param input: a tensor stores all the feature maps of a layer of vgg19, of size:
    batch size(1 for this project) * #channnels * H * W
    :type input: torch.Tensor
    :return normalized Gram matrix, i.e. a tensor of size N * N, N = 1 * #channels. 
    :type: torch.Tensor
    '''
    
    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        
        return G.div(c * d)

