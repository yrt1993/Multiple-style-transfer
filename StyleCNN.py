import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import GramMatrix as gm
import numpy as np
import utils


class StyleCNN(object):
    '''
    our neural network for style transfer
    '''
    def __init__(self, name):
        super(StyleCNN, self).__init__()
        # CUDA configuration
        self.use_cuda = torch.cuda.is_available()
        
        self.reset(name)
      
        self.loss_network = models.vgg19(pretrained=True)
        self.gram = gm.GramMatrix()
        self.loss = nn.MSELoss()
        
        if self.use_cuda:
            self.loss_network.cuda()
            self.gram.cuda()
            self.loss.cuda()
    
    def reset(self, name):
        '''
        reset the StyleCNN network using the name, which is a string containing all the configuration 
        information of StyleCNN. name is like 'pc-c1s1-a0b1w1-cl4_2sl1_12_13_14_15_1-oLBFGS' and can be 
        divided to 5 parts separated by '-'.
        1. first part, which is 'pc' or 'pn', records input image. 'pc' means input image is the same 
        as content image. 'pn' means input image is random noise.
        2. second part, like 'c1s1' or 'c1s1s2', records the content image and style images. 
        'c1' means content1.jpg, s1 means style1.jpg, etc.
        3. third part, like 'a1b1000w1', records weight information. 'a' means content weight, in general, 
        we choose it to be 0 or 1. 'b' means style weight, if <1, use 0_01 to represent 0.01. 'w' means in-
        styles weight. The number following 'w' is the type of in-styles weight we choose. 1 means average 
        distribution. 2 means weight of style i = loss of style i / total style loss
        4. fourth part, like 'cl4_2sl1_12_13_14_15_1', records content and style layers used to compute loss. 
        'cl' means content layer, '4_2' means conv_4_2. 'sl' means style layer, '1_12_13_14_15_1' means using 
        ['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_4_1', 'conv_5_1'].
        5. fifth part, like 'oLBFGS', records optimizer information. 'o' means optimizer. 'LBFGS' means 
        optim.lBFGS optimizer. We only use optim.SGD, optim.Adam and optim.LBFGS. 
             
        :param name: a string containing all the configuration information of StyleCNN
        :type name: str
        :return none
        
        '''
        # CUDA Configurations
        dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        
        ## decode name and reset style_cnn
        keys = name.split('-')
        
        # reset content image and style images
        # self.content is a tensor, 1 * 3 * H * W
        # self.styles is a tensor, #styles * 3 * H * W
        images = keys[1].split('s')
        self.content = utils.image_loader('./contents/content' + images[0][1:] + '.jpg').type(dtype)
        self.styles = torch.Tensor(0)
        for i in range(1, len(images)):
            style = utils.image_loader('./styles/style' + images[i] + '.jpg')
            self.styles = torch.cat((self.styles, style), 0)
        self.styles = self.styles.type(dtype) 
        
        # reset num of styles and in_style_weights
        self.num_styles = self.styles.size()[0]
            
        # reset pastiche
        # back propagate to update pixel values of pastiche 
        # self.pastiche is Tensor here
        if keys[0] == 'pn':
            self.pastiche = nn.Parameter(torch.randn(self.content.size()).type(dtype))
        else:
            self.pastiche = nn.Parameter(self.content.clone())

        # reset content weight and style weight
        index_b = keys[2].find('b')
        index_w = keys[2].find('w')
        self.content_weight = float(keys[2][1:index_b])
        self.style_weight = float(keys[2][index_b + 1:index_w].replace('_', '.'))
        self.in_styles_weight = keys[2][index_w + 1:]

        # reset content layers and style layers
        self.content_layers = ['conv_' + keys[3][2:5]]
        num_style_layers = (len(keys[3]) - 7) // 3
        self.style_layers = []
        for i in range(num_style_layers):
            self.style_layers.append('conv_' + keys[3][7 + 3 * i:10 + 3 * i])
        
        # reset optimizer
        if keys[4][1:] == 'SGD':
            self.optimizer = optim.SGD([self.pastiche], 
                                       lr=0.001 ,
                                       momentum=0.9)
        elif keys[4][1:] == 'Adam':
            self.optimizer = optim.Adam([self.pastiche])
        else:
            self.optimizer = optim.LBFGS([self.pastiche])
        
        # reset losses
        '''
        content_loss, style_loss, total_loss records the corresponding loss 
        after each iteration. They are all tensors
        '''
        self.total_loss = 0
        self.content_loss = 0
        self.style_losses = 0
        return

    def train(self):
        '''
        .train() defines an iteration of input image towards pastiche
        
        '''
        def closure():
            self.optimizer.zero_grad()

            pastiche = self.pastiche.clone()
            # before every iteration, clamp the pixel value of pastiche in [0,1]
            pastiche.data.clamp_(0, 1)
            content = self.content.clone()
            styles = self.styles.clone()
            
            content_loss = 0
            style_losses = torch.zeros(self.num_styles, 1)
            if self.use_cuda:
                style_losses = style_losses.cuda()
            
            # i, j are index of layers. Conv(i)_(j)
            i = 1
            j = 1
            not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer

            # loss calculation in a feed forward manner
            for layer in list(self.loss_network.features):
                layer = not_inplace(layer)
                if self.use_cuda:
                    layer.cuda()

                pastiche, content, styles = layer.forward(pastiche), layer.forward(content), layer.forward(styles)

                if isinstance(layer, nn.Conv2d):
                    name = "conv_" + str(i) + '_' + str(j)

                    if name in self.content_layers:
                        '''
                        user nn.MSELoss require the two input have same size. so pastiche.size() == content.size()

                        '''
                        content_loss += self.loss(pastiche, content.detach())

                    if name in self.style_layers:
                        '''
                        size of gram matrix is #channels * #channels, so pastiche and style can have different size 
                        normalized in gram matrix so no need to change the size of style image
                        '''
                        pastiche_g = self.gram.forward(pastiche)
                        for k in range(self.num_styles):
                            style_g = (self.gram.forward(styles[k:k+1]))
                            style_losses[k] += self.loss(pastiche_g, style_g.detach())

                if isinstance(layer, nn.ReLU):
                    j += 1
                if isinstance(layer, nn.MaxPool2d):
                    i += 1
                    j = 1
                 

            
            style_losses = style_losses.view(self.num_styles) / len(self.style_layers)
            if self.in_styles_weight == '1':
                if self.use_cuda:
                    in_styles_weight = torch.ones(self.num_styles).cuda()
                else:
                    in_styles_weight = torch.ones(self.num_styles)   
            else:
                in_styles_weight = style_losses / torch.sum(style_losses)

            style_losses = style_losses * in_styles_weight
            total_loss = self.content_weight * content_loss + self.style_weight * torch.sum(style_losses)
            
            # update self.content_loss, self.style_losses, self.total_loss
            if total_loss.is_cuda:
                self.content_loss = content_loss.cpu().item()
                self.style_losses = style_losses.detach().cpu().numpy()
                self.total_loss = total_loss.cpu().item()
            else:
                self.content_loss = content_loss.item()
                self.style_losses = style_losses.detach().numpy()
                self.total_loss = total_loss.item()
                
                
            # back propagation
            total_loss.backward()
            return total_loss
    
   
        self.optimizer.step(closure)
        return self.pastiche

