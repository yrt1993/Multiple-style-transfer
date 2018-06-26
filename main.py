import numpy as np

from StyleCNN import StyleCNN
import utils


from matplotlib.pylab import subplots
import matplotlib.pylab as mpylab


def produce_pastiche(num_epochs, style_cnn, name, saved_images):
    '''
    Produce pastiche according to corresponding configuration encoded in name. At first, we reset StyleCNN 
    neural network according to configuration information stored in name. Then train the neural network for 
    num_epochs epochs. In the process, we save pastiche produced after specific iterations, which is indicated 
    in saved_images. These images have name (name + '-i%d.png' %(iteration in saved_images)), like 
    'pn-c1s1s4-a1b100w1-cl2_2sl1_11_22_12_23_1-oLBFGS-i10.png'. Besides, we also save loss vs iteration image, 
    which has name ('loss_' + name + '.png'), like 'loss_pn-c1s1s4-a1b100w1-cl2_2sl1_11_22_12_23_1-oLBFGS.png'.
    
    :param num_epochs: num of epochs we train the neural network
    :type num_epochs: int
    :param style_cnn: the StyleCNN neural network
    :type style_cnn: StyleCNN
    :param name: a string containing all the configuration information of StyleCNN
    :type name: str
    :param saved_images: a list of iterations after which we save produced pastiche
    :type saved_images: list
    :return none
    
    '''
    # reset StyleCNN 
    style_cnn.reset(name)
    
    
    content_losses = []
    style_losses = []
    total_losses = []
    
    for i in range(num_epochs):
        # train 
        pastiche = style_cnn.train()
        
        # record losses
        content_losses.append(style_cnn.content_loss)
        style_losses.append(style_cnn.style_losses)
        total_losses.append(style_cnn.total_loss) 
            
        # save pastiche 
        if i in saved_images:
            print("Iteration: %d" % (i))
            path = './outputs/'+ name + '-i%d.png' % (i)
            pastiche.data.clamp_(0, 1)
            utils.save_image(pastiche, path)
    
    content_losses = np.array(content_losses)
    style_losses = np.array(style_losses)
    total_losses = np.array(total_losses)
    
    # plot and saveloss vs epochs curve
    epoches = [i+1 for i in range(num_epochs)]
    fig, axes = subplots()
    axes.plot(epoches, np.log(total_losses), label = 'total loss')
    axes.plot(epoches, np.log(style_cnn.content_weight * content_losses), label = 'content loss')
    for i in range(style_losses.shape[1]):
        axes.plot(epoches, np.log(style_cnn.style_weight * style_losses[:, i]), label = 'style loss%d' %(i + 1))
    
    axes.set_xlabel('epoch')
    axes.set_ylabel('loss')
    axes.legend(loc='best')
    
    fig.show()
    mpylab.savefig('./outputs/loss_' + name + '.png', bbox_inches='tight')
    
    return 

