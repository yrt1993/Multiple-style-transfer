import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
from matplotlib.pylab import subplots
import matplotlib.pyplot as plt

imsize = 256

'''
do two transforms on an PIL image. 
1. resize the image to be imsize * imsize
2. transform the PIL image to a tensor with pixel value in [0,1], of size 
   3 * H * W
'''
loader = transforms.Compose([
             transforms.Resize((imsize, imsize)),
             transforms.ToTensor()
         ])

# transform a torch.FloatTensor of size #channels * H * W to PIL image
unloader = transforms.ToPILImage()

def image_loader(image_name):
    '''
    read image from path stored in image_name and transfer it to a tensor of size 1 * 3 * imsize * imsize. 
    Besides, make pixel value be in [0,1]
    
    :param image_name: path where we read image from
    :type image_name: str
    :return a image represented by a tensor
    :type return: torch.FloatTensor
    
    '''
    
    image = Image.open(image_name)
    image = loader(image)
    # add batch size 1
    image = image.unsqueeze(0)
    return image
  
def save_image(input, path):
    '''
    save image stored in input to path.
    
    :param input: a tensor, of size 1 * 3 * H * W, represent an image
    :type input: torch.cuda.FloatTensor
    :param path: the path to save the image
    :type path: str
    :return none
    '''
    image = input.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    image.save(path)
    
    
def show_images(name, shows):
    '''
    show content image, style images and produced pastiches. 
    
    
    :param name: a string containing all the configuration information of StyleCNN, 
    like 'pn-c1s1s4-a1b100w1-cl2_2sl1_11_22_12_23_1-oLBFGS'
    :type name: str
    :param shows: a list of iterations for which we show corresponding pastiches, like 
    [10, 20, 30, 40, 50, 60]
    :type shows: list
    :return none
    
    '''
    ## decode name
    keys = name.split('-')
    images = keys[1].split('s')
    
    num_cols = (len(images) + len(shows) + 1) / 2
    fig = plt.figure()
    
    # set image size
    fig.set_figheight(5 * num_cols)
    fig.set_figwidth(15)
    
    # plot content image
    a = fig.add_subplot(num_cols, 2, 1)
    plt.imshow(Image.open('./contents/content'+ images[0][1:] + '.jpg'))
    a.set_title('content')
    
    # plot style images
    for i in range(1, len(images)):   
        a = fig.add_subplot(num_cols, 2, 1 + i)
        plt.imshow(Image.open('./styles/style'+ images[i] + '.jpg'))
        a.set_title('style%d' %(i))
    
    # plot specified pastiches
    for i in range(len(shows)):   
        a = fig.add_subplot(num_cols, 2, 1 + len(images) + i)
        plt.imshow(Image.open('./outputs/' + name + '-i' + str(shows[i]) + '.png'))
        a.set_title('pastiche %d' %(shows[i]))
        
    return

