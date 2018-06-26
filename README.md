# Multiple-style-transfer

In this project, an artificial system based on Convolutional Neural Network(CNN) is built to transfer the styles of several artworks to a picture. Desired image is called **pastiche**.

This project is based on [Gatys et al's paper](https://arxiv.org/abs/1508.06576), which transfer only one style and act as an generalization.

# Implementation
In the system, an input image (either random noise or same as content image) is passed through the pre-trained VGG19 network along with the content image and style images. The feature maps of intermediate layers are used to compute the **content loss**, which measures the difference of content of iput image and content image, and the **style loss**, which measures the difference of styles of input image and style image. Then back propagate to the input image to minimize a total loss consisting of weighted content loss and style loss. Therefore, with each iteration, the input image becomes 
more like the desired pastiche, combing content from content image with styles from several artworks. 



# package used 
- python2
- pytorch
- cuda

Actually, codes are run on AWS conda-pytorch-p27.

# how to run the code
The three jupyter notebooks shows how to run the code and some examples. 

1. reconstruction.ipynb shows content and style reconstruction under different conditions, including different 
input images and different style and content layers.

2. transfer1style.ipynb shows transferring one style to content image under different conditions, including different 
input images and different style and content layers.

3. transfer2styles.ipynb shows several examples of transferring two styles to content image.

# code reference
The codes refers to [Neural Artistic Style Transfer: A Comprehensive Look](https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199). 

I fix some errors and make many modifications. Besides, it is generalized to transfer several styles.
