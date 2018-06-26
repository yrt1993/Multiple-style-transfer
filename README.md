# Multiple-style-transfer

In this project, I build an artificial system based on Convolutional Neural Network(CNN) to create artistic images of high quality. Such new images, called **pastiche**, are created through transferring the styles of several artworks, known as the  **style image** to a picture, known as the **content image**.

In the system, an input image (either random noise or same as content image) is passed through the pre-trained VGG19 network along with the content image and style images. The feature maps of intermediate layers are used to compute the **content loss**, which measures the difference of content of iput image and content image, and the **style loss**, which measures the difference of styles of input image and style image. Then back propagate to the input image to minimize a total loss consisting of weighted content loss and style loss. Therefore, with each iteration, the input image becomes 
more like the desired pastiche, combing content from content image with styles from several artworks. 

This project is based on [Gaty et al's paper](https://arxiv.org/abs/1508.06576) and act as a generalization.


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
