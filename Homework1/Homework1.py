'''
Created on Oct 6, 2015

@author: Keith Mikoleit
'''

import cv2
import numpy
from random import randint


def findConnectedComponents(image_name, open_ellipse_size, close_ellipse_size):
    """ Find and display the connected components of an image.
    
    Args:
        image name -- name of image file, must be in same directory as python script
        open_ellipse_size -- the size of the mask used to perform the open morphology
        close_ellipse_size -- the size of the mask used to perform the close morphology
        
    """
    
    #get original image and display
    im_orig = cv2.imread(image_name, cv2.IMREAD_COLOR)
    
    cv2.imshow(image_name + ": Original", im_orig)
    
    # cv2.waitKey(0)
    
    #threshold the image to create a binary image
    threshold = 128
    maxval = 255
    _, im_th = cv2.threshold(im_orig,threshold, maxval, cv2.THRESH_BINARY)
    
    #convert image to gray scale type
    im_gray = cv2.cvtColor(im_th, cv2.COLOR_RGB2GRAY)
    
    cv2.imshow(image_name + ": Threshold", im_gray)
    
    # cv2.waitKey(0)
    
    # create a circular mask and use it to morph the image via opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_ellipse_size)
    im_open = cv2.morphologyEx(im_gray, cv2.MORPH_OPEN, kernel_open)
      
    # create a smaller mask and use it to morph the image via closing to get rid of
    # small imperfections
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_ellipse_size)
    im_close = cv2.morphologyEx(im_open, cv2.MORPH_CLOSE, kernel_close) 
    
    cv2.imshow(image_name + ": Kidney Open", im_open)
    cv2.imshow(image_name + ": Kidney Close", im_close)
    
    # cv2.waitKey(0)
    
    #get the countours 
    contours,_ = cv2.findContours(im_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    img_bin = numpy.zeros((im_orig.shape[0], im_orig.shape[1],3),numpy.uint8)

    #cv2.drawContours(img_bin, countours, -1, (255, 0, 255), -1)
    for i in range(0,len(contours)):
        cv2.drawContours(img_bin, contours, i, (randint(0,255),randint(0,255),randint(0,255)), -1)
        
    #cv2.drawContours(img_bin, contours, -1, (255, 0, 255), -1)
    
    cv2.imshow(image_name + ": Connected Components", img_bin)
    
    cv2.waitKey(0)
    
    # clear all windows
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    image1 = 'g006.png'
    image2 = 'e030.png'
    image3 = 'kidney.png'

    # find connected components for g006.png
    open_ellipse_size = (17,17)
    close_ellipse_size = (1,1)
    findConnectedComponents(image1, open_ellipse_size, close_ellipse_size)
    
    # find connected components for e030.png
    open_ellipse_size = (30,30)
    close_ellipse_size = (1,1)
    findConnectedComponents(image2, open_ellipse_size, close_ellipse_size)
    
    # find connected components for kidney.png
    open_ellipse_size = (17,17)
    close_ellipse_size = (2,2)
    findConnectedComponents(image3, open_ellipse_size, close_ellipse_size)
        
    print('Done')
