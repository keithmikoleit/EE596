'''
Created on Oct 6, 2015

@author: Keith Mikoleit
'''

import cv2
import numpy


def findConnectedComponents(image_name, ellipse_size):

    #get original image and display
    im_orig = cv2.imread(image_name, 0)
    
    cv2.imshow('Original', im_orig)
    
    cv2.waitKey(0)
    
    #threshold the image to create a binary image
    threshold = 128
    maxval = 255
    _, im_th = cv2.threshold(im_orig,threshold, maxval, cv2.THRESH_BINARY)
    
    cv2.imshow('Threshold', im_th)
    
    cv2.waitKey(0)
    
    # create a circular mask and use it to morph the image via opening and closing 
    # dont need to do this since we are using the countours method
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ellipse_size)
    
    im_open = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
      
    im_close = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel) 
    
    cv2.imshow('Kidney Open', im_open)
    cv2.imshow('Kidney Close', im_close)
    
    cv2.waitKey(0)
    
    #get the countours 
    countours,_ = cv2.findContours(im_open, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    img_bin = numpy.zeros((im_orig.shape[0], im_orig.shape[1],3),numpy.uint8)
    
    cv2.drawContours(img_bin, countours, -1, (255, 0, 255), -1)
    
    cv2.imshow("Connected Components", img_bin)
    
    cv2.waitKey(0) 

if __name__ == '__main__':
    image1 = 'E:\UW PMP\EE596\g006.png'
    image2 = 'E:\UW PMP\EE596\e030.png'
    image3 = 'E:\UW PMP\EE596\kidney.png'
    image = ''
    cmd = ''
    print('Enter the image file or q to quite')
    while cmd != 'q':
        cmd = raw_input('Which image do you want to process? ')
        if cmd != 'q':
            ellipse_size = tuple(int(x.strip()) for x in raw_input('Enter mask size (as ellipse parameters in a tuple) >> ').split(','))
            
            if cmd == '1':
                image = image1
            elif cmd == '2':
                image = image2
            elif cmd == '3':
                image = image3
            print (image)
            findConnectedComponents(image, ellipse_size)
        
    print('Done')
