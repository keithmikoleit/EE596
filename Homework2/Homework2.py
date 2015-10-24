'''
Created on Oct 16, 2015

@author: Keith Mikoleit
'''

def OpenImage(filepath, RG):
    import cv2
    import numpy as np
    """
    Opens a color image.  If RG is true, the image pixels are converted to RG space
    
    Arguments:
        filepath -- path to image file
        RG -- boollean indicating whether to convert the image to RG or not
    
    Return:
        image file opened with opencv (cv2)
    
    """
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    
    if RG:
        # reshape image into single vector
        ivec = img.reshape((-1,3))
        # convert integer values to float
        fvec = np.float32(ivec)
        
        # pixels are represented B,G,R in python
        for (i,pixel) in enumerate(fvec):
            a = np.sum(pixel)
            if a > 0:
                pixel = np.divide(pixel,np.sum(pixel))
            fvec[i] = pixel
            
                
        # convert back into image shape
        img = fvec.reshape((img.shape))

    return img

def Kmeans(filename, K, RG):
    """
    Run Kmeans algorithm to find best fit clusters of image pixels
    
    Arguments:
        filename -- path to image file to be analyzed
        K -- number of clusters to group image pixels into
        RG -- whether to open images as RG or to leave as RGB
    
    Return:
        pixels -- label per pixel of which cluster the pixel belongs to
        centers -- cluster value, which is the mean value the cluster converged to
        img -- original image
    
    """
    import cv2
    import numpy as np
    
    # read in test image
    img = OpenImage(filename, RG)
    
    #===========================================================================
    # cv2.imshow('test image', img)
    # cv2.waitKey(0)
    #===========================================================================
    
    # reshape image into single vector
    ivec = img.reshape((-1,3))
    # convert integer values to float
    fvec = np.float32(ivec)
    # create the criteria for kmeans
    # epsilon is the convergence criteria, amount which the individual parameters change it iteration
    if RG == True:
        epsilon = 0.05
    else:
        epsilon = 2
    # max_iter is the max iterations before finishing
    max_iter = 50
    # create the k means criteria
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    # number of different K means initial clusters to try
    attempts = 50
    
    ret,pixels,centers = cv2.kmeans(fvec,K,criteria,attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    #following code is from http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
    #convert back into uint8 and make original image
    
    #===========================================================================
    # if RG == False:
    #     centers = np.uint8(centers)
    # res = centers[pixels.flatten()]
    # res2 = res.reshape((img.shape))
    #  
    # cv2.imshow('kmeans image', res2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #===========================================================================
    
    return (pixels,centers,img)

def DefineSkin(pixels, face_mask, K):
    """
    Compare the pixels in the labels image to the labels in the face_mask.  If > 50% of the clustered
    pixels in the labels image are defined as skin in the face_mask, set those pixels as skin.
    
    Arguments:
        labels -- processed test image with clustered labels in pixel indices
        face_mask -- associated groundtruth showing which pixels were actually skin in test image
        K -- number of clusters test label was sorted into
    
    Return:
        pixels -- processed test image showing which pixels were associated with skin in
                  ground truth image
        labels -- which labels from the input image are classified as skin
    
    """
    
    from collections import Counter
    import numpy as np
    import cv2
    

    # open up groundtruth image
    groundtruth = cv2.imread(face_mask, cv2.IMREAD_COLOR )
    groundtruth = cv2.cvtColor(groundtruth, cv2.COLOR_RGB2GRAY)
    gvec = groundtruth.flatten()
    gvec_counts = Counter(gvec)

    #===========================================================================
    # cv2.imshow('groundtruth', groundtruth)
    # cv2.waitKey(0)
    #===========================================================================

    scores = [0] * K
    skin = 255
    notskin = 0

    pvec = pixels.flatten()
    # loop through each pixel in the label image
    for idx in range(len(pvec)):
        # if the associated pixel in the groundtruth is labeled as skin
        if gvec[idx] == skin:
            # add one to the score of the associated value in the label
            scores[pvec[idx]] = scores[pvec[idx]] + 1

    # compare the score of the K means center value with total number of pixels in that cluster
    # if it is greater than 50%, that K means cluster is labeled as skin
    # if the cluster was labeled as skin, set the associated pixels to skin
    # otherwise, set them to non skin        
    for idx in range(len(scores)):
        # set the label for that K means cluster as 1 indicating it is skin
        if scores[idx] > 0.5 * Counter(pvec)[idx]:
            scores[idx] = 1
        # otherwise set as -1 indicating it is not skin
        else:
            scores[idx] = 0
    
    #===========================================================================
    # # display processed labels
    # tempscores = np.uint8(scores)
    # res = tempscores[pvec]
    # res2 = res.reshape((groundtruth.shape)) 
    # 
    # cv2.imshow('skin image', res2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #===========================================================================
    
    return (pixels,scores)

def jaccard(s1, s2):
    """ 
        takes two flattened images as input and returns Jaccard coefficient
    
    """
    TP = 0
    FP = 0
    FN = 0
    for idx in range(len(s1)):
        if (s1[idx] == 255) & (s2[idx] == 255):
            TP = TP + 1
        elif (s1[idx] == 255) & (s2[idx] != 255):
            FP = FP + 1
        elif (s1[idx] != 255) & (s2[idx] == 255):
            FN = FN + 1
    
    return float(TP)/(TP + FP + FN)

def gatherimages(directory_and_type):
    import glob
    """
    gather all image paths in a given directory
    
    Arguments:
        directory_and_type -- the directory to search through appended with with
                              type of image file to gather.
                              formate as: directorypath/*.imagetype
    
    Return:
        images -- list of strings containing image path and filename
               -- for use with cv2.imread
    
    """
    images = []
    for (i,image_file) in enumerate(glob.iglob(directory_and_type)):
        images.append(image_file)
    return images

if __name__ == '__main__':
    
    import cv2
    import numpy as np
    
    # number of clusters to group pixels into
    K = 8
    
    # use RG space or not (RGB)
    RG = True
    
    # use Bayes or Forest
    Bayes = False
    
    # average jaccard score
    jaccaordscore = []
    
    #training data arrays
    samples = []
    labels = []
    
    # get training ground truth image files
    traininggroundtruths = gatherimages('./face_training_groundtruth/*.png')
    #===========================================================================
    # for (i,image_file) in enumerate(glob.iglob('./face_training_groundtruth/*.png')):
    #     groundtruths.append(image_file)
    #===========================================================================
    # get training images
    trainingimages = gatherimages('./face_training/*.png')
    
    #test for RG vs RGB comparison
    #===========================================================================
    # img = OpenImage(trainingimages[0], False)
    # cv2.imshow('test image', img)
    # cv2.waitKey(0)
    #  
    # img = OpenImage(trainingimages[0], True)
    # cv2.imshow('RG image', img)
    # cv2.waitKey(0)
    #===========================================================================
    
    # run k means and get training data for classifier based on comparison with groundtruth
    for (i,image_file) in enumerate(trainingimages):
        #=======================================================================
        # Test for directory loop
        # print image_file
        # img = cv2.imread(image_file)
        # cv2.imshow('test image', img)
        # cv2.waitKey(0)
        #=======================================================================
        (pixels,centers,_) = Kmeans(image_file, K, RG)
        (_,skin) = DefineSkin(pixels, traininggroundtruths[i], K)
        # append labels and centers to data structure to prepare for classifier training
        for i in range(len(centers)):
            samples.append(centers[i])
            labels.append(skin[i])
    
    # convert for use with cv2 functions
    samples = np.array(samples, np.float32)
    labels = np.array(labels, np.float32)
    
    if Bayes:
        # generate model with training data
        model = cv2.NormalBayesClassifier()
        model.train(samples, labels)
    else:
        forest = cv2.RTrees()
        rtree_params = dict(max_depth=8, min_sample_count=8, use_surrogates=False, max_categories=5, calc_var_importance=False, nactive_vars=0, max_num_of_trees_in_the_forest=1000, termcrit_type=cv2.TERM_CRITERIA_MAX_ITER, term_crit=(cv2.TERM_CRITERIA_MAX_ITER,100,1))
        forest.train(samples, cv2.CV_ROW_SAMPLE, labels, params=rtree_params)
    
    #training data arrays
    testsamples = []
    testlabels = []
    
    # get the groundtruths for the testing images
    testgroundtruths = gatherimages('./face_testing_groundtruth/*.png')
    #===========================================================================
    # for (i,image_file) in enumerate(glob.iglob('./face_testing_groundtruth/*.png')):
    #     testgroundtruths.append(image_file)
    #===========================================================================
        # print image_file
        #=======================================================================
        # img = cv2.imread(image_file)
        # cv2.imshow('test image', img)
        # cv2.waitKey(0)
        #=======================================================================
    # get test images
    testimages = gatherimages('./face_testing/*.png')
    
    # run test images through model and check performance 
    # with the jaccard similarity measure
    for (imgidx,image_file) in enumerate(testimages):
        print "test image: %s" %image_file
        # run test image through kmeans to get K clusters
        (pixels,centers,img) = Kmeans(image_file, K, RG)
        
        # convert for use with cv2
        centers = np.array(centers, np.float32)
        
        # run test image through model
        if Bayes:
            # Bayes model can take all of the centers as
            # an array
            _, p = model.predict(centers)
        else:
            # the tree model takes each center individually
            p = []
            for i in range(len(centers)):
                p.append(forest.predict(centers[i]))
            
        # convert to greyscale image based on skin prediction results
        greycenter = [0] * len(centers)
        for j in range(len(centers)):
            if p[j] == 1:
                greycenter[j] = (255)
                # for visual debug
                centers[j] = (255)
            else:
                greycenter[j] = (0)
                # for visual debug
                centers[j] = (0)
        
        if RG == False:
            centers = np.uint8(centers)
        res = centers[pixels.flatten()]
        res2 = res.reshape((img.shape))
          
        #=======================================================================
        # cv2.imshow('predicted skin image', res2)
        #=======================================================================
        
        # open up groundtruth for testing image and convert to grey scale
        print testgroundtruths[imgidx]
        groundtruth = cv2.imread(testgroundtruths[imgidx], cv2.IMREAD_COLOR )
        groundtruth = cv2.cvtColor(groundtruth, cv2.COLOR_RGB2GRAY)
        
        #=======================================================================
        # cv2.imshow('truth skin image', groundtruth)
        #=======================================================================

        
        # get jaccard score by comparing truth with model result
        greycenter = np.uint8(greycenter)
        res3 = greycenter[pixels.flatten()]
        jac = jaccard(res3, groundtruth.flatten())
        jaccaordscore.append(jac)
        print jac
        
        #=======================================================================
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #=======================================================================
        
    #===========================================================================
    #     # append labels and centers to data structure to prepare for classifier training
    #     for i in range(len(centers)):
    #         testsamples.append(centers[i])
    #         testlabels.append(p[i])
    # 
    #===========================================================================
    
#===============================================================================
#     Test code for single image
#     # run k means on each image to retrieve the labels and centers
#     test_image = './face_training/face3.png'
#     K = 8
#     (pixels,centers) = Kmeans(test_image, K)
# 
#     # compare against the training ground truth images to decide if it is skin or not
#     face_mask = './face_training_groundtruth/facemask3.png'
#     (_,skin) = DefineSkin(pixels, face_mask, K)
#     print skin
#===============================================================================
    average = sum(jaccaordscore)/len(jaccaordscore)
    print ("Average Jaccard Score: %s") %average
    print 'Done'