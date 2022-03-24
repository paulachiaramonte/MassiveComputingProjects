## functions first practical work

import numpy as np 
import multiprocessing as mp
from multiprocessing.sharedctypes import Value, Array, RawArray
import ctypes

def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock
    return np.frombuffer(mp_arr.get_obj(),dtype=np.uint8)


def filter_init(img, filter_mk, shared_array_):
    
    """Initializes the filter function by declaring the global variables and the shared space memory to store the filtered image"""
    
    global image #image that is being filtered
    global filter_mask #the filter mask applied to the image
    global shared_space # shared read/write data, with lock
    global shared_filtered_image #to stored the filtered image

    # initialize the global read only memory data
    image = img
    filter_mask = filter_mk

    size = image.shape
    #Assign the shared memory  to the local reference
    shared_space = shared_array_
    
    #Defines the numpy array-matrix reference to handle data, which uses the shared memory buffer
    shared_filtered_image = tonumpyarray(shared_space).reshape(size)


def filter_fun(row):
    
    """Receives a row from the image and returns the filtered row"""
    
    # Calls the global variables
    global image
    global filter_mask
    global shared_space
    
    # locks the shared memory space, avoiding the other parallel processes to write on the shared_space
    with shared_space.get_lock():
        
        # Obtains the dimensions of the filter
        len_filter_sh = len(filter_mask.shape)

        # Store the dimensions in f_rows and f_cols
        if len_filter_sh == 1: #for the filters of dimension 1XM
            f_rows = 1
            f_cols = filter_mask.shape[0]
        else: #for the other filters dimensions
            f_rows, f_cols = filter_mask.shape

        #Obtains the dimensions of the image
        len_image = len(image.shape)

        if len_image == 2: # images of only one layer
            rows,cols = image.shape
            depth = 1
        else: # images of three layers
            rows, cols, depth = image.shape
        
        # Define frow where the resulted filtered row is going to be stored
        if depth == 1: #when the image has one layer
            frow = np.zeros(cols)
        else: # when the image has three layers
            frow = np.zeros((cols,depth))

        """ In the following, the filter mask is going to be multiplied with all the 
        columns of the images in the different layers of the image. Each filter dimensions
        has a different behaviour and different way of handle. """
        
        ### ------- Filter of dimension 5x5 -------- ###
        if f_rows == 5 and f_cols == 5: 

            if depth == 3: # handling three layers image
            
                third_row = image[row,:,:] # current row
                
                """Taking care of the row's edges """
                
                # handling the first two rows of the filter
                if (row == 0): #in the first row of the image
                    first_row = image[row,:,:]
                    second_row = image[row,:,:]
                elif (row == 1): #in the second row of the image
                    first_row = image[row-1,:,:]
                    second_row = image[row-1,:,:]
                else: #not edges
                    first_row = image[row-2, :,:]
                    second_row = image[row-1,:,:]

                # handling the last two rows of the filter
                if(row == (rows-2)): # in the penultimate row of the image
                    fourth_row = image[row + 1,:,:]
                    fifth_row = image[row + 1,:,:]
                elif (row == (rows-1)): #in the last row of the image
                    fourth_row = image[row,:,:]
                    fifth_row = image[row,:,:]
                else: #not edges
                    fourth_row = image[row +1,:,:]
                    fifth_row = image[row +2,:,:]

                """We go through all the layers of the image for the multiplication between
                the filter mask and the image """
                for d in range(depth):
                    
                    """Taking care of the column's edges"""
                    #calculations for first column
                    temp = 0.0
                    temp = ((first_row[0,d]*filter_mask[0,0] + first_row[0,d]*filter_mask[0,1] + first_row[0,d]*filter_mask[0,2] +first_row[1,d]*filter_mask[0,3] + first_row[2,d]*filter_mask[0,4])+(second_row[0,d]*filter_mask[1,0] + second_row[0,d]*filter_mask[1,1] + second_row[0,d]*filter_mask[1,2] + second_row[1,d]*filter_mask[1,3] + second_row[2,d]*filter_mask[1,4])+(third_row[0,d]*filter_mask[2,0] + third_row[0,d]*filter_mask[2,1] + third_row[0,d]*filter_mask[2,2] + third_row[1,d]*filter_mask[2,3] + third_row[2,d]*filter_mask[2,4]) + (fourth_row[0,d]*filter_mask[3,0] + fourth_row[0,d]*filter_mask[3,1] + fourth_row[0,d]*filter_mask[3,2]+ fourth_row[1,d]*filter_mask[3,3] + fourth_row[2,d]*filter_mask[3,4])+(fifth_row[0,d]*filter_mask[4,0] + fifth_row[0,d]*filter_mask[4,1] + fifth_row[0,d]*filter_mask[4,2] + fifth_row[1,d]*filter_mask[4,3] + fifth_row[2,d]*filter_mask[4,4]))

                    frow[0,d] = int(temp) # stores the result in the first column of every layer of the image

                    #calculations for second column
                    temp = ((first_row[0,d]*filter_mask[0,0] + first_row[0,d]*filter_mask[0,1] + first_row[1,d]*filter_mask[0,2] + first_row[2,d]*filter_mask[0,3] + first_row[3,d]*filter_mask[0,4])+(second_row[0,d]*filter_mask[1,0] + second_row[0,d]*filter_mask[1,1] + second_row[1,d]*filter_mask[1,2] + second_row[2,d]*filter_mask[1,3] + second_row[3,d]*filter_mask[1,4])+(third_row[0,d]*filter_mask[2,0] + third_row[0,d]*filter_mask[2,1] + third_row[1,d]*filter_mask[2,2] + third_row[2,d]*filter_mask[2,3] + third_row[3,d]*filter_mask[2,4]) + (fourth_row[0,d]*filter_mask[3,0] + fourth_row[0,d]*filter_mask[3,1] + fourth_row[1,d]*filter_mask[3,2]+ fourth_row[2,d]*filter_mask[3,3] + fourth_row[3,d]*filter_mask[3,4])+(fifth_row[0,d]*filter_mask[4,0] + fifth_row[0,d]*filter_mask[4,1] + fifth_row[1,d]*filter_mask[4,2] + fifth_row[2,d]*filter_mask[4,3] + fifth_row[3,d]*filter_mask[4,4]))

                    frow[1,d] = int(temp) # stores the result in the second column of every layer of the image
                    
                    # calculations for the penultimate column
                    temp = ((first_row[cols-4,d]*filter_mask[0,0] + first_row[cols-3,d]*filter_mask[0,1] + first_row[cols-2,d]*filter_mask[0,2] + first_row[cols-1,d]*filter_mask[0,3] + first_row[cols-1,d]*filter_mask[0,4])+(second_row[cols-4,d]*filter_mask[1,0] + second_row[cols-3,d]*filter_mask[1,1] + second_row[cols-2,d]*filter_mask[1,2] + second_row[cols-1,d]*filter_mask[1,3] + second_row[cols-1,d]*filter_mask[1,4]) + (third_row[cols-4,d]*filter_mask[2,0] + third_row[cols-3,d]*filter_mask[2,1] + third_row[cols-2,d]*filter_mask[2,2] + third_row[cols-1,d]*filter_mask[2,3] + third_row[cols-1,d]*filter_mask[2,4]) + (fourth_row[cols-4,d]*filter_mask[3,0] + fourth_row[cols-3,d]*filter_mask[3,1] + fourth_row[cols-2,d]*filter_mask[3,2]+ fourth_row[cols-1,d]*filter_mask[3,3] + fourth_row[cols-1,d]*filter_mask[3,4])+(fifth_row[cols-4,d]*filter_mask[4,0] + fifth_row[cols-3,d]*filter_mask[4,1] + fifth_row[cols-2,d]*filter_mask[4,2] + fifth_row[cols-1,d]*filter_mask[4,3] + fifth_row[cols-1,d]*filter_mask[4,4]))

                    frow[cols-2,d] = int(temp) # stores the result in the penultimate column of every layer of the image

                    #calculation for last column
                    temp = ((first_row[cols-3,d]*filter_mask[0,0] + first_row[cols-2,d]*filter_mask[0,1] + first_row[cols-1,d]*filter_mask[0,2] + first_row[cols-1,d]*filter_mask[0,3] + first_row[cols-1,d]*filter_mask[0,4])+(second_row[cols-3,d]*filter_mask[1,0] + second_row[cols-2,d]*filter_mask[1,1] + second_row[cols-1,d]*filter_mask[1,2] + second_row[cols-1,d]*filter_mask[1,3] + second_row[cols-1,d]*filter_mask[1,4])+(third_row[cols-3,d]*filter_mask[2,0] + third_row[cols-2,d]*filter_mask[2,1] + third_row[cols-1,d]*filter_mask[2,2] + third_row[cols-1,d]*filter_mask[2,3] + third_row[cols-1,d]*filter_mask[2,4]) + (fourth_row[cols-3,d]*filter_mask[3,0] + fourth_row[cols-2,d]*filter_mask[3,1] + fourth_row[cols-1,d]*filter_mask[3,2]+ fourth_row[cols-1,d]*filter_mask[3,3] + fourth_row[cols-1,d]*filter_mask[3,4])+(fifth_row[cols-3,d]*filter_mask[4,0] + fifth_row[cols-2,d]*filter_mask[4,1] + fifth_row[cols-1,d]*filter_mask[4,2] + fifth_row[cols-1,d]*filter_mask[4,3] + fifth_row[cols-1,d]*filter_mask[4,4]))

                    frow[cols-1,d] = int(temp) # stores the result in the last column of every layer of the image
                    
                    # for loop to go through the center columns (not edges) of the image
                    
                    for i in range(2, cols-2): 
                        temp = 0.0

                        temp = ((first_row[i-2,d]*filter_mask[0,0] + first_row[i-1,d]*filter_mask[0,1] + first_row[i,d]*filter_mask[0,2] + first_row[i+1,d]*filter_mask[0,3] + first_row[i+2,d]*filter_mask[0,4])+(second_row[i-2,d]*filter_mask[1,0] + second_row[i-1,d]*filter_mask[1,1] + second_row[i,d]*filter_mask[1,2] + second_row[i+1,d]*filter_mask[1,3] + second_row[i+2,d]*filter_mask[1,4])+(third_row[i-2,d]*filter_mask[2,0] + third_row[i-1,d]*filter_mask[2,1] + third_row[i,d]*filter_mask[2,2] + third_row[i+1,d]*filter_mask[2,3] + third_row[i+2,d]*filter_mask[2,4]) + (fourth_row[i-2,d]*filter_mask[3,0] + fourth_row[i-1,d]*filter_mask[3,1] + fourth_row[i,d]*filter_mask[3,2]+ fourth_row[i+1,d]*filter_mask[3,3] + fourth_row[i+2,d]*filter_mask[3,4])+(fifth_row[i-2,d]*filter_mask[4,0] + fifth_row[i-1,d]*filter_mask[4,1] + fifth_row[i,d]*filter_mask[4,2] + fifth_row[i+1,d]*filter_mask[4,3] + fifth_row[i+2,d]*filter_mask[4,4]))

                        frow[i,d] = int(temp) #stores the multiplications/sum result in the i column and d layer

            elif depth == 1: #when the image has only one layer

                """When the image only has one layer the row has two dimensions """
                
                third_row = image[row,:] # current row (only two dimensions)
                
                """Taking care of the row's edges"""
                # handling the first two rows of the filter
                if (row == 0): #in the first row of the image
                    first_row = image[row,:]
                    second_row = image[row,:]
                elif (row == 1): #in the second row of the image
                    first_row = image[row-1,:]
                    second_row = image[row-1,:]
                else:
                    first_row = image[row-2, :]
                    second_row = image[row-1,:]

                # handling the last two rows of the filter
                if(row == (rows-2)): # in the penultimate row of the image
                    fourth_row = image[row + 1,:]
                    fifth_row = image[row + 1,:]
                elif (row == (rows-1)): #in the last row of the image
                    fourth_row = image[row,:]
                    fifth_row = image[row,:]
                else:
                    fourth_row = image[row +1,:]
                    fifth_row = image[row +2,:]


                """Handling the column's edges"""
                #calculations for first column
                temp = 0.0
                temp = ((first_row[0]*filter_mask[0,0] + first_row[0]*filter_mask[0,1] + first_row[0]*filter_mask[0,2] +first_row[1]*filter_mask[0,3] + first_row[2]*filter_mask[0,4])+(second_row[0]*filter_mask[1,0] + second_row[0]*filter_mask[1,1] + second_row[0]*filter_mask[1,2] + second_row[1]*filter_mask[1,3] + second_row[2]*filter_mask[1,4])+(third_row[0]*filter_mask[2,0] + third_row[0]*filter_mask[2,1] + third_row[0]*filter_mask[2,2] + third_row[1]*filter_mask[2,3] + third_row[2]*filter_mask[2,4]) + (fourth_row[0]*filter_mask[3,0] + fourth_row[0]*filter_mask[3,1] + fourth_row[0]*filter_mask[3,2]+ fourth_row[1]*filter_mask[3,3] + fourth_row[2]*filter_mask[3,4])+(fifth_row[0]*filter_mask[4,0] + fifth_row[0]*filter_mask[4,1] + fifth_row[0]*filter_mask[4,2] + fifth_row[1]*filter_mask[4,3] + fifth_row[2]*filter_mask[4,4]))

                frow[0] = int(temp) # stores the result in the first column of the image

                #calculations for second column
                temp = ((first_row[0]*filter_mask[0,0] + first_row[0]*filter_mask[0,1] + first_row[1]*filter_mask[0,2] + first_row[2]*filter_mask[0,3] + first_row[3]*filter_mask[0,4])+(second_row[0]*filter_mask[1,0] + second_row[0]*filter_mask[1,1] + second_row[1]*filter_mask[1,2] + second_row[2]*filter_mask[1,3] + second_row[3]*filter_mask[1,4])+(third_row[0]*filter_mask[2,0] + third_row[0]*filter_mask[2,1] + third_row[1]*filter_mask[2,2] + third_row[2]*filter_mask[2,3] + third_row[3]*filter_mask[2,4]) + (fourth_row[0]*filter_mask[3,0] + fourth_row[0]*filter_mask[3,1] + fourth_row[1]*filter_mask[3,2]+ fourth_row[2]*filter_mask[3,3] + fourth_row[3]*filter_mask[3,4])+(fifth_row[0]*filter_mask[4,0] + fifth_row[0]*filter_mask[4,1] + fifth_row[1]*filter_mask[4,2] + fifth_row[2]*filter_mask[4,3] + fifth_row[3]*filter_mask[4,4]))

                frow[1] = int(temp) # stores the result in the second column of the image
                
                #calculations for the penulatimate column
                temp = ((first_row[cols-4]*filter_mask[0,0] + first_row[cols-3]*filter_mask[0,1] + first_row[cols-2]*filter_mask[0,2] + first_row[cols-1]*filter_mask[0,3] + first_row[cols-1]*filter_mask[0,4])+(second_row[cols-4]*filter_mask[1,0] + second_row[cols-3]*filter_mask[1,1] + second_row[cols-2]*filter_mask[1,2] + second_row[cols-1]*filter_mask[1,3] + second_row[cols-1]*filter_mask[1,4]) + (third_row[cols-4]*filter_mask[2,0] + third_row[cols-3]*filter_mask[2,1] + third_row[cols-2]*filter_mask[2,2] + third_row[cols-1]*filter_mask[2,3] + third_row[cols-1]*filter_mask[2,4]) + (fourth_row[cols-4]*filter_mask[3,0] + fourth_row[cols-3]*filter_mask[3,1] + fourth_row[cols-2]*filter_mask[3,2]+ fourth_row[cols-1]*filter_mask[3,3] + fourth_row[cols-1]*filter_mask[3,4])+(fifth_row[cols-4]*filter_mask[4,0] + fifth_row[cols-3]*filter_mask[4,1] + fifth_row[cols-2]*filter_mask[4,2] + fifth_row[cols-1]*filter_mask[4,3] + fifth_row[cols-1]*filter_mask[4,4]))

                frow[cols-2] = int(temp) # stores the result in the penultimate column of the image

                #calculation for last column
                temp = ((first_row[cols-3]*filter_mask[0,0] + first_row[cols-2]*filter_mask[0,1] + first_row[cols-1]*filter_mask[0,2] + first_row[cols-1]*filter_mask[0,3] + first_row[cols-1]*filter_mask[0,4])+(second_row[cols-3]*filter_mask[1,0] + second_row[cols-2]*filter_mask[1,1] + second_row[cols-1]*filter_mask[1,2] + second_row[cols-1]*filter_mask[1,3] + second_row[cols-1]*filter_mask[1,4])+(third_row[cols-3]*filter_mask[2,0] + third_row[cols-2]*filter_mask[2,1] + third_row[cols-1]*filter_mask[2,2] + third_row[cols-1]*filter_mask[2,3] + third_row[cols-1]*filter_mask[2,4]) + (fourth_row[cols-3]*filter_mask[3,0] + fourth_row[cols-2]*filter_mask[3,1] + fourth_row[cols-1]*filter_mask[3,2]+ fourth_row[cols-1]*filter_mask[3,3] + fourth_row[cols-1]*filter_mask[3,4])+(fifth_row[cols-3]*filter_mask[4,0] + fifth_row[cols-2]*filter_mask[4,1] + fifth_row[cols-1]*filter_mask[4,2] + fifth_row[cols-1]*filter_mask[4,3] + fifth_row[cols-1]*filter_mask[4,4]))

                frow[cols-1] = int(temp) # stores the result in the last column of the image
        
                 # for loop to go through the center columns (not edges) of the image
                for i in range(2, cols-2):
                    temp = 0.0

                    temp = ((first_row[i-2]*filter_mask[0,0] + first_row[i-1]*filter_mask[0,1] + first_row[i]*filter_mask[0,2] + first_row[i+1]*filter_mask[0,3] + first_row[i+2]*filter_mask[0,4])+(second_row[i-2]*filter_mask[1,0] + second_row[i-1]*filter_mask[1,1] + second_row[i]*filter_mask[1,2] + second_row[i+1]*filter_mask[1,3] + second_row[i+2]*filter_mask[1,4])+(third_row[i-2]*filter_mask[2,0] + third_row[i-1]*filter_mask[2,1] + third_row[i]*filter_mask[2,2] + third_row[i+1]*filter_mask[2,3] + third_row[i+2]*filter_mask[2,4]) + (fourth_row[i-2]*filter_mask[3,0] + fourth_row[i-1]*filter_mask[3,1] + fourth_row[i]*filter_mask[3,2]+ fourth_row[i+1]*filter_mask[3,3] + fourth_row[i+2]*filter_mask[3,4])+(fifth_row[i-2]*filter_mask[4,0] + fifth_row[i-1]*filter_mask[4,1] + fifth_row[i]*filter_mask[4,2] + fifth_row[i+1]*filter_mask[4,3] + fifth_row[i+2]*filter_mask[4,4]))

                    frow[i] = int(temp) #stores the multiplications/sum result in the i column

        ### ----------------------------------------------- ###

        ### ------- Filter of dimension 5x1 -------- ###
        if f_rows == 5 and f_cols == 1:

            if depth == 3: #if the image hast three layers
                
                third_row = image[row,:,:] # current row
                
                """Taking care of the row's edges"""
                # handling the first two rows of the filter
                if (row == 0): #in the first row of the image
                    first_row = image[row,:,:]
                    second_row = image[row,:,:]
                elif (row == 1): #in the second row of the image
                    first_row = image[row -1,:,:]
                    second_row = image[row-1,:,:]
                else:
                    first_row = image[row-2, :,:]
                    second_row = image[row-1,:,:]

                # handling the last two rows of the filter
                if(row == (rows-2)): # in the penultimate row of the image
                    fourth_row = image[row + 1,:,:]
                    fifth_row = image[row + 1,:,:]
                elif (row == (rows-1)): #in the last row of the image
                    fourth_row = image[row,:,:]
                    fifth_row = image[row,:,:]
                else:
                    fourth_row = image[row +1,:,:]
                    fifth_row = image[row +2,:,:]
                 
                
                """For loop to go trhough all the layers and columns of the image """
                for d in range(depth):
                    
                    ## Because the filter has a vertical figure, it doesn't have any problem
                    ## when handling with the edges of the columns. 
                    
                    for i in range(0,cols):
                        temp = 0.0

                        temp = ( (first_row[i,d]*filter_mask[0,0]) + (second_row[i,d]*filter_mask[1,0]) + (third_row[i,d]*filter_mask[2,0]) + (fourth_row[i,d]*filter_mask[3,0]) + (fifth_row[i,d]*filter_mask[4,0]) )

                        frow[i,d] = int(temp) #stores the multiplications/sum result in the i column and the d layer

            elif depth == 1: #handling a image of one layer 
                
                third_row = image[row,:] # current row of only two dimensions
                
                """Taking care of the row's edges"""
                # handling the first two rows of the filter
                if (row == 0): #in the first row of the image
                    first_row = image[row,:]
                    second_row = image[row,:]
                elif (row == 1): #in the second row of the image
                    first_row = image[row -1,:]
                    second_row = image[row-1,:]
                else:
                    first_row = image[row-2, :]
                    second_row = image[row-1,:]

                # handling the last two rows of the filter
                if(row == (rows-2)): # in the penultimate row of the image
                    fourth_row = image[row + 1,:]
                    fifth_row = image[row + 1,:]
                elif (row == (rows-1)): #in the last row of the image
                    fourth_row = image[row,:]
                    fifth_row = image[row,:]
                else:
                    fourth_row = image[row +1,:]
                    fifth_row = image[row +2,:]
                
                """For loop to go through all the columns of the image"""
                for i in range(0,cols):
                    temp = 0.0

                    temp = ( (first_row[i]*filter_mask[0,0]) + (second_row[i]*filter_mask[1,0]) + (third_row[i]*filter_mask[2,0]) + (fourth_row[i]*filter_mask[3,0]) + (fifth_row[i]*filter_mask[4,0]) )

                    frow[i] = int(temp) #stores the multiplications/sum result in the i column

        ### --------------------------------------------------####

        ### ------- Filter of dimension 1x5 -------- ###

        if f_rows == 1 and f_cols == 5:

            if depth == 3: #if the image has three layers
                
                crow = image[row,:,:] #only one row (current row) because of the shape of the filter (horizontal)
                
                # There's no problem with the row's edges because of the shape of the filter
                
                """For loop to go through all the layers of the image"""
                for d in range(depth):
                    
                    """Taking care of the column's edges"""
                    temp = 0.0
                    #handling first column
                    temp = crow[0,d]*filter_mask[0] + crow[0,d]*filter_mask[1] + crow[0,d]*filter_mask[2] + crow[1,d]*filter_mask[3] + crow[2,d]*filter_mask[4]

                    frow[0,d] = int(temp) # stores the result in the first column of the image in the d layer

                    #handling second column
                    temp = crow[0,d]*filter_mask[0] + crow[0,d]*filter_mask[1] + crow[1,d]*filter_mask[2] + crow[2,d]*filter_mask[3] + crow[3,d]*filter_mask[4]

                    frow[1,d] = int(temp) # stores the result in the second column of the image in the d layer

                    #handling penultimate column
                    temp = crow[cols-4,d]*filter_mask[0] + crow[cols-3,d]*filter_mask[1] + crow[cols-2,d]*filter_mask[2] + crow[cols-1,d]*filter_mask[3] + crow[cols-1,d]*filter_mask[4]

                    frow[cols-2,d] = int(temp) # stores the result in the penultimate column of the image in the d layer


                    #handling last column
                    temp = crow[cols-3,d]*filter_mask[0] + crow[cols-2,d]*filter_mask[1] + crow[cols-1,d]*filter_mask[2] + crow[cols-1,d]*filter_mask[3] + crow[cols-1,d]*filter_mask[4]

                    frow[cols-1, d] = int(temp)
                    
                    """For loop to go through all the columns (not edges) of the image"""
                    for i in range(2,cols-2):
                        temp = 0.0

                        temp = crow[i-2,d]*filter_mask[0] + crow[i-1,d]*filter_mask[1] + crow[i,d]*filter_mask[2] + crow[i +1,d]*filter_mask[3] + crow[i+2,d]*filter_mask[4]

                        frow[i,d] = int(temp) #stores the multiplications/sum result in the i column and the d layer

            elif depth == 1: # when the images has only one layer
                
                crow = image[row,:] #two dimensional row because of the shape of the filter (horizontal)
                
                """Taking care of the column's edges"""
                temp = 0.0
                #handling first column
                temp = crow[0]*filter_mask[0] + crow[0]*filter_mask[1] + crow[0]*filter_mask[2] + crow[1]*filter_mask[3] + crow[2]*filter_mask[4]

                frow[0] = int(temp) # stores the result in the first column of the image

                #handling second column
                temp = crow[0]*filter_mask[0] + crow[0]*filter_mask[1] + crow[1]*filter_mask[2] + crow[2]*filter_mask[3] + crow[3]*filter_mask[4]

                frow[1] = int(temp) # stores the result in the second column of the image

                #handling penultimate column
                temp = crow[cols-4]*filter_mask[0] + crow[cols-3]*filter_mask[1] + crow[cols-2]*filter_mask[2] + crow[cols-1]*filter_mask[3] + crow[cols-1]*filter_mask[4]

                frow[cols-2] = int(temp) # stores the result in the penultimate column of the image

                #handling last column
                temp = crow[cols-3]*filter_mask[0] + crow[cols-2]*filter_mask[1] + crow[cols-1]*filter_mask[2] + crow[cols-1]*filter_mask[3] + crow[cols-1]*filter_mask[4]
                
                frow[cols-1] = int(temp) # stores the result in the last column of the image

                """For loop to go through all the columns of the image (not edges)"""
                for i in range(2,cols-2):
                    temp = 0.0

                    temp = crow[i-2]*filter_mask[0] + crow[i-1]*filter_mask[1] + crow[i]*filter_mask[2] + crow[i +1]*filter_mask[3] + crow[i+2]*filter_mask[4]

                    frow[i] = int(temp) #stores the multiplications/sum result in the i column 
        #### ---------------------------------------- ####

       ### ------- Filter of dimension 3x3 -------- ###
        if f_rows == 3 and f_cols == 3:

            if depth == 3: #if the image has 3 layers
                
                crow = image[row,:,:] # actual row
                
                """Taking care of the row's edges"""
                if (row > 0): #if not the first row
                    prow = image[row-1,:,:] 
                else: #if we are in the first row
                    prow = image[row,:,:]

                if (row == (rows-1)): #if we are in the last row
                    nrow = image[row,:,:] #next row
                else:
                    nrow = image[row + 1,:,:]
                
                """For loop to go through all the layers of the image"""
                for d in range(depth):
                    
                    """Handling the column's edges"""
                    # First Column
                    temp = 0.0

                    temp = ((prow[0,d]*filter_mask[0,0]+prow[0,d]*filter_mask[0,1]+prow[1,d]*filter_mask[0,2])+ #previous row
                        (crow[0,d]*filter_mask[1,0]+crow[0,d]*filter_mask[1,1]+crow[1,d]*filter_mask[1,2])+ # current row
                        (nrow[0,d]*filter_mask[2,0]+nrow[0,d]*filter_mask[2,1]+nrow[1,d]*filter_mask[2,2])) # next row

                    frow[0,d] = int(temp) # stores the result in the first column of the image and layer d

                    # Last Column

                    temp = ((prow[cols-2,d]*filter_mask[0,0]+prow[cols-1,d]*filter_mask[0,1]+prow[cols-1,d]*filter_mask[0,2])+ #prev row
                        (crow[cols-2,d]*filter_mask[1,0]+crow[cols-1,d]*filter_mask[1,1]+crow[cols-1,d]*filter_mask[1,2])+ # current row
                        (nrow[cols-2,d]*filter_mask[2,0]+nrow[cols-1,d]*filter_mask[2,1]+nrow[cols-1,d]*filter_mask[2,2])) # next row

                    frow[cols-1,d] = int(temp) # stores the result in the last column of the image and layer d
                    
                    """For loop to go through all the columns (not edges) of the image"""
                    for i in range(1, cols-1): 
                        temp = 0.0
                        
                        temp = ((prow[i-1,d]*filter_mask[0,0]+prow[i,d]*filter_mask[0,1]+prow[i+1,d]*filter_mask[0,2])+#prev row
                        (crow[i-1,d]*filter_mask[1,0]+crow[i,d]*filter_mask[1,1]+crow[i+1,d]*filter_mask[1,2])+ # current row
                        (nrow[i-1,d]*filter_mask[2,0]+nrow[i,d]*filter_mask[2,1]+nrow[i+1,d]*filter_mask[2,2])) # next row

                        frow[i,d] = int(temp) #stores the multiplications/sum result in the i column and the d layer

            elif depth == 1: #if the image has only one layer

                crow = image[row,:] # actual row of only two dimensions
                
                """Taking care of the row's edges"""
                if (row > 0): #if not in the first row
                    prow = image[row-1,:] 
                else: #if we are in the first row
                    prow = image[row,:]

                if (row == (rows-1)): #if we are in the last row
                    nrow = image[row,:] #next row
                else:
                    nrow = image[row + 1,:]
                
                """Handling the column's edges"""
                # First Column
                temp = 0.0

                temp = ((prow[0]*filter_mask[0,0]+prow[0]*filter_mask[0,1]+prow[1]*filter_mask[0,2])+ #previous row
                    (crow[0]*filter_mask[1,0]+crow[0]*filter_mask[1,1]+crow[1]*filter_mask[1,2])+ # current row
                    (nrow[0]*filter_mask[2,0]+nrow[0]*filter_mask[2,1]+nrow[1]*filter_mask[2,2])) # next row

                frow[0] = int(temp) # stores the result in the first column of the image 

                # Last Column

                temp = ((prow[cols-1]*filter_mask[0,0]+prow[cols-2]*filter_mask[0,1]+prow[cols-2]*filter_mask[0,2])+ #previous row
                    (crow[cols-1]*filter_mask[1,0]+crow[cols-2]*filter_mask[1,1]+crow[cols-2]*filter_mask[1,2])+ # current row
                    (nrow[cols-1]*filter_mask[2,0]+nrow[cols-2]*filter_mask[2,1]+nrow[cols-2]*filter_mask[2,2])) # next row

                frow[cols-1] = int(temp) # stores the result in the last column of the image 
                
                """For loop to go through all the columns of the image (not edges)"""
                for i in range(1, cols-1): 
                    temp = 0.0

                    temp = ((prow[i-1]*filter_mask[0,0]+prow[i]*filter_mask[0,1]+prow[i+1]*filter_mask[0,2])+#prev row
                        (crow[i-1]*filter_mask[1,0]+crow[i]*filter_mask[1,1]+crow[i+1]*filter_mask[1,2])+ # current row
                        (nrow[i-1]*filter_mask[2,0]+nrow[i]*filter_mask[2,1]+nrow[i+1]*filter_mask[2,2])) # next row

                    frow[i] = int(temp) #stores the multiplications/sum result in the i column

        ############# ------------------- ############

        ### ------- Filter of dimension 3x1 -------- ###
        if f_rows == 3 and f_cols == 1:

            if depth == 3: #if the image has three layers

                crow = image[row,:,:] # actual row
                
                """Taking care of the row's edges"""
                if (row > 0):
                    prow = image[row-1,:,:] #previous row
                else: #if we are in the first row
                    prow = image[row,:,:]

                if (row == (rows-1)): #if we are in the last row
                    nrow = image[row,:,:] #next row
                else:
                    nrow = image[row + 1,:,:]
                
                """For loop to go through all the layers of the image"""
                for d in range(depth):
                    # because of the shape of the filter (vertical) there's no problem with the edges
                    # of the columns of the image 
                    
                    """For loop to go through all the columns of the image"""
                    for i in range(0,cols):

                        temp = 0.0

                        temp = prow[i,d]*filter_mask[0,0] + crow[i,d]*filter_mask[1,0] + nrow[i,d]*filter_mask[2,0]

                        frow[i,d] = int(temp) #stores the multiplications/sum result in the i column and layer d

            elif depth == 1: # when the image only has one layer

                crow = image[row,:] # actual row of only two dimensions
                
                """Taking care of the row's edges"""
                if (row > 0):
                    prow = image[row-1,:] #previous row
                else: #if we are in the first row
                    prow = image[row,:]

                if (row == (rows-1)): #if we are in the last row
                    nrow = image[row,:] #next row
                else:
                    nrow = image[row + 1,:]
                
                """For loop to go through all the columns of the image"""
                for i in range(0,cols):

                    temp = 0.0

                    temp = prow[i]*filter_mask[0,0] + crow[i]*filter_mask[1,0] + nrow[i]*filter_mask[2,0]

                    frow[i] = int(temp)  #stores the multiplications/sum result in the i column


        ### ---------------------------------- ###

       ### ------- Filter of dimension 1x3-------- ###
        if f_rows == 1 and f_cols == 3:

            if depth == 3: #when the image has three layers

                # Because of the shape of the filter (horizontal) there's no problem
                # with the row's edges 
                crow = image[row,:,:]

                """ For loop to go through all the layers of the image """
                for d in range(depth):
                    
                    """Taking care of the column's edges"""
                    temp = 0.0
                    #handling first column
                    temp = crow[0,d]*filter_mask[0] + crow[0,d]*filter_mask[1] + crow[1,d]*filter_mask[2]

                    frow[0,d] = int(temp) # stores the result in the first column of the image and layer d

                    #handling last column
                    temp = crow[cols-2,d]*filter_mask[0] + crow[cols-1,d]*filter_mask[1] + crow[cols-1,d]*filter_mask[2]

                    frow[cols-1,d] = int(temp) # stores the result in the last column of the image and layer d

                    """For loop to go through all the columns (not edges) of the image"""
                    for i in range(1,cols-1):
                        temp = 0

                        temp = crow[i-1,d]*filter_mask[0] + crow[i,d]*filter_mask[1] + crow[i+1,d]*filter_mask[2]

                        frow[i,d] = int(temp) #stores the multiplications/sum result in the i column and layer d

            elif depth == 1: #when the image only has one layer
                
                crow = image[row,:] #row of two dimensions
                # Because of the shape of the filter there's no problem 
                # with the row's edges
                
                """Taking care of the column's edges"""
                temp = 0.0
                #handling first column
                temp = crow[0]*filter_mask[0] + crow[0]*filter_mask[1] + crow[1]*filter_mask[2]

                frow[0] = int(temp) # stores the result in the first column of the image

                #handiling last column
                temp = crow[cols-2]*filter_mask[0] + crow[cols-1]*filter_mask[1] + crow[cols-1]*filter_mask[2]

                frow[cols-1] = int(temp) # stores the result in the last column of the image

                """For loop to go through all the columns of the image (not edges) """
                for i in range(1,cols-1):
                    temp = 0

                    temp = crow[i-1]*filter_mask[0] + crow[i]*filter_mask[1] + crow[i+1]*filter_mask[2]

                    frow[i] = int(temp) #stores the multiplications/sum result in the i column

        #### ---------------------------------- ####
        """After computing the matrix multiplication/sum of the current row of 
        the image we stored the result in the shared memory array """
        
        #while we are in this block of code, no one (except this execution thread) 
        # can write in the shared memory, so we are avoiding a race condition 
        
        if depth == 3: # store the results to the shared memory when the image has three layers
            shared_filtered_image[row,:,:] = frow
        elif depth == 1: # store the results to the shared memory when the image has one layer
            shared_filtered_image[row,:] = frow
            
        # the function doesn't return anything, because the result is being stored in the shared memory buffer
        # which is in the global memory, so it can be accessed anywhere. 
        return




#This cell should be the last one
#this avoid the execution of this script when is invoked directly.
if __name__ == "__main__":
    print("This is not an executable library")

