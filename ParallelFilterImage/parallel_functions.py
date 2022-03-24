#parallel_functions.py

import numpy as np 
import multiprocessing as mp
from multiprocessing.sharedctypes import Value, Array, RawArray
import ctypes

"""We import the filter_functions to use the filter_fun function 
that is going to be parallelized with the mp.Pool function """
import filter_functions as my


def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock
    
    return np.frombuffer(mp_arr.get_obj(),dtype=np.uint8)


def image_filter(image, filter_mask, numprocessors,filtered_image):

    """Develops the parallel process of the filter function """

    # image is the image that is going to be filtered
    # filter_mask is the filter that is going to be applied
    # numprocessors is the number of processors of the gpu use for the Pool
    # filtered_image is a mp.Array where the filtered image is going to be stored

    
    """Because the filter_fun function receives a row, we are going to 
    iterate through all the rows in the image with parallel programming 
    and filter the whole image """
    
    rows = image.shape[0]
    r = range(rows)
    
    """Parallel processing process of the filtered function """
    # the processes attribute receives the number of processors, 
    # the initializer receives the initialization function that store the variables in the global memory
    # and the initargs which are the attributes that the initialization function receives. 
    with mp.Pool(processes = numprocessors, initializer = my.filter_init, initargs = [image, filter_mask, filtered_image]) as p:
        # maps the function that filters a row with the number of rows in the image through parallel processing
        row = p.map(my.filter_fun, r)
        
    # the function doesn't return anything, because the result is being stored in the shared memory buffer
    # which is in the global memory, so it can be accessed anywhere. 
    return 


if __name__ == "__main__":
    print("This is not an executable library")