### PARTE 2 FUNCTIONS
import numpy as np 
import multiprocessing as mp
from multiprocessing.sharedctypes import Value, Array, RawArray
import ctypes

"""We import the parallel_function to use the image_filter function 
that is going to be multiprocessed with the mp.Process function """
import parallel_functions as pf


def filters_execution(image, filter_mask1, filter_mask2, numprocessors, filtered_image1, filtered_image2):
    """Executes the double process of two parallel processing of one execution function (image_filter)
    that will be applied to multiple data (two different filters) by using a Multiple Data Program 
    (mp.Process) and semaphores to avoid the race condition"""
    
    # Divide the number of processors by the number of Processes being executed (2)
    num_processors = int(numprocessors/2)
    
    """we construct as many multiprocessing.Process objects we need, assigning the filter function 
    which will be executed in each one of the parallel process, and passing their arguments values"""
    f1 = mp.Process(target = pf.image_filter, args = [image, filter_mask1, num_processors, filtered_image1])
    f2 = mp.Process(target = pf.image_filter, args = [image, filter_mask2, num_processors, filtered_image2])

    # Start the processes
    f1.start()
    f2.start()

    # wait until the processes are finished
    f1.join()
    f2.join()
    
    """In this function we avoid the race condition by using a shared memory with locks in the image_filter function
    which uses the filter_fun function that receives a shared memory array with locks. So at last we are using two
    different shared memory allocations to avoid overwriting """
    
    return