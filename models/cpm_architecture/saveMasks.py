# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 17:40:52 2016

@author: denitome
"""
import os
from sys import argv
import scipy.io as sio
import numpy as np

def createFile(matfile, output):
    keys = ['mask_camera','mask_action','mask_person']
    mat = sio.loadmat(matfile)        
    with open(output, "w") as outputfile:    
        masks = np.array([mat[keys[0]].shape[1]])
        for key in keys:
            curr_mask = mat[key][0,:]
            masks = np.concatenate((masks,curr_mask))
        array_byte = bytearray(masks.astype(np.uint32))
        outputfile.write(array_byte)
    
def main():
    
    num_param = len(argv)
    if num_param < 3:
        print "Expected format: saveMasks.py input.mat output"
        return
        
    (_, file_path, output_path) = argv
    
    if not os.path.isfile(file_path):
        print "Inexistent input file"
        return
        
    createFile(file_path, output_path)    

if __name__ == "__main__":
    main()   