# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:58:56 2014

@author: daniel
"""

import numpy as np
import os
import matplotlib.pyplot as plt

DATA_START_TOKEN = 'data_start'
DATA_END_TOKEN = 'data_end'

# This is not part of the TT class
def _readFile(fname,np_dtype,onlyReadHeader=False):
    BATCH_SIZE = 1024*4 # 4Kb
    
    hasBinary = False if np_dtype == False else True

    if not os.path.isfile(fname):
        if onlyReadHeader:
            return {},
        else:
            return {},np.array([])
        
    with open(fname,'rb') as f:
        if hasBinary:  
            # If this file type has binary data, we have to find the DATA_START_TOKEN,
            # And put everything before that into headerStr.  Then seek to the start of the data.
            headerStr = newBatch = f.read(BATCH_SIZE)
            while len(newBatch):
                newBatch = f.read(BATCH_SIZE)
                headerStr += newBatch
                data_start_loc = headerStr.find(DATA_START_TOKEN)
                if data_start_loc:
                    headerStr = headerStr[:data_start_loc]
                    f.seek(data_start_loc + len(DATA_START_TOKEN))
                    break
        else:
            #If this file doesnt have binary data we just read the whole thing into the headerStr
            headerStr = f.read()
            
        # Now we parse the headerStr into a dict
        header = {} if not hasBinary else {'data_start_seek': data_start_loc + len(DATA_START_TOKEN)}
        lines = headerStr.splitlines()
        for line in lines:
            line = str(line).rstrip()
            line = line.split(' ',1)
            header[line[0]] = line[1] if len(line) > 1 else None
        
        if not hasBinary or onlyReadHeader:
            return header,
            
        # Now we read in the binary data using the numpy data type list provided
        data = np.fromfile(f, np.dtype(np_dtype))
        
        # Stip off the data_end token (TODO: do this properly!)
        # print(data.shape)
        tailLen = int(len(DATA_END_TOKEN) / np.dtype(np_dtype).itemsize)
        if tailLen > 0:
            data = data[:-tailLen]
        return header, data
