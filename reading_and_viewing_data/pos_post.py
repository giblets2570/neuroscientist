# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 00:27:51 2014

@author: daniel


This module is something of a mess, but hopefully it is simplit into sufficiently 
small pieces so as to be able to make sense of it all.

Basically there are lots of steps to processing the pos data, this module atempts
to bring them all together in one place and make it easyish to swap in different 
bits in the chain of processing.



TODO: there is likely a confusion between x/y and w/h in various places...if it
matters to you you'll have to fix it.

TODO: there are a number of points where we loop over nLEDs, although this isnt 
going to be a bottleneck, it makes the code a bit messier, and can probably be
vectorised into a single statement.
"""


from numpy import * # ahve it both ways, why not...
import numpy as np
from warnings import warn
import scipy as sp
import matplotlib.pyplot as plt

from jumploopproc import jumpLoopProc
from circlefit import fitCircle
from custom_casting import str2float
from accumarray_numpy import accumarray

class _PROC_STATE: #a glorified dictionary for holding and passing around proc info
    pass


class POS_PROCESSOR:
    def __init__(self,
                 window_mode=1,
                 ppmOverride=None,
                 swap_filter=1,
                 speed_filter=1,
                 MAXSPEED=4.0, #speed filter in m/s
                 fitShape = None,
                 EDGE_PERCENT_CIRC=2,  #for matching to environment shape
                 EDGE_PERCENT_RECT=0.5, #for matching to environment shape
                 CIRCLE_ANG_RES=15,
                 CIRCLE_RAD_RES=1,
                 CIRCLE_DWELL_THRESH= 0.25/1200, # fraction of trial spent in circle segment to count as occupied
                 CIRCLE_FORCE_RADIUS=None, # value in cm, or None to use fitted value
                 SMOOTH_MS=400, # this gives a 400ms smoothing window for pos averaging
                 SWAP_THRESH = 5,
                 fitted_padding=1.02, #this is the factor used when padding around the fitted shape
                 fixedW=None,fixedH=None # if a shape is fitted it will be centered within a box this size
                 ):
        """
            Note: originally this wasn't a class, it was a factory function (i.e. a closure). It's now
            be turned into a class to make pickling possible, but the conversion to class-form was 
            rushed and didn't really change much, meaning comments and other stuff may be a bit
            misleading.
            ------
            
            You supply a set of optional paramters here and then you get out a function of the form::
            
                process(led_pos,led_pix,posSampleRate,posHeader,nLEDs,returnWorking=False)
    
            That function will take in raw pos data and go through a number of steps to produce xy, and dir data.
            
            There are quite a few options going in to this factory function, best to check in the code for details.
        
            If you want to add your own stuff, it's best to stick to the pardaigm already in place,
            i.e. make a small function and put it into to pipieline somehwere in the process function,
            ideally with flags controlling its execution.
            
            Note that calcualting speed is pretty easy once you have a "nice" set of xy values, so we leave it up to 
            a separate function in trial.py
            
            TODO
            -----
            Finish off basic processing of dir and speed. (see Robin's code at bottom of file)
        """
        self._ = locals() # TODO: this was a quick hack to convert from the function-factorary paradigm
        del self._['self']
        
    def __str__(self):
        return "POS_PROCESSOR with parameters: " + str(self._)
        
    def __call__(self,led_pos,led_pix,posSampleRate,posHeader,nLEDs,returnWorking=False):
        """
        led_pos and led_pix are masked arrays.
        This function aplies a series of transformations, error masks, and LED swaps
        and then interpolates across the missing values.
        """
        # We store all the data stuff in a "structure" (i.e. an empty class) called P, this makes
        # it easy to  pass it a bit more explicit that eacah of the subfunctions is modifying
        # the same single copy of the data...unless they choose to copy it.
        P = _PROC_STATE()
        P.nLEDs = nLEDs
        P.posHeader = posHeader
        P.posSampleRate = posSampleRate
        P.ppm = str2float(posHeader['pixels_per_metre']) if self._['ppmOverride'] is None else self._['ppmOverride']
        nPos = P.nPos = shape(led_pos)[0]
        led_pos = reshape(led_pos,[nPos,nLEDs,2])
        
        # Unfortunately the code is written for led_pos as [xy,nLEDs,nPos],
        # and led_pix as [nLEDs, nPos], so we need to permute a.k.a. transpose the arrays:
        P.led_pos = transpose(led_pos,axes=[2,1,0]) 
        P.led_pix = transpose(led_pix,axes=[1,0])
        
        if ma.getmask(P.led_pos) is not ma.nomask:
            _matchMaskForXY(P)
        
        #From this point on we should maintain matched masking for (x,y) pairs (independantly for each LED)
        
        if self._['window_mode'] == 1:
            _windowCoordinates(P)
            
        if any(P.led_pix) and nLEDs == 2:
            if self._['swap_filter'] == 1:
                _findShrunkenBigLed(P) 
                _findCrossovers(P,self._['SWAP_THRESH'])
                swapInds = (P.shrunkenBig & P.crossovers).nonzero()
                led_pos,led_pix = P.led_pos,P.led_pix
                led_pos[:,:,swapInds ] = led_pos[:,::-1,swapInds ]
                led_pix[:,swapInds ] = led_pix[::-1,swapInds ]              
        
        if self._['speed_filter'] == 1:
            _ledSpeedFilter(P,self._['MAXSPEED']) 
        
        _castPosToFloat(P) #need to do this before converting to cm, or we will encur significant rounding
        
        _pixToCm(P)
        
        P.envShape = None
        if self._['fitShape'] == "circleA":
            _findCircle(P,self._['EDGE_PERCENT_CIRC'],self._['CIRCLE_RAD_RES'],
                        self._['CIRCLE_ANG_RES'],self._['CIRCLE_DWELL_THRESH'],self._['CIRCLE_FORCE_RADIUS'])
            _windowCoordinatesCirc(P,self._['fitted_padding'])
        elif self._['fitShape'] == "rectA":
            _findRect(P,self._['EDGE_PERCENT_RECT'])
            _windowCoordinatesRect(P,self._['fitted_padding'])            
        elif self._['fitShape'] is not None:
            warn("postProcessor fitShape=%s, is not a recognised fitting option." % (self._['fitShape']))
            
            
        if self._['fixedW'] is not None or self._['fixedH'] is not None:            
            _windowFixedWH(P,self._['fixedW'],self._['fixedH'])
        
        _getLEDWeights(P) # we do this just before interpolating across the mask
            
        _interpAcrossMask(P)
        
        _smooth(P,self._['SMOOTH_MS']) # this has to be done after interpolating across masked parts of led_pos
            
        _combineLEDs(P)
                
        if returnWorking:
            return P
        else:            
            return P.xy,None,0,P.w,P.h, P.envShape
            #return xy,dir,dir_disp



class POS_PROCESSOR_MAT_A:
    def __init__(self,
                 offset=None,
                 fixedW=None,fixedH=None
                 ):
        """
            This is a simpler version of the POS_PROCESSOR class. It is designed for
            reading in already-processed data that has been stored as xy and possibly xy_2.
            The Moser data is of this form.
            
            ------
            
            You supply a set of optional paramters here and then you get out a function of the form::
            
                process(xy,xy2)
    
            That function will shift the xy data and calculate direction....well, no actually there is no dir yet.
            
        """
        self._ = locals() # TODO: this was a quick hack to convert from the function-factorary paradigm
    def __str__(self):
        return "POS_PROCESSOR_MAT_A with parameters: " + str(self._)
        
    def __call__(self,xy,xy_2):

        # Although we don't need to do that much processing, we make the data look a bit like the axona-style data
        # this allows us to reuse some of the axona processing functions        
        P = _PROC_STATE()
        P.nLEDs = 1 if (xy_2 is None or len(xy_2) is 0) else 2
        P.led_pos = empty((2,P.nLEDs,xy.shape[1]))
        P.led_pos[:,0,:] = xy
        if P.nLEDs == 2:
            P.led_pos[:,1,:] = xy_2
            
        # deal with nans if there are any
        if any(isnan(P.led_pos)):
            P.led_pos = ma.array(P.led_pos,mask=isnan(P.led_pos)) # the _interpAcrossMask function expectes a amsked array, but then unmaskes it following interpolation
            _interpAcrossMask(P)
            
        # Shift to +ve cartesian quadrant
        if self._['offset'] is None:
            P.led_pos -= amin(amin(P.led_pos,axis=2,keepdims=True),axis=1,keepdims=True)
        else:
            P.led_pos += array(self._['offset']).reshape((2,1,1))
    
        # Calcualte/return 
        w = self._['fixedW'] or amax(P.led_pos[0])*1.1
        h = self._['fixedH'] or amax(P.led_pos[1])*1.1
        return P.led_pos[:,0,:],None,None, w,h
            #return xy,dir,dir_disp
            
            
            
"""
What follows is a bunch of smallish functions each of which accepts the posWorking dictionary plus some settings.
Each function may or may not be used at some point in the process pipeline.
"""

def _matchMaskForXY(P):
    """ For each pair of x,y values in the led_pos (and there may be two such pairs per pos sample)
        it checks if either x or y is masked, and if so it masks the pair entirely.
    """
    led_pos,nLEDs = P.led_pos, P.nLEDs
    for led in range(nLEDs):
        bad_xy = any(led_pos.mask[:,led,:],axis=0)
        led_pos[:,led,bad_xy] = ma.masked
                    
 
def _pixToCm(P):
    led_pos,ppm,w,h = P.led_pos,P.ppm,P.w,P.h
    fac =  (100./ppm)
    led_pos *= fac;
    P.w = int(w *fac);
    P.h = int(h *fac);    

def _castPosToFloat(P):
    P.led_pos = ma.array(P.led_pos,dtype=single)


def _windowCoordinates(P):
    """ Constrain the led_pos to lie within the box defined by:
            window_min_x, window_max_x, window_min_y, window_max_y
        Any (x,y) pairs outside the region are masked.
        
        Not sure if its possible for data to lie outside this box, but whatever.
    """
    
    led_pos,posHeader,nLEDs,nPos = P.led_pos,P.posHeader,P.nLEDs,P.nPos
    
    min_x = str2float(posHeader['window_min_x'])
    min_y = str2float(posHeader['window_min_y'])
    P.w = w = str2float(posHeader['window_max_x']) - min_x
    P.h = h = str2float(posHeader['window_max_y']) - min_y
    
    # It seems to be the case that min values are already subtracted
    #led_pos[0,:, :] -= min_x
    #led_pos[1,:, :] -= min_y
    
    bad_x = ((led_pos[0,:, :] < 0) | (led_pos[0,:, :] > w)).reshape(nLEDs,nPos)
    bad_y = ((led_pos[1,:, :] < 0) | (led_pos[1,:, :] > h)).reshape(nLEDs,nPos)
    
    for led in range(nLEDs):
        led_pos[:,led,bad_x[led,:]] = ma.masked
        led_pos[:,led,bad_y[led,:]] = ma.masked
    
    
def _findRect(P,EDGE_PERCENT):
    led_pos = P.led_pos
    
    # put a rough rectangle around the data
    q = (EDGE_PERCENT, 100-EDGE_PERCENT)
    xy = reshape(led_pos[:,0,:].compressed(),[2,-1]) #only use first LED
    box = array(percentile(xy,q,1) ).T
    P.envShape = _SHAPE_RECT(box[0,0],box[0,1],box[1,0],box[1,1])
    
    
def _windowCoordinatesRect(P,fitted_padding):
    """
    Shift min values/clip max values so that centre of rect is at centre 
    of window, and window edge is 2% of width beyond the edge of the circle.
    Values outside the rect are masked.
    """
    led_pos,w,h,box,nLEDs = P.led_pos,P.w,P.h,P.envShape,P.nLEDs

    # Mask pos for each LED pos if x or y is outside box
    for led in range(nLEDs):
        badPos = (led_pos[0,led,:] < box.x1) | (led_pos[0,led,:] > box.x2) 
        badPos |= (led_pos[1,led,:] < box.y1) | (led_pos[1,led,:] > box.y2)
        led_pos[:,led,badPos] = ma.masked
                            
    # shift x and y to centre the box, and define a larger W and H, using fitted_padding
    boxW = box.x2 - box.x1
    boxH = box.y2 - box.y1
    dx = box.x1 - (fitted_padding-1.0)*boxW/2.0
    dy = box.y1 - (fitted_padding-1.0)*boxH/2.0
    led_pos[0,:,:] -= dx
    led_pos[1,:,:] -= dy
    box.shiftXY(-dx,-dy)
    P.w = boxW*fitted_padding
    P.h = boxH*fitted_padding

            
def _findCircle(P,EDGE_PERCENT,CIRCLE_RAD_RES,CIRCLE_ANG_RES,CIRCLE_DWELL_THRESH,CIRCLE_FORCE_RADIUS):
    """
    Fit circle to pos data algorithm by DM. Feb 2014.
    """
    led_pos,w,h,nPos = P.led_pos,P.w,P.h,P.nPos
    
    # First step is to approximate the centre of the circle by putting a rought rectangle around the data
    q = (EDGE_PERCENT, 100-EDGE_PERCENT)
    xy = reshape(led_pos[:,0,:].compressed(),[2,-1]) #only use first LED
    rect = percentile(xy,q,1) 
    xy_c = mean(rect,axis=0) 
    
    # Now that we have an apprixmate centre, we divide the environment up into small arcs, centred on this point.
    # The width and length of the arcs is defined by CIRCLE_RAD_RES and CIRCLE_ANG_RES respectively.
    xy = xy - reshape(xy_c,[2 ,-1])
    r = hypot(xy[0,:],xy[1,:])
    th = arctan2(xy[1,:], xy[0,:]) #not sure igf that's right
    rInd = (r/CIRCLE_RAD_RES).astype(int)
    thInd = ((th+pi)* 180/pi /CIRCLE_ANG_RES).astype(int)
    nTh = int(360.0/CIRCLE_ANG_RES)
    clip(thInd,0,nTh-1,out=thInd)    
    
    # For each angle we find the furthest out bin with more than CIRCLE_DWELL_THRESH fraction of the dwell time
    totalTimeInSeg = accumarray((rInd,thInd), 1, sz=[max(rInd)+1,nTh])
    totalTimeInSeg = totalTimeInSeg[::-1,:] # revrse by r to for argmax search...
    segR = argmax(totalTimeInSeg > CIRCLE_DWELL_THRESH * nPos,axis=0)
    segR = (totalTimeInSeg.shape[0]-segR) * CIRCLE_RAD_RES
    
    # We now a set of points that roughly form a circle. So it should be easy to fit an actual circle..
    segTh = arange(totalTimeInSeg.shape[1])*(CIRCLE_ANG_RES*pi/180) -pi
    x,y = segR * cos(segTh) +xy_c[0], segR * sin(segTh)+xy_c[1]
    xc,yc,r = fitCircle(x,y)
    
    if CIRCLE_FORCE_RADIUS is not None:
        r = CIRCLE_FORCE_RADIUS
        
    P.envShape = _SHAPE_CIRC(xc,yc,r,x,y)
    
def _windowFixedWH(P,fixedW,fixedH):
    """
    if fixedW and fixedH are not none, the existing data will be shifted to centre
    it in the box defined by fixedW and fixedH. Outside this range it will be masked.
    W and H will be set to fixedW and fixedH.
    
    You can actually specify netiher, one, or both of fixedW and fixedH.
    """

    led_pos,nPos,nLEDs = P.led_pos,P.nPos,P.nLEDs
    
    badPos = zeros((nLEDs,nPos),dtype=bool)
    dx = dy = 0
    if fixedW is not None:
        dx = (P.w - fixedW)/2.
        led_pos[0,:,:] -= dx
        P.w = fixedW
        for led in range(nLEDs): 
            badPos[led,:] |= (led_pos[0,led,:] < 0) | (led_pos[0,led,:] >= fixedW)

    if fixedH is not None:
        dy = (P.h - fixedH)/2.
        led_pos[1,:,:] -= dy
        P.h = fixedH
        for led in range(nLEDs):
            badPos[led,:] |= (led_pos[1,led,:] < 0) | (led_pos[1,led,:] >= fixedH)
        
    for led in range(nLEDs):
        led_pos[:,led,badPos[led,:]] = ma.masked
        
    P.envShape.shiftXY(-dx,-dy)
    
def _windowCoordinatesCirc(P,circ_padding):
    """
    Shift min values/clip max values so that centre of circle is at centre 
    of window, and window edge is 2% of radius beyond the edge of the circle.
    Values outside the circle are masked.
    """
    led_pos,w,h,circ,nLEDs = P.led_pos,P.w,P.h,P.envShape,P.nLEDs
                    
    posR2 = (led_pos[0,:,:]-circ.cx)**2 + (led_pos[1,:,:]-circ.cy)**2
    badPos = posR2 > circ.r**2       
    for led in range(nLEDs):
        led_pos[:,led,badPos[led,:]] = ma.masked
        
    dx = circ.cx - circ_padding*circ.r
    dy = circ.cy - circ_padding*circ.r
    led_pos[0,:,:] -= dx
    led_pos[1,:,:] -= dy
    P.w = 2.0*circ_padding*circ.r
    P.h = 2.0*circ_padding*circ.r
    circ.shiftXY(-dx,-dy)
    
def _findShrunkenBigLed(P):
    """ For each sample, it checks if size of big light is closer to that of 
        the small light (as Z score). Returns a logical array, length nPos.
    """
    led_pix = P.led_pix
    mean_npix = led_pix.mean(axis=1)
    std_npix = led_pix.std(axis=1)
    z11 = (mean_npix[0] - led_pix[0,:]) / std_npix[0]
    z12 = (led_pix[0,:] - mean_npix[1]) / std_npix[1]
    P.shrunkenBig = z11 > z12
   
def _findCrossovers(P,SWAP_THRESH):
    """
    Find where:
    
    # Big LED is significantly closer to small LED's previous position than 
    big LED's previous position. ..or..
    # Small LED is significantly closer to big LED's previous position than 
    small LED's previous position.
    
    I thinks that's basically what it's doing, right?
    """
    led_pos = P.led_pos
    
    # Calculate Euclidean distances from one or other LED at fist time point 
    # to one or other LED at second time point
    dist12 = hypot(led_pos[0,0,1:]-led_pos[0,1,:-1], led_pos[1,0,1:]-led_pos[1,1,:-1])
    dist11 = hypot(diff(led_pos[0,0,:]),diff(led_pos[1,0,:]))
    dist21 = hypot(led_pos[0,1,1:]-led_pos[0,0,:-1], led_pos[1,1,1:]-led_pos[1,0,:-1])
    dist22 = hypot(diff(led_pos[0,1,:]), diff(led_pos[1,1,:]))
    
    isCrossover = zeros(shape(led_pos)[-1],dtype(bool))
    isCrossover[:-1] = (dist11 - dist12 > SWAP_THRESH) | (dist22 - dist21 > SWAP_THRESH)
    isCrossover[-1] = False
    P.crossovers = isCrossover
                            
def _ledSpeedFilter(P,MAXSPEED):
    '''filters for impossibly fast tracked points, separately for each LED'''

    led_pos,ppm,sampRate,nLEDs = P.led_pos,P.ppm,P.posSampleRate,P.nLEDs

    max_ppm_per_sample = MAXSPEED * ppm / sampRate
    max_ppms_sqd = max_ppm_per_sample ** 2
    for led in range(nLEDs):
        xVals = led_pos.data[0,led,:].squeeze();
        yVals = led_pos.data[1,led,:].squeeze();
        isBad = any(led_pos.mask[:,led,:],axis=0).squeeze();
        jmp = jumpLoopProc(xVals,yVals,isBad,max_ppms_sqd) #this is a Cython-compiled function which takes ~0.1s for 2LEDs/20min rather than 1s
        led_pos[:,led,jmp] = ma.masked         
            

def _interpAcrossMask(P):
    '''        
    does a basic linear interpolation over missing values in the led_pos masked 
    array and returns the unmasked result
    '''
    led_pos,nLEDs = P.led_pos,P.nLEDs

    missing = reshape(led_pos[0,:,:].mask,[nLEDs,-1]) # we should have x-y missing matching
    ok = reshape(~missing,[nLEDs,-1])

    for led in range(nLEDs):
        ok_idx, = ok[led,:].nonzero() #gets the indices of ok poses
        missing_idx, = missing[led,:].nonzero() #get the indices of missing poses
        ok_data = led_pos.data[:,led,ok_idx]

        # separtely for x and y, take the ok_idx and the ok_data and fill in the data for missing_idx
        # (note that unlike matlab np's interp automatically extrapolates at the edges, using repeated value)
        led_pos.data[0,led,missing_idx.ravel()] = interp(missing_idx,ok_idx,ok_data[0,:])
        led_pos.data[1,led,missing_idx.ravel()] = interp(missing_idx,ok_idx,ok_data[1,:]) 
    
    P.led_pos = led_pos.data #unmask the array
    
def _getLEDWeights(P):
    """ counts the number of masked pos in each led and defines the ratio across LEDs as the weighting.
    """
    weights = ma.count(P.led_pos[0,:,:],axis=-1)
    P.weights = weights.astype(float) / sum(weights)
    

def _combineLEDs(P):
    nLEDs, led_pos, weights, nPos = P.nLEDs, P.led_pos, P.weights,P.nPos
    if nLEDs == 1:
        P.xy = led_pos[:,0,:].squeeze()
    else:
        P.xy = xy = empty([2,nPos]);
        xy[0,:] = reshape(weights,[1,nLEDs]).dot(led_pos[0,:,:].squeeze())
        xy[1,:] = reshape(weights,[1,nLEDs]).dot(led_pos[1,:,:].squeeze())
            
def _smooth(P,SMOOTH_MS):
    """
    Does a boxcar smoothing of ``led_pos`` using a width of ``SMOOTH_MS``-miliseconds.        
    
    ``led_pos`` must be a non-masked array.
    """
    led_pos, posSampleRate = P.led_pos,P.posSampleRate
    wid = int(posSampleRate * SMOOTH_MS/1000);
    
    kern = ones(wid)/wid
    P.led_pos = sp.ndimage.filters.convolve1d( \
                        led_pos, kern, mode='nearest', axis=-1 ) # the "nearest" means repeated values at the ends
    
    
    
class _SHAPE():
    """ A small class for holding info on a shape and alowing you to shift it around easily.
        
        Constructor
        ----
        
        * ``shapeType`` custom string for identifying the shape type
        
        * ``xDict``, ``yDict`` are dictionaries with attribute names as
          keys and scalars or np arrays as values.  Calling ``shiftXY(dx,dy)``
          updates all the values in each dict.
            
        * ``lenDict`` is similar to ``xDict`` and ``yDict`` but it specifies values
          that scale with length rather than specify coordinates, e.g. widths and radii.

        * All other key/values will be added to the SHAPE instance as attributes
        
        
        Methods
        ----
        
        * ``shiftXY(dx,dy)`` updates the x and y values.

        * ``makeMask(W,H,binSize)`` - produces a mask of Trues outside the shape
          and False inside the shape.  Values are converted to bins using floor(val/binSize).
          W and H specify the full width and height of the region, ie. they must be
          converted to bin units to get the size of the mask.
        
        Subclasses
        ----
        ``_SHAPE_CIRC``, ``_SHAPE_RECT``
        
        TODO: implement the scaling thing. May also want a plot function.
        
        Example::
            
            >>> myCirc = _SHAPE("simpleCirc",xDict={'cx':8},yDict={'cy':3},somethingElse="hello")
            >>> print myCirc.cx, myCirc.somethingElse
             8 hello
            >>> myCirc.shiftXY(3,2)
            >>> print myCirc.cx
             11     
    """
    def __init__(self,shapeType,xDict=None,yDict=None,lenDict=None,**kwargs):
        """
        Note we dont check for duplicating of keys across dicts, this will cause a bug.
        """
        self.shapeType = shapeType
        self._yAttrs = self._xAttrs = self._lenAttr = ()
        if xDict is not None:
            self._xAttrs = xDict.keys()
            self.__dict__.update(xDict)
        if yDict is not None:
            self._yAttrs = yDict.keys()
            self.__dict__.update(yDict)
        if lenDict is not None:
            self._lenAttrs = lenDict.keys()
            self.__dict__.update(lenDict)            
        if kwargs is not None:
            self.__dict__.update(kwargs)                
                
    def shiftXY(self,dx,dy):
        for attr in self._xAttrs:
            v = getattr(self,attr)
            v += dx
            setattr(self,attr,v)
            
        for attr in self._yAttrs:
            v = getattr(self,attr)
            v += dy
            setattr(self,attr,v)
    
    def makeMask(self,W,H,binSize,inner_boundary=None):
        """ Normally the mask is True outside the shape and False inside.
        However, when inner_boundary > 0, we create a label matrix not
        a mask.  The region outside the shape is labeled 0, and inside the
        shape is labeled 1 in the outer anulus of width inner_boundary cm
        and labeled 2 inside that anulus.
    
        TODO: check this and all subclass versions are correct.       """
        W,H,binSize= float(W),float(H),float(binSize)
        m = zeros([int(ceil(W/binSize)),int(ceil(H/binSize))],dtype=bool) 
        if inner_boundary:
            raise NotImplementedError()
        return m
        
    def distToBoundary(self,xy,):
        """ For an array of 2xn, returns the distance to the boundary for
        each value.        """
        raise NotImplemented()
        
    def isOutside(self,x,y):
        """For a point, (x,y) in cm, it returns True if the point is outside
        the shape, otherwise False.        """
        return False
        
    def plot(self,W,H,binSize,inner_boundary=None):
        m = self.makeMask(W,H,binSize,inner_boundary=inner_boundary)
        if inner_boundary is None:
            m = ma.array(m,mask=~m)
        else:
            m = ma.array(m,mask=m==0)
        plt.imshow(m.T,alpha=0.5,extent=[0,W,0,H],interpolation='nearest')
        
class _SHAPE_CIRC(_SHAPE):
    def __init__(self,cx,cy,r,px,py):
        _SHAPE.__init__(self,"circ",xDict={'cx':cx,'px':px},yDict={'cy':cy,'py':py},lenDict={'r':r})
        
    def makeMask(self,W,H,binSize,inner_boundary=None):
        W,H,binSize= float(W),float(H),float(binSize)
        x = floor(abs((arange(0,W,binSize)+0.5 - self.cx)/binSize))*binSize
        y = floor(abs((arange(0,H,binSize)+0.5 - self.cy)/binSize))*binSize
        
        r_now = self.r
        m = (x*x)[:,newaxis] + (y*y)[newaxis,:] >= r_now**2
        if inner_boundary is not None:
            if inner_boundary < 0:
                raise Exception("inner_boundary cannot be negative.")
            m = (~m).astype(uint8)
            r_now -= inner_boundary
            m += (x*x)[:,newaxis] + (y*y)[newaxis,:] <= r_now**2
        return m
        
    def distToBoundary(self,xy):
        """For a circle, the distance to the boundary is radius-dist_to_centre.
        """
        return self.r - hypot(xy[0] - self.cx, xy[1] - self.cy)

    def isOutside(self,x,y):
        return (x-self.cx)**2 + (y-self.cy)**2 > self.r**2
        
    def plot(self,W=None,H=None,binSize=None,showMask=False,inner_boundary=None):
        """
        plots a circle using the cx,cy and r attribues.
        If showMask is True, you must give W,H and binSize and it will
        also plot a mask that is transparent inside the shape and partially opaque
        outside.
        """
        if showMask:
            _SHAPE.plot(self,W,H,binSize,inner_boundary=inner_boundary)
        plt.axis('equal')
        plt.gca().add_patch(plt.Circle((self.cx,self.cy),radius=self.r, 
            color=[0,0,0],linewidth=2,linestyle='solid',alpha=0.8,fill=False))
        if inner_boundary:
            plt.gca().add_patch(plt.Circle((self.cx,self.cy),radius=self.r-abs(inner_boundary), 
                color=[0,0,0],linewidth=2,linestyle='solid',alpha=0.8,fill=False))
        plt.gca().relim()  
            
        
        
class _SHAPE_RECT(_SHAPE):
    def __init__(self,x1,x2,y1,y2):
        _SHAPE.__init__(self,"rect",xDict={'x1':x1,'x2':x2},yDict={'y1':y1,'y2':y2})
        
    def makeMask(self,W,H,binSize,inner_boundary=None):
        binSize = float(binSize)
        m = _SHAPE.makeMask(self,W,H,binSize)
        
        m[:int(floor(self.x1/binSize)),:] = True
        m[int(ceil(self.x2/binSize)):,:] = True
        m[:,:int(floor(self.y1/binSize))] = True
        m[:,int(ceil(self.y2/binSize)):] = True
        if inner_boundary is not None:
            if inner_boundary < 0:
                raise Exception("inner_boundary cannot be negative.")
            m = (~m).astype(uint8)
            
            m[  int(ceil((self.x1+inner_boundary)/binSize)) : \
                int(ceil((self.x2-inner_boundary)/binSize)),  \
                int(ceil((self.y1+inner_boundary)/binSize)) : \
                int(ceil((self.y2-inner_boundary)/binSize)) ] += 1            
        return m
        
    def distToBoundary(self,xy):
        return np.minimum(\
                    np.minimum(np.abs(xy[0]-self.x1),np.abs(xy[0]-self.x2)),
                    np.minimum(np.abs(xy[1]-self.y1),np.abs(xy[1]-self.y2)) )

    def isOutside(self,x,y):
        return not (self.x1 < x < self.x2 and self.y1 < y < self.y2)
        
    def plot(self,W=None,H=None,binSize=None,showMask=False,inner_boundary=None):
        """
        plots a circle using the cx,cy and r attribues.
        If showMask is True, you must give W,H and binSize and it will
        also plot a mask that is transparent inside the shape and partially opaque
        outside.
        """
        if showMask:
            _SHAPE.plot(self,W,H,binSize,inner_boundary=inner_boundary)
        else:
            plt.xlim(0,W)
            plt.ylim(0,H)
        plt.axis('equal')
        plt.gca().add_patch(plt.Rectangle((self.x1,self.y1),width=self.x2-self.x1,height=self.y2-self.y1, 
            color=[0,0,0],linewidth=2,linestyle='solid',alpha=0.8,fill=False))
        if inner_boundary:
            plt.gca().add_patch(plt.Rectangle((self.x1+inner_boundary,self.y1+inner_boundary),
                                width=self.x2-self.x1-2*inner_boundary,
                                height=self.y2-self.y1 -2*inner_boundary, 
                color=[0,0,0],linewidth=2,linestyle='solid',alpha=0.8,fill=False))
        plt.gca().relim()  

    
"""           

Remaining stuff from Robin's code not yet incorportated here....
(just dir)            
"""


#
#    self.dir[0,:-2] = np.mod(180/math.pi *np.arctan2(-self.xy[1,1:] + self.xy[1,:-2],
#                                                     +self.xy[0,1:] - self.xy[0,:-2]), 360)
#    self.dir[0,-1] = self.dir[0,-2]
#    self.dir_disp = self.dir
#elif self.nLEDs == 2:
#    lightBearings = np.zeros([2,1])
#    lightBearings[0] = str2float(setHeader['lightBearing_1'])
#    lightBearings[1] = str2float(setheader['lightBearing_2'])
#    #TODO: should try and guess correction from the data based on sections where animal is running in a straight line.
#    front_back_xy_sm = np.zeros([4,self.npos])
#    for i in range(4):
#        front_back_xy_sm[i,:] = sm.smooth(led_pos[i,:],BOXCAR,'flat')
#    correction = lightBearings[0]
#    self.dir[0,:] = np.mod(180/math.pi * (np.arctan2(-front_back_xy_sm[1,:] + front_back_xy_sm[3,:],
#                                                     +front_back_xy_sm[0,:] - front_back_xy_sm[2,:]) -correction),360)
#                      
#self.dir_disp[0,pos2] = np.mod(((180/math.pi) * (np.arctan2(-self.xy[1,pos2+1] + self.xy[1,pos2],+self.xy[0,pos2+1]-self.xy[0,pos2]))) ,360)
#self.dir_disp[0,-1] = self.dir_disp[0,-2]

    
