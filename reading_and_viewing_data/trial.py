# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:06:15 2014

@author: daniel
"""

#from __future__ import division #TODO: update code to sensible division rules
import numpy as np
import scipy as sp
import glob
import os
import re
from custom_casting import str2float,str2int
from pos_post import POS_PROCESSOR, POS_PROCESSOR_MAT_A
from custom_dict import LazyDict, LazyDictProperty
from dateutil import parser as dateparser
import mat_a
from trial_basic_analysis import TT_BASIC_ANALYSIS
from trial_gc import TT_GC
from trial_freq import TT_FREQ
from trial_phase_precession import TT_PHASE_PRECESSION
from readDACQfile import _readFile
from warnings import warn
import warnings

from custom_sp import matlabfriendly
from trial_ball import TT_BALL

PICKLE_FOLDER = 'generated'
POS_NAN = 1023
defaultPosProc_axona = POS_PROCESSOR() #If you want to apply different processing for different trials you create two or more of your own and assign them to the relevant trials
defaultPosProc_mat_a = POS_PROCESSOR_MAT_A()

class TT(object,TT_BASIC_ANALYSIS,TT_GC,TT_FREQ,TT_PHASE_PRECESSION,TT_BALL):
    """
    This class is for loading data for a single trial and perfoming analysis on it.
    It can/will eventually be able to load all files associated with a DACQ trial.
    It also at partly supports loading of data in other formats.

    Loading is done lazily, i.e. individual files are not read until you call a method that needs to read them.
    The relevant file is then read, parsed, cached in the class's relevant map and then returned to the caller.
    
    The base TT class is augmented with "mixin" classes, which are split across
    several files with names like trial_plot.py and trial_gc.py, trial_freq.py etc.
        
    
    Constructor
    -----------
    TT(fileName) :
            create a TT instance from the given fileName, which is the full path, optionally including extension


    optional args, some of which can be set after init, i.e. there are similarly named properties with setter functions...
    
    * ``date`` - a timetuple. 
    If you don't provide this then the date propertey will try looking up the date in the setHeader.

    * ``posProcessor`` POS_PROCESSOR instance from pos_post.py 
    If ``mode`` is not ``'axona'`` you may need to use a different kind of POS_PROCESSOR such as 
    POS_PROCESSOR_MAT_A (also in pos_pos.py). If you leave this as ``None`` the default POS_PROCESSOR will be used for
    the given mode. Node that this default is a singleton, i.e. it will be shared between multiple TT instances (but
    this shouldn't have any noticable impact because you should never modify a pos proc after creating it.)
    
    * ``cutFilePattern`` - a string including one or more of the following tokens:
    ``%PATH%``, ``%EXPERIEMENT%`` and ``%TET%``. The tokens will be replaced with their respective values
    in order to obtain the correct cutfile name.  If you leave the cutFilePattern as ``None``
    the cut file will be chosen according to the logic within ``availableCuts`` instead.
    
    * ``info`` -  this can be anything..probably expected to be a dict, for holding custom metadata for the trial.

    Supported modes
    ----------
    ``mode`` is detected from the fileName at construction.  It will take one of the following values:   
    
    * ``'axona'`` (internally represented as ``self._mode=None``)
       files use the `'fileName'` + `'.set'` `'.pos'` `'.eeg'` `'.1'` `'.2'` `'.3'` etc.   and are in axona format.
       the ``fileName`` provided at construction should end with a `".set"` (or anything that doesn't match one of the other modes here).   
       
    * ``mat_a`` all files are matlab `.mat` files. 
                There is no set file, pos and eeg are `'fileName` + `'_POS'` and + `'_EEG'`
                There are no tetrode files, rather each tet-cell has a it's own file names as `'fileName'` + `'_T#C#'`.
                TODO: support _EGF and EG2 etc. Support for this mode may be a little buggy. This mode is detected when you pass
                an ``fn`` that ends with ``"_pos.mat"``.  This is what you need for loading Moser data.
                
                
    High level analysis 
    -------------        
    Most analysis exists as at least two methods, one called ``plotSomething`` and    
    another called ``getSomething``. Both should take roughly the same inputs, and the
    ``plotSomething`` method should call the ``getSomethign`` method and then display
    a nice plot of the results.  In most cases you can also provide some pre-computed 
    results to the plot function and get it to plot them.
    
    Here is a selection of *some* of the available plotting functions:
    
        ``plotAutoCorrT``, ``plotGCmeasuresWithShuffles``, ``plotPosWithGCFields``, ``plotSpeedFreq``
    
    You also get an easy way to run KlustaKwik:
    
        ``runKlustakwik``
    
    
    The low-ish level methods:
    -------------
        tetTimes(t,[c],[asType]) : 
            returns times for all spikes or for spikes of given cell
        tetWaves(t,[c]) :
            returns waveforms for all spikes or for spikes of given cell
        tetCut[t] : 
            returns the cut for the given tetrode    
        thetaAmp, thetaPhase, thetaFreq :
            returns the amplitude of theta obtained via the Hilbert transform on filtered eeg
        xy :
            return ``[2,nPos]`` numpy array of position
            
    To change what cut is loaded you need to set ``.cutFileNames[3] = "full of file tet3.cut"``. 
    ...TODO: check whether this works. I think you actually need to assign the cutFilenames dict
    back to the TT in order to trigger an update, although having said that there doesn't seem
    to be a setter...so basically you probably have to clear the cache in order to force and update.
    
    To change what eeg file is laoded you need to set ``.eegFileName = "some other file.eeg"``
    
    To change the band on which eeg is filtered set ``.thetFreqBand = [3,9]``
    
    To change the way in which pos data is filtered, you need to create a posProcessor using
    ``pos_post.buildPosProcessor``, and then assign it to the trial with ``.posProcessor=mySpecialProcessor``
    
    You can also set a rotation of pos. By changing ``.rot`` to some value in degrees (rather than ``None``).
    The rotation is computed form the raw xy each time you access ``xy`` (TODO: cache it).
    
    If you are working with several trial instances, you will probably run in to problems with memory, becuase
    each trial instance will have cached more than neccessary. You can do ``.clearCaches()`` to get rid
    of stuff that could, if enccessary, be re-loaded from file.

    Convention
    ----------
    * Attributes called ``_loadSomething`` will read from disk and store the result
      in one or more of the instance's ``_something`` variables.  
    
    * Attribues called ``getSomething`` will read from the ``_something`` variables
      and return a result without storing it.  If needed they will call the ``_loadSomething``
      method.
    
    * Attribues called ``something``, i.e. ones that have neither ``load`` nor ``get`` 
      in the name, will generally load and/or get something and return it, the difference
      is that they can be used as properties without ``()``...although that's not always true.

    * Methods called ``plotSomething`` will have a sister function ``getSomething`` which
      they pass their arguments to and then plot the result.
    
    * If tetrode and cell are in the argument list they should be first and second and 
      called ``t`` and ``c``.  All ``plot`` functions should accept a ``t`` and ``c``, even if
      they ignore  it...it makes it easier to plot different things.
    
    * Rather than duplicating get or plot functions, as far as is reaonsable, some 
      effort should be made to generalise existing functions to accept masks, or extra
      plotting options.  In such cases it is probably better to part implement the generalisation
      than to copy the old function.
    
    This convention probably isn't fully observed though.
    
    TODOs
    --------------    
    * allow class to handle trials that have been split into multiple parts,
      i.e. it should accept multiple file names and concatenate data as expected.        
      
      
    * this file has become rather large.  Some of the non-loading stuff could probably be moved
    to one of the mixin claseses..eg. ratemaps could be moved to trial_gc and theta stuff could be
    moved to trial_freq
    """
    def __init__(self,fn,posProcessor=None,info=None,cutFilePattern=None,date=None):
        mode = None
        if fn.lower().endswith(".set"):
            fn = fn[:-4]
            if posProcessor is None:
                posProcessor = defaultPosProc_axona
        elif fn.lower().endswith("_pos.mat"):
            fn = fn[:-8]
            mode = 'mat_a'
            if posProcessor is None:
                posProcessor = defaultPosProc_mat_a
            
        self.fn = fn
        self.path, self.experimentName = os.path.split(fn)
        self._setHeader = None
        self.info = info if info is not None else {}#a dict to be used however the user sees fit
        self._posProcessor = posProcessor
        self.rot = None # if not None will be treated as an angle in degrees to rotate xy about centre
        self.jitter = None # if not None will be treated as a cm offset to add to xy data (after any rotation)
        self._mode = mode
        self._date = date
        
        # This list is populated by reading the file system, this is done on request
        # as part of cut and eeg things
        self._trialFileNames = None       
       
        #set if and when we load pos data
        self._clearCachePos(headerAlso=True)
        
        # These things are maps with tetrode numbers as keys, the maps are only populated on request
        self._tetTimes = None
        self._tetWaves = None
       
        self._availableCuts = None # this is a dict with tet nums as keys, and values being lists of the full file paths
        self._availableTets = None # this is a dict with tet nums as keys, and True for values
        self._cutFileNames = None #this will be initialised with the most recent cut for each tetrode, but can be modified
        self._cutFilePattern = cutFilePattern # this is a string of the form r"%PATH%\%EXPERIMENT%_bestest_%TET%.cut", 
                                    
        
        # set if and when we load eeg data
        self._availableEegs = None
        self._eegFile = None # gives the name of the file from which we laoded the eeg. When it is changed all eeg stuff is set to None        
        self._eeg = self._eegHeader = self._filteredEeg = self._thetaAmp = self._thetaPhase = self._thetaFreq = None
        self._thetaFreqBand = [6,12] # when this is changed we set theta stuff to None
        
        # set if and when we load ball data
        self._xyBall = self._ballHeader = None
        
    def __repr__(self):
        return "Trial [%s] {TT instance}" % (self.experimentName)
        
    @property
    def date(self):
        if self._date is None:
            self._date = dateparser.parse(self.setHeader['trial_date']).timetuple()
        return self._date
        
    @property
    def mode(self):
        return 'axona' if self._mode is None else self._mode
        
    @property
    def setHeader(self):
        if self._setHeader is None:
            self._loadSet()
        return self._setHeader
    
    @property
    def duration(self):
        if self._mode is None:
            return str2float(self.setHeader['duration'])
        elif self._mode == 'mat_a':
            return str2float(self.posHeader['duration'])
        
    @property 
    def nLEDs(self):
        if self._mode is None:
            hd = self.setHeader
            return int(sum( [str2float(hd['colactive_' + x]) for x in ['1','2']]))
        elif self._mode == 'mat_a':
            return str2int(self.posHeader['nLEDs'])
            
    @property
    def posHeader(self):
        if self._posHeader is None:
            self._loadPos(headerOnly=True) 
        return self._posHeader
        
    @property
    def nPos(self):
        if self._nPos is None:
            self._nPos = str2int(self.posHeader['num_pos_samples'])
        return self._nPos
        
    @property
    def posSampRate(self):
        return str2float(self.posHeader['sample_rate'])
        
    @property
    def availableTets(self):
        if self._availableTets is None:
            self._findAvailableTets()
        return self._availableTets

    @property
    def availableCuts(self):
        if self._availableCuts is None:
            self._findAvailableCuts()
        return self._availableCuts
        
    @property
    def availableEEGs(self):
        if self._availableEegs is None:
            self._findAvailableEegs()
        return self._availableEegs
        
    @property
    def eegHeader(self):
        if self._eegHeader is None:
            self._loadEeg(self.eegFileName,headerOnly=True)
        return self._eegHeader

    @property
    def eegSampRate(self):
        return str2float(self.eegHeader['sample_rate'])
    
    @property
    def eeg(self):
        if self._eeg is None:
            self._loadEeg(self.eegFileName)
        return self._eeg
        
    @property
    def eegTime(self):
        sampRate = self.eegSampRate
        nEeg = str2int(self.eegHeader['num_EEG_samples'])
        return np.arange(0,nEeg/sampRate,1./sampRate)
        
    @property
    def cutFilePattern(self):
        return self._cutFilePattern
        
    @cutFilePattern.setter
    def cutFilePattern(self,value):
        if self._cutFilePattern == value:
            return
        self._cutFilePattern = value
        #invalidate any existing cut file names and cut loaded cut files
        self._cutFileNames = None
        self.tetCut._cache.clear()
        
    @property
    def cutFileNames(self):
        """returns a dictionary with tetrode numbers as keys, one for each of
        the availble tet files.  The values give the chosen cut file name for
        the given tetrode.  This choice is by default the most recently modified
        cut file.  But this can be overriden with TT.cutFilePattern. 
        
        Note that you can modify the dictionary returned by this property to 
        change which cut files you want to be using, but if you do so you will
        need to manually clear the tetCut cache or risk using a previously loaded cut::

        >>>TT.tetCut._cache.clear()        
        """
        
        if self._cutFileNames is None:
            self._cutFileNames = {}            
            if self.cutFilePattern is None:                
                
                # If there is no particular required cut file naming pattern
                # then look in the same folder as the set file for cut files
                # that seem to match this experiment.  For each tetrode, take
                # the most recently modified cut file that seems to correspond
                # to that tetrode.
                availableCuts = self.availableCuts
                for t in self.availableTets:
                    if t in availableCuts:
                        self._cutFileNames[t] = availableCuts[t][-1]
                    else:
                        # this file doesn't actually exists, so perhaps could do somehting mroe usefuyl here..?
                        self._cutFileNames[t] = self.fn + "_" + str(t) + ".cut" 
                        
            else:
                
                #If a cut file naming pattern has been provided then apply that pattern, even
                #if no such files exist.
                for t in self.availableTets:
                    self._cutFileNames[t] = self._cutFilePattern.replace("%PATH%",self.path) \
                                                                .replace("%EXPERIMENT%",self.experimentName) \
                                                                .replace("%TET%",str(t))
        return self._cutFileNames
        
    @property
    def posProcessor(self):
        return self._posProcessor        
        
    @posProcessor.setter
    def posProcessor(self,value):
        if self._posProcessor == value:
            return     
        self._posProcessor = value
        self._clearCachePos()
               
    @staticmethod
    def _load_dict_(d):
        warnings.warn("Be careful loading from stored dicts...there may be issues.")
        tt = TT(d['fn'] if isinstance(d['fn'],basestring) else '',
                posProcessor=POS_PROCESSOR(**d['posProcessor']),
                info=d['info'],
                cutFilePattern=d.get('cutFilePattern',None),
                date=d.get('date',None)
            )
        tt.rot = d.get('rot',None)
        tt.jitter = d.get('jitter',None)
        tt._eegFile = d.get('eegFile',None)
        return tt
        
    def _save_dict_(self):
        """
        return a dictionary that can be saved with scipy.io.savemat
        rather than try and use some complicated hacked version of 
        an autogernerated dictionary, we explicitly construct it.
        """
        warnings.warn("We dont record customised cut file choices, only cutFilePattern...and there may be other issues.")
        
        d = dict(
         thetaFreqBand= self._thetaFreqBand,
         posProcessor= matlabfriendly(self.posProcessor._), # this is a dict of the arguments used to initialise the posprocessor
                                     #..it's a bit wasteful if laods of TTs share the same pos proc, but the data will be samll so it's not really
                                     #a big issue, and compression within the mat file may take care of it.
         fn=self.fn,
         info= matlabfriendly(self.info)
        )
        if self.rot is not None:
            d['rot'] = self.rot
        if self.jitter is not None:
            d['jitter'] = self.jitter
        if self._date is not None:
            d['date'] = matlabfriendly(self._date)
        if self.cutFilePattern is not None:
            d['cutFilePattern'] = self.cutFilePattern
        if self._eegFile is not None:
            d['eegFile'] = self._eegFile
        
        return d
        
    def _clearCachePos(self,headerAlso=False):
        """
        See also ``TT.clearCaches``
        
        Because there are so many things to do with pos we collect them 
        all here so that we can selectively clear the pos cache if we need to.
        
        TODO: probably want to do something similar for eeg stuff, and may
        want to provide granularity in ``TT.clearCaches``, allowing us to
        keep pos stuff, but ditch spike stuff (which is probably larger, but
        easier to reload).
        """
        self._xy = self._dir = self._dir_disp = self._speed = self._pos_dist_to_boundary = \
            self._w = self._h = self._posShape = self._pathLen = self._nPos = None
        if headerAlso:
            self._posSampleRate = self._posHeader = None
        
    @property
    def pathLenCum(self):
        """
        Returns the cumulative path length in the same units as xy (which should be cm).
        The array will be nPos long (we pad at the start with one zero).
        """
        if self._pathLen is None:
            self._pathLen = np.cumsum(np.hypot(\
                        np.ediff1d(self.xy[0],to_begin=0),
                        np.ediff1d(self.xy[1],to_begin=0)  ))
        return self._pathLen
        
    @property
    def pos_dist_to_boundary(self):
        """ Returns boolean vector of length nPos, with True where pos
        is within dist_to_boundary of the boundary and False elsewhere.
        """
        if self._pos_dist_to_boundary is None:
            xy = self.xy
            self._pos_dist_to_boundary = self._posShape.distToBoundary(xy)
        return self._pos_dist_to_boundary
        
    @property
    def xy(self):
        #TODO cache ret following rot and jitter and clear with other pos stuff
    
        if self._xy is None:
            self._loadPos()
        if self.rot is None:
            ret = self._xy
        else:
            ret = _rotateXY(self._xy,self._w,self._h,self.rot)
        if self.jitter is None:
            return ret
        else:
            return _jitterXY(ret,self._w,self._h,self.jitter) 
        
    def xy_bin_inds(self, binSizeCm=2.0,xy=None):
        """Converts xy in cm to xy in bins. Returns `(xyBinInds, nBinsW, nBinsH)`.
        """
        xy = self.xy if xy is None else xy
        return (xy/binSizeCm).astype(int), \
               int(np.ceil(self._w/float(binSizeCm))), \
               int(np.ceil(self._h/float(binSizeCm)))
        
    @property
    def dir(self):
        if self._xy is None:
            self._loadPos()
        ret = self._dir
        if self.rot is not None:
            raise NotImplementedError("please code rotation of dir data")
        return ret
        
    @property
    def speed(self):
        """
        Returns speed in cm/s
        """
        if self._speed is None:
            diff_xy = np.diff(self.xy,axis=1)
            if len(diff_xy):
                self._speed = np.hypot(diff_xy[0,:],diff_xy[1,:])
                self._speed = np.append(self._speed,[0])
                self._speed = self._speed * self.posSampRate
            else:
                self._speed = np.array([])
        return self._speed

    @property        
    def dir_disp(self):
        """
        Calculates directional of displacement, i.e. the direction of change
        in xy.  Where speed is low this directional estimate is not valid.
        
        These values are a bit confused, I think the sign of dy needs to be changed.
        We compensate for this in _getDirRatemap in order to match Tint.
        """
        if self._dir_disp is None:
            xy = self.xy
            self._dir_disp = np.arctan2(np.ediff1d(xy[0,:],to_end=[0]),np.ediff1d(xy[1,:],to_end=[0]))
        return self._dir_disp


    @property
    def eegFileName(self):
        if self._eegFile is None:
            self._eegFile = self.availableEEGs[0]
        return self._eegFile
    
    @eegFileName.setter
    def eegFileName(self,value):
        if self._eegFile == value:
            return
        self._eegFile = value
        self._eeg = self._eegHeader = self._filteredEeg = self._thetaPhase \
        = self._thetaAmp = None # invlaidate the old ones (if they exist) 
 
    @property
    def thetaFreqBand(self):
        return tuple(self._thetaFreqBand) # tuple because we don't want it to be mutable -use the setter to change the value
        
    @thetaFreqBand.setter
    def thetaFreqBand(self,val):
        self._thetaFreqBand = val
        self._filteredEeg = self._thetaAmp = self._thetaPhase = self._thetaFreq = None #invalidate theta stuff
    
    @property
    def thetaPhase(self):
        if self._thetaPhase is None:
            self._transformEeg()            
        return self._thetaPhase

    @property
    def thetaAmp(self):
        if self._thetaAmp is None:
            self._transformEeg()            
        return self._thetaAmp
        
    @property
    def thetaFreq(self):
        if self._thetaFreq is None:
            self._thetaFreq = np.append(np.diff(self.thetaPhase),[0]) * (self.eegSampRate/2/np.pi)
        return self._thetaFreq
        
    @property
    def filteredEeg(self):
        if self._filteredEeg is None:
            self._filteredEeg = _eegfilter(self.eeg,self.eegSampRate,self._thetaFreqBand)
        return self._filteredEeg
                        
        
    def tetTimes(self,t,c=None,asType='s',acceptable_bleed_time=1.0,min_spike_count=100):
        """
        Gets the times for tetrode number ``t``.  If ``c`` is not none, it
        returns only the times for the cell on the given tetrode.
        
        ``asType`` can take one of the following values:
        
        * ``'x'`` - do not convert, i.e. units are given by ``self.tetTimebase[t]``
        * ``'s'`` - seconds (returned as double,all others are ints)
        * ``'p'`` - pos index
        * ``'e'`` - eeg index
        
        ``acceptable_bleed_time`` - a value in seconds.  If the last few spikes times 
        are beyond the end of the trial by less than this amount they will silently
        be removed.  If they go beyond the end of the trial by more than this amount
        an error will be thrown.   (Set to 0/Infinity to throw error for any bleed or
        swallow all extra spikes.)
    
        ``min_spike_count`` - if the number of spikes is less than this an
        error will be thrown.
        """
        
        if self._mode is None:
            if t not in self.availableTets:
                raise Exception("Did not find tet %d for file %s" % (t,self.fn))
                
            if self._tetTimes is None or t not in self._tetTimes:
                self._loadTet(t)
                
            times = self._tetTimes[t]       
            timebase = self.tetTimebase[t]
            if c is not None:
                cut = self.tetCut[t]
                times = times[cut==c] 
                if len(times) == 0:
                    raise ValueError("No spikes found for cell in cut.")
                if len(times) < min_spike_count:
                    raise ValueError("More than zero, but fewer than min_spike_count spikes found for cell in cut.")
            
        elif self._mode == 'mat_a':
            if c is None:
                raise Exception("In mat_a mode you must specify cell as well as tet to get times.")
            tc_key = 't'+str(t)+'c'+str(c)
            if self._tetTimes is None:
                self._tetTimes = {}
            if tc_key not in self._tetTimes:
                self._tetTimes[tc_key] = mat_a.load_tc(self.fn,t,c)
            times = self._tetTimes[tc_key]
            timebase = 1.0 #times are already in seconds
            
            
        if not len(times):
            return times
            
        # check for spikes that bleed beyond the end of the trial
        if times[-1] > self.duration*timebase:
            # If they only bleed a little beyond the end of the trial we can just discard them, otherwise we should raise an error
            if times[-1] > (self.duration + acceptable_bleed_time)*timebase:
                raise Exception("At least one spike time is more than {:0.2f}s beyond the end of the trial".format(acceptable_bleed_time))
            else:
                times = times[:np.searchsorted(times,self.duration*timebase)]
                    
        if asType == 'x':
            pass
        elif asType == 's':
            times = times.astype(np.double) / timebase
        elif asType == 'p':
            factor =  self.posSampRate/timebase            
            times = (times*factor).astype(int) #note this is rounding down
        elif asType == 'e':
            factor = self.eegSampRate/timebase
            times = (times*factor).astype(int) #note this is rounding down
        else:
            raise Exception("unknown type flag for tetTimes: %s" % (asType))
   
            
        return times

    def tetWaves(self,tetNum,cell=None):
        if self._mode is not None:
            raise Exception("No tet waves for mode=%s" % self._mode)
            
        if tetNum not in self.availableTets:
            raise Exception("Did not find tet %d for file %s" % (tetNum,self.fn))

        if self._tetWaves is None or tetNum not in self._tetWaves:
            self._loadTet(tetNum,waves=True)
            
        waves = self._tetWaves[tetNum]
        if cell is None:
            return waves
        
        cut = self.tetCut(tetNum)
        return waves[cut==cell]
        
        
        
    def _findAvailableCuts(self):
        if self._mode is not None:
            raise Exception("Cannot find available tets for mode=%s" % self._mode)
            
        if self._trialFileNames is None:
            self._findTrialFiles()
            
        cutRe = re.compile('_(\d+)\.cut$|\.clu\.(\d+)$')
        cutFns = {}
        base = os.path.split(self.fn)[0]
        for f in self._trialFileNames:
            m = cutRe.search(f)
            if m is not None:
                if m.group(1) is not None:
                    tet = int(m.group(1)) # _n.cut syntax
                else:
                    tet = int(m.group(2)) # .clu.n syntax
                if tet in cutFns:
                    cutFns[tet].append(base + os.sep + f)
                else:
                    cutFns[tet] = [base + os.sep + f]

        # Note that each list is sorted from oldest to newest
        # because that's how the self._trialFileNames are sorted
        # TODO, may want to do the sorting here instead since its probably redundant
        # for the other files
        self._availableCuts = cutFns
                
    def _findAvailableTets(self):
        if self._mode is not None:
            raise Exception("Cannot find available tets for mode=%s" % self._mode)
            
        if self._trialFileNames is None:
            self._findTrialFiles()
        tetRe = re.compile(re.escape(self.experimentName) + r'\.(\d+)$')
        self._availableTets = {}
        for f in self._trialFileNames:
            m = tetRe.match(f)
            if m is not None:
                self._availableTets[int(m.group(1))] = True
   
    def _findAvailableEegs(self):
        if self._mode is not None:
            raise NotImplementedError("Should be able to find avilable eegs in non-axona mode")
            
        if self._trialFileNames is None:
            self._findTrialFiles()
        eegRe = re.compile(re.escape(self.experimentName) + r'\.(eeg|egf|egg|ee?g\d)$')
        self._availableEegs = []
        for f in self._trialFileNames:
            m = eegRe.match(f)
            if m is not None:
                self._availableEegs.append(f)
        
    def _findTrialFiles(self):
        """
        Finds all files in the same directory as the set file which begin with
        the name of the set file...ie. this should include tet,pos,inp,fet, cut etc.
        and stores them sorted by modified date in self._trialFileNames
        """
        fnames = glob.glob(self.fn + "*") 
        sorted(fnames,key=os.path.getmtime)
        self._trialFileNames = map(lambda f: os.path.split(f)[1], fnames)
        
            
    def _loadSet(self):
        if self._mode is not None:
            raise Exception("No set file for mode=%s" % self._mode)
        self._setHeader, = _readFile(self.fn + ".set",False)        

    def _loadPos(self,returnWorking=False,headerOnly=False):        
        if self._mode is None:
            if headerOnly:
                 self._posHeader, = _readFile(self.fn + ".pos",None,True)
                 return
                 
            header,data = _readFile(self.fn + ".pos",[('ts','>i'),('pos','>8h')])
            self._posHeader = header
    
            if not header:
                self._xy, self._dir, self._w, self._h = np.array([]),np.array([]), np.nan,np.nan
                self._xy.shape = (0,0)
                return
                
            posSampleRate = str2float(header['sample_rate'])
            
            if self.nLEDs == 1:
                led_pos = np.ma.masked_values(data['pos'][:,0:2], value=POS_NAN)
                led_pix = np.ma.masked_values(data['pos'][:,4:5], value=POS_NAN)
            elif self.nLEDs == 2:
                led_pos = np.ma.masked_values(data['pos'][:,0:4], value=POS_NAN)
                led_pix = np.ma.masked_values(data['pos'][:,4:6], value=POS_NAN)
            
            if returnWorking is False:
                self._xy,self._dir,_,self._w,self._h,self._posShape = \
                        self.posProcessor(led_pos,led_pix,posSampleRate,header,self.nLEDs)        
            else:
                return self.posProcessor(led_pos,led_pix,posSampleRate,header,self.nLEDs,returnWorking=True)
                
        elif self._mode == 'mat_a':
            
            self._posHeader,xy_data,xy_data2 = mat_a.load_pos(self.fn)
            if headerOnly:
                return
            self._xy,self._dir,self._dir_disp,self._w,self._h = self.posProcessor(xy_data,xy_data2)
            return
            
    def _loadTet(self,num,waves=False,headerOnly=False):        
         #TODO: hopefully can avoid loading waves when waves=False 
        if self._tetTimes is None:
            self._tetTimes = {}
            self._tetWaves = {}
            
        if headerOnly:
            self.tetHeader._cache[num], = _readFile(self.fn + "." + str(num),None,True)
            return
            
        header,data = _readFile(self.fn + "." + str(num),[('ts','>i'),('waveform','50b')])
        self.tetHeader._cache[num] = header
        self._tetTimes[num] = data['ts'][::4]
        
        if waves:
            nSpikes = str2int(header['num_spikes'])
            nChans = str2int(header['num_chans'])
            nSamples = str2int(header['samples_per_spike'])
            try:
                self._tetWaves[num] = data['waveform'].reshape(nSpikes, nChans, nSamples)
            except ValueError:
                raise Exception("Header for tet-%d of trial '%s' says nSpikes=%d, but data is of size %s." %(num,self.experimentName,nSpikes,data['waveform'].shape) )
                
    def _loadEeg(self,f,headerOnly=False):
        if self._mode == 'mat_a':
            self._eegHeader,self._eeg = mat_a.load_eeg(self.fn)
            return
            
        base = os.path.split(self.fn)[0]
        f = base + os.path.sep + f
        
        self._eegHeader, = _readFile(f,None,True)
        if headerOnly: return
            
        bytePerSamp = str2float(self._eegHeader.get('bytes_per_sample','1'))
        fmt = '=b' if bytePerSamp == 1 else '=h' # TODO: check high samp rate 
        self._eegHeader,eeg = _readFile(f,[('eeg',fmt)])
        eeg = eeg['eeg']
        nEeg = str2int(self._eegHeader['num_EEG_samples'] if 'num_EEG_samples' in self._eegHeader \
                        else self._eegHeader['num_EGF_samples'])
        self._eeg = eeg[:nEeg] # this is important as there can be nonsense at the end of the file.
        
    def _loadInp(self):
        #TODO: more than this
        self.inpHeader,self.inpData = _readFile(self.fn + ".inp",[('ts','>i4'),('type','>b'),('value','>2b')])
    
    @LazyDictProperty   
    def tetAmps(self,tetNum):
        if tetNum in self.tetAmps._cache:
            pass
        else:
            waves = self.tetWaves(tetNum)
            self.tetAmps._cache[tetNum] = np.uint8(np.max(waves,2)+128) - np.uint8(np.min(waves,2)+128)
        return self.tetAmps._cache[tetNum]
        
    @LazyDictProperty
    def tetHeader(self,tetNum):
        if tetNum in self.tetHeader._cache:
            pass
        else:
            if self._mode is not None:
                raise Exception("No tet headers available in mode=%s" % self._mode)
            self._loadTet(tetNum,headerOnly=True)
        return self.tetHeader._cache[tetNum]
    
    @LazyDictProperty
    def tetTimebase(self,tetNum):
        return str2float(self.tetHeader[tetNum]["timebase"])

    @LazyDictProperty        
    def cellsForCut(self, tetNum):
        """Returns the list of cell numbers on the cut currently selected for the 
        specified tetrode. Zero is not included in the list"""
        fname = self.cutFileNames[tetNum]        
        if tetNum in self.cellsForCut._cache and self.tetCut._cache[tetNum][0] == fname:
            pass
        else:
            u =  np.unique(self.tetCut[tetNum])
            u = u[u!=0]
            self.cellsForCut._cache[tetNum] = (fname, u)
        return self.cellsForCut._cache[tetNum][1]
            

    @LazyDictProperty
    def tetCut(self, tetNum):
        fname = self.cutFileNames[tetNum]
        
        if tetNum in self.tetCut._cache and self.tetCut._cache[tetNum][0] == fname:
            pass # in this case we can load straight from cache
        elif os.path.isfile(fname) is  True:
            with open(fname,'r') as f:
                cut_data = f.read()
                f.close()
            if fname[-4:] == ".cut":
                cut_data = cut_data.split('spikes: ',1)[1]
                #cut_data is now: n_spikes\n# # # # # etc.
            else: # for clu files
                #cut_data is now: n_clusters\n#\n#\n#\n#\n etc.
                pass
            cut = np.array(map(int,cut_data.split()[1:]),dtype=int)
            self.tetCut._cache[tetNum] = (fname, cut) 
        else:
            warn("Cut file note found: %s" % (fname))
            self.tetCut._cache[tetNum] = ('',[])
        
        return self.tetCut._cache[tetNum][1]
        
    @tetHeader.iter_function
    @tetCut.iter_function
    @tetTimebase.iter_function
    @tetAmps.iter_function
    def _getTetIter(self):
        return iter(self.availableTets)
        
    def _transformEeg(self):
        eeg = self.filteredEeg        
        # If we don't pad, then we may end up with a length that has very large prime factors
        # that would be very bad indeed when it comes to doing the fft (within the hilbert)    
        # As it happens, the data len should already be a multiple of the sample rate,
        # but if it's not we enforce it now...actually, you might want to add a x10 or
        # something for a small, but significant speed benefit (often/awlays trials get
        #assigned a length that is 1s longer than it should be, which ruins the ease of
        #factorsiing the length).
        padFactor = int(self.eegSampRate)
        padding = (padFactor-len(self.filteredEeg)) % padFactor
        if padding > 0:
            eeg = np.hstack((eeg,np.zeros(padding)))
            
        analytic = sp.signal.hilbert(eeg)
        
        if padding > 0:
            analytic = analytic[:-padding-1]
            
        self._thetaPhase = np.unwrap(np.angle(analytic))
        self._thetaAmp = np.abs(analytic)        
        

             
         
    def clearCaches(self):
        """
        Clears the various caches, hopefully leaving in tact anything that 
        could not be recreated by reading from disk again.
        
        Be careful using this, there may be bugs associated with using it.
                        
        Don't delete the things with roughly the followign names:
         ``posProcessor, thetaFreqBand, cutFileName, eegFileName, info``
        
        """

        self._clearCachePos(headerAlso=True) #this is kept sort of separate, though some values do appear again in TT.CLEARABLE_KEYS
                
        # Clear the LazyDict _caches, and set everythign else to None
        for k in TT.CLEARABLE_KEYS:
            d = self.__dict__.get(k,None)
            if isinstance(d,LazyDict):
                d._cache.clear()
            else:
                self.__dict__[k] = None


TT.CLEARABLE_KEYS = """_xy _dir _dir_disp _speed _w _h _posShape _eeg _eegHeader
_filteredEeg _thetaAmp _thetaPhase _thetaFreq _xyBall _ballHeader
_tetTimes _setHeader _tetWaves tetAmps _posHeader tetCut tetTimebase tetHeader
_LazyDicts""".split()
# TODO: could let the mixin modules append to this list

def _eegfilter(eeg,sampRate,freqBand):
    '''filters the ``eeg`` using a 1second-tap bandpass blackman filter
    between the values given in ``freqBand[0]`` and ``freqBand[1]``, e.g. ``[6,12]``.
    '''
    nyquist = sampRate / 2
    eegfilter = sp.signal.firwin(sampRate+ 1, map(lambda x: x/nyquist,freqBand),
                    window='black', pass_zero=False)
    return sp.signal.filtfilt(eegfilter, [1], eeg.astype(np.single))

def _rotateXY(xy,w,h,rot):
    """Rotates xy around the centre of wxh by rot degrees.
    TODO: also rotate direction values and shape.
    TODO: check W adn H are the correct way arouund.
    """
    rads = float(rot)/180*np.pi
    rotMatrix = np.asarray([[np.cos(rads), -np.sin(rads)], 
                            [np.sin(rads),  np.cos(rads)]])
                
    centre = np.asarray([[float(w)/2],[float(h)/2]])
    xy_r = xy - centre
    xy_r = np.asarray(np.matrix(rotMatrix) * np.matrix(xy_r))
    xy_r += centre
    eps = 1e-5
    np.clip(xy_r,np.asarray([[0],[0]]),np.asarray([[w-eps],[h-eps]]),out=xy_r)
    return xy_r
    

def _jitterXY(xy,w,h,jit):
    xy_j = xy + np.asarray(jit).reshape(2,1)
    eps = 1e-5
    np.clip(xy_j,np.asarray([[0],[0]]),np.asarray([[w-eps],[h-eps]]),out=xy_j)
    return xy_j


