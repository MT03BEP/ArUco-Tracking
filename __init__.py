#!/usr/bin/env python3

"""
This file creates a class 'RunFile' which allows reading cfg files and their corresponding binfiles.
syntax:
run1=RunFile('.\202000_Run1.cfg')

It uses the following functions:
------------
read_data(): create data and time based on binfiles

syntax: run1.read_data()

relevant returns: run1.data (data),
                  run1.timedat(timeline),
                  run1.Source(info on each Source),
                  run1.chan(info on channels)

-------------
plot_data(): create plots per source and total overview

syntax: run1.plot_data()

creates plot per source and total overview; plots are placed in workdirectory (where script is placed)

more info: J.RodriguesMonteiro@tudelft.nl
"""

import re
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt


class RunFile:
    def __init__(self, filename):
        #declare initial values
        self.filename = filename
        self.cfg = []
        self.srcidx = []
        self.nameidx = []
        self.sridx = []
        self.fidx = []
        self.chanidx = []
        self.unitidx = []
        self.pid = 0
        self.rn = 0
        self.nchan = np.zeros([1])
        self.nsrc = 0
        self.chanfind = []
        self.unitfind = []
        self.Source = Struct(prefix = 'Source')
        self.chan = Struct(prefix = 'Channel')
        self.data = {}
        self.timedat = {}


    def run_id(self):
        split = re.findall('\d+', self.filename)  # find the runnumber in filename
        if len(split) > 1:
            self.pid = split[0]  #ProjectID
            self.rn = split[1]  #RunNumber
        else:
            self.rn = split[0]
            self.pid = 0
        return

    def read_cfg(self):
        self.cfg = loadtxt(self.filename,
                           dtype = str,
                           delimiter = '\n',
                           unpack=False)
        # find index numbers of source, name of source, samplerate and .bin-file
        srcidx = np.where(np.char.find(self.cfg, '[So')>=0)
        nameidx = np.where(np.char.find(self.cfg, 'Naam=')>=0)
        sridx = np.where(np.char.find(self.cfg, 'SampleRate=')>=0)
        fidx = np.where(np.char.find(self.cfg, 'File=')>=0)
        chanidx = np.where(np.char.find(self.cfg, 'Kanaal')>=0)
        unitidx = np.where(np.char.find(self.cfg, 'Unit')>=0)

        # get array form
        self.srcidx = srcidx[0]
        self.nameidx = nameidx[0]
        self.sridx = sridx[0]
        self.fidx = fidx[0]
        self.chanidx = chanidx[0]
        self.unitidx = unitidx[0]

        self.nsrc = self.srcidx.size  #number of sources
        self.nchan = np.zeros([self.nsrc])  #array for number of channels p/source

        nn = 1
        for n in range(self.chanidx.size):
            strtag = 'Source' + str(nn)
            if nn<=self.srcidx.size:
                try:
                    if self.chanidx[n]<self.srcidx[nn]:
                        self.chanfind.append(re.split('=', self.cfg[self.chanidx[n]], 2)[1])
                        self.unitfind.append(re.split('=', self.cfg[self.unitidx[n]], 2)[1])
                    else:
                        self.chan.add({'Source': strtag,
                                       'Channels' : self.chanfind,
                                       'Units' : self.unitfind})
                        nn+=1
                        self.chanfind =[]
                        self.unitfind=[]
                        strtag = 'Source' + str(nn)
                except IndexError:
                    #print('woopsie')
                    self.chanfind.append(re.split('=', self.cfg[self.chanidx[n]], 2)[1])
                    self.unitfind.append(re.split('=', self.cfg[self.unitidx[n]], 2)[1])
                    if n == self.chanidx.size-1:
                        self.chan.add({'Source': strtag,
                                       'Channels' : self.chanfind,
                                       'Units' : self.unitfind})


        for ii in range(self.nsrc):
            try:
                self.nchan[ii] = np.where(np.char.find(self.cfg[self.srcidx[ii]:self.srcidx[ii+1]], 'Kanaal')>=0)[0].size
                if self.nchan[ii] < 1:
                    self.nchan[ii] = np.where(np.char.find(self.cfg[self.srcidx[ii]:self.srcidx[ii+1]], 'Kanaal')>=0)[0].size
            except IndexError:
                self.nchan[ii] = np.where(np.char.find(self.cfg[self.srcidx[ii]:], 'Kanaal')>=0)[0].size

        for i in range(self.nsrc):
            self.Source.add({'Naam': self.cfg[nameidx][i][5:],
                             'SampleRate': self.cfg[sridx][i][11:],
                             'BinFile': self.cfg[fidx][i][5:],
                             'nChan': self.nchan[i]})
        return self

    def read_data(self):
        self.read_cfg()
        srcdict = self.Source.__dict__  # prefix, i, Source1... dictionary format
        for i in range(srcdict['i']):
            keystr = 'Source' + str(i+1)   #get source e.g. Source1
            binfile = srcdict[keystr]['BinFile']    #binfile belonging to source
            sr = int(srcdict[keystr]['SampleRate'])  # samplerate of this source
            src_nchan = srcdict[keystr]['nChan']    # number of channels in source

            dt = np.dtype(np.float32)
            data_tot = np.fromfile(binfile, dt)
            datapoints = int(np.size(data_tot) / src_nchan)
            datacols = int(src_nchan)
            data = np.reshape(data_tot, (datapoints, datacols))
            # create timeline
            time = np.arange(0, datapoints/sr, 1/sr)
            self.data[keystr]=data
            self.timedat[keystr]=time

        return self.timedat, self.data

    def plot_data(self):
        self.run_id()
        chan = self.chan.__dict__
        for s in range(self.Source.i):
            plt.figure(s, figsize=(16,9))
            cstr = 'Channel' + str(s+1)
            sstr = 'Source' + str(s+1)
            for c in range(chan[cstr]['Channels'].__len__()):
                plt.plot(self.timedat[sstr], self.data[sstr][:, c], label=(chan[cstr]['Channels'][c]+
                                                                           chan[cstr]['Units'][c]))
            plt.xlabel = 'Time'
            plt.legend()
            plt.title(sstr)
            plt.grid()
            #plt.show()
            if self.pid == 0:
                plt.savefig('Run' + str(self.rn) +'_' + sstr + '.png')
            else:
                plt.savefig(str(self.pid) + '_Run' + str(self.rn) +'_' + sstr + '.png')

        plt.figure(figsize=(16,9))
        for s in range(self.Source.i):
            cstr = 'Channel' + str(s+1)
            sstr = 'Source' + str(s+1)
            for c in range(chan[cstr]['Channels'].__len__()):
                plt.plot(self.timedat[sstr], self.data[sstr][:, c], label=(chan[cstr]['Channels'][c]+
                                                                           chan[cstr]['Units'][c]))
        plt.xlabel = 'Time'
        plt.legend()
        plt.title(str(self.rn) + ': All Channels available')
        plt.grid()

        if self.pid == 0:
            plt.savefig('Run' + str(self.rn) + '_All.png')
        else:
            plt.savefig(str(self.pid) + '_Run' + str(self.rn) +'_All.png')
        #plt.show()
        #plt.close(all)
        return

class Struct:
    def __init__(self, *args, prefix='arg'): # constructor
        self.prefix = prefix
        if len(args) == 0:
            self.i = 0
        else:
            i=0
            for arg in args:
                i+=1
                arg_str = prefix + str(i)
                # store arguments as attributes
                setattr(self, arg_str, arg) #self.arg1 = <value>
            self.i = i
    def add(self, arg):
        self.i += 1
        arg_str = self.prefix + str(self.i)
        setattr(self, arg_str, arg)



