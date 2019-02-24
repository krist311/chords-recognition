# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:24:43 2014

@author: cuaras
"""

import chords2MIDI as c2m
import glob
import os
from midiutil.MidiFile import MIDIFile

# Create the MIDIFile Object with 1 track
mChords = MIDIFile(1)

# Tracks are numbered from zero. Times are measured in beats.
track = 0   
time = 0
tempo = 120
trackName = "Sonified Chords"

# Add track name and tempo.
mChords.addTrackName(track,time, trackName)
mChords.addTempo(track, time, tempo)

# Add a note. addNote expects the following information:
track = 0
channel = 0
pitch = 60
time = 0
duration = 1
volume = 100

path = '../outputs/billboard2012/BillboardTest2012'
i = 0
for filename in glob.iglob(os.path.join(path, '*', '*.lab')):
    mChords = MIDIFile(1)    
    mChords.addTrackName(track,time, trackName)
    mChords.addTempo(track, time, tempo)
    
    print filename
    c2m.parseChords(mChords, filename)
    out = filename[:-3] + '.mid'
    c2m.save(mChords, out)
    if i%20 == 0: print str(i) + ' files processed'
    i+=1