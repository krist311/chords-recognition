# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 12:09:27 2014

Code with functions to extract chord data (MIREX notation) and convert it
to MIDI files. Use parseChords to save data into a MIDIFile object and save
to write file.

@author: cuaras
"""

from midiutil.MidiFile import MIDIFile
import numpy as np

#Create Dictionary for MIDI notes
MIDInote = {'A': 57, 'A#': 58, 'Bb': 58, 'B': 59, 'Cb': 59,'C': 60, 'C#': 61, 'Db': 61,
         'D': 62, 'D#': 63, 'Eb': 63, 'E': 64, 'E#': 65, 'F': 65, 'F#': 66, 'Gb': 66, 
         'G': 67, 'G#': 68, 'Ab': 68}
         
#Create dicitionary for chords in semitone intervals
chType = {'N': 0, 'maj' : [0, 4, 7], 'min' : [0, 3, 7], 'maj7' : [0, 4, 7, 11],
          'min7' : [0, 3, 7, 10], '7' : [0, 4, 7, 10], 'sus4' : [0, 5, 7], 
          'sus2' : [0, 2, 7], 'maj6' : [0, 4, 7, 9], '1' : [0], '5' : [0, 7],
          'maj11' : [0, 4, 7, 11, 14, 17], '11' : [0, 4, 7, 10, 14, 17],
          'min11' : [0, 3, 7, 10, 14, 17], 'maj13' : [0, 4, 7, 11, 14, 17, 21],
          '13' : [0, 4, 7, 10, 14, 17, 21], 'min13' : [0, 3, 7, 10, 14, 17, 21],
          'dim' : [0, 3, 6], 'aug' : [0, 4, 8], 'min6' : [0, 3, 7, 9],
          '9' : [0, 4, 7, 10, 14], 'maj9' : [0, 4, 7, 11, 14], 'min9' : [0, 3, 7, 10, 14],
          'hdim7' : [0, 3, 6, 10], '' : [0], 'minmaj7': [0, 3, 7, 11],
          'dim7' : [0, 3, 6, 9]}

#Intervalos a intervalos en semitonos
interval = {'1' : 0, '2' : 2, '3' : 4, '4' : 5, '5' : 7, '6' : 9, '7' : 11,
            '8' : 12, '9' : 14, '10' : 16, '11' : 17, '12' : 19, '13' : 21}

modifier = {'b' : -1, '#': 1}



def save(midi, filename='"output.mid'):
    # Write it to disk.
    binfile = open(filename, 'wb')
    midi.writeFile(binfile)
    binfile.close()


def __semitoneInterval(note):
    hasAccidental = note.find('#') != -1 or note.find('b') != -1
    if hasAccidental:
        note = modifier[note[0]] + interval[note[1:]]
    else:
        note = interval[note[0:]]
    return note

def newNote(start, stop, note):
    #Start y Stop en Segundos a 120 bpm es el tiempo*2 (un segundo dos barras)
    start = float(start)*2      
    stop = float(stop)*2
    track = 0
    channel = 0
    pitch = int(note)
    time = start
    duration = stop-start
    volume = 100    
    return track, channel, pitch, time, duration, volume
    
def __newChord(start, stop, chord, midi):
    if chord != None:
        for note in chord:
            midi.addNote(*newNote(start, stop, note))

def parseChords(midi, filename = 'sample.txt'):
    with open(filename, 'r') as f:
        for line in f:
            #Intentar con separadores ' ' y '\t'            
            try:            
                start, stop, chord = line.strip().split('\t')
            except:
                try:
                    start, stop, chord = line.split(' ')
                except:
                    print 'error'
            chord = chord.strip('\n')       #Quitar End carriage
            #print '-'+str(start), '-'+str(stop), '-'+str(chord)          
            notes = __getNotes(chord)
            #print chord, notes
            __newChord(start, stop, notes, midi)

def __getBass(rest):
#Gets Bass and removes it from string
    try:
        idx = rest.index('/')
        bass = rest[idx+1 :]
        rest = rest[:idx]
        bass = __semitoneInterval(bass)

    except:
        bass = 0
        
    return bass, rest

def __getNotes(chord):
    #Gets the rootnote and returns it MIDI    
    if chord.strip() != 'N' and chord.strip() != 'X':
        
        tmpSplit = chord.split (':')
        if len(tmpSplit) == 1:  #if chord doesn't have type consider major
            root = tmpSplit[0]            
            rest = 'maj'
        else: root, rest = tmpSplit       
        
        #Get Bass
        bass, rest = __getBass(rest)
        if root.find('/') != -1:
            #In case notation is missing a :, get bass from root
            bass, root = __getBass(root)
        
        #Find additional notes inside parenthesis
        addNotes = None
        if rest.find('(') != -1:
            addNotes = rest[rest.index('(')+1 : rest.index(')')]
            addNotes = addNotes.split(',')
            for i in range(len(addNotes)):    
                addNotes[i] = __semitoneInterval(addNotes[i].strip())
            rest = rest[:rest.index('(')]
            
        #Get the components for shorthand notation in semitones
        notes = list(chType[rest])  #Hacer copia
        if bass != 0:
            try:
                notes.insert(0, notes.pop(notes.index(bass)) - 12)
            except:
                #Si no se puede es que no hay que quitarlo del acorde y lo bajamos una octava
                notes.append(bass-12)
                notes.sort()
        
        if addNotes != None:
            notes += addNotes
            notes.sort()
        
        notes = np.array(notes)+MIDInote[root]
        
        
        
        return notes


        
if __name__ == "__main__":
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
    
    # Now add the note.
    #mChords.addNote(track,channel,pitch,time,duration,volume)