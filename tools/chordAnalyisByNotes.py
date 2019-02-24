# -*- coding: utf-8 -*-
"""
Created on Fri May  2 11:02:49 2014

Script to analyse chords by notes contained, instead of by chord label.

@author: cuaras
"""
import os
import glob
import json
import operator
import csv
import copy
import chords2MIDI as c2m

#Hacer pruebas para comprobar que quedó bien, aunque parece que si

class Chord:
#Data struct to include start, end, label.
    pass

class ChordTable:
    
    def setChords(self, gtSong, song):
    #Input chords as strings
        gtChord = str(gtSong.notes).strip('[]')
        chord = str(song.notes).strip('[]')
        
        if gtChord not in self.chordTable.keys():
            self.chordTable[gtChord] = {}
        if chord not in self.chordTable[gtChord].keys():
            self.chordTable[gtChord][chord] = [0, gtSong.label, song.label]
        self.chordTable[gtChord][chord][0] += 1
    
    def mergeTable(self, table):
        #Method to merge 2 chord tables with chord occurances into a single one
        for gtChord in table.keys():
            if gtChord not in self.chordTable:
                self.chordTable[gtChord] = {}
            for chord in table[gtChord].keys():
                if chord not in self.chordTable[gtChord]:
                    gtLabel = table[gtChord][chord][1]
                    label = table[gtChord][chord][2]
                    self.chordTable[gtChord][chord] = [0, gtLabel, label]
                self.chordTable[gtChord][chord][0] += table[gtChord][chord][0]
            
    
    def __init__(self, filename= None):
    #Load a Chord table from a json file if specified
        self.chordTable = {}
        self.chordTable.clear()
        if filename != None:        
            with open(filename, 'r') as f:    
                self.chordTable = json.load(f)
    
def parseChords(filename):
#Opens a mirex chord file and returns a list with chord objects
    song = []
    with open(filename, 'r') as f:
        for line in f:
            #Intentar con separadores ' ' y '\t' 
            try:            
                start, end, label = line.strip().split('\t')
            except:
                try:
                    start, end, label = line.split(' ')
                except:
                    print 'error'
            label = label.strip('\n')       #Quitar End carriage
            chord = Chord()
            chord.start = float(start)
            chord.end = float(end)
            chord.label = label
            chord.notes = c2m.__getNotes(label)
            song.append(chord)
    return song

def processSong(gtSong, song):
    idx = 0
    chordTable = ChordTable()
    chordTable.chordTable.clear()
    MIN_CHORD_DURATION = .3
    #Avanza los dos cuando hay que cambiar chord
    for chord in song:
        isGreater = True
        while isGreater:
            #print chord.start, chord.end, gtSong[idx].start, gtSong[idx].end
            #Si la intersección es mayor a MIN_CHORD_DURATION 
            if min(chord.end, gtSong[idx].end) - max(chord.start, gtSong[idx].start) > MIN_CHORD_DURATION:
                chordTable.setChords(gtSong[idx], chord)
            isGreater = chord.end > gtSong[idx].end
            if isGreater: 
                idx += 1 #Aumentar el indice para gt
                isGreater = False if idx >= len(gtSong) else True
    return chordTable

def analyseAllSongs():
    #Analyze chords vs ground truth and generate json files per song
    baseFolder = '../outputs/billboard2012/BillboardTest2012/'
    subDirs = os.walk(baseFolder).next()[1]
    subDirs.remove('Ground-truth')    
    
    for gtFilename in glob.glob(baseFolder+'Ground-truth/*.lab'):
        print gtFilename
        baseFilename = gtFilename.split('/')[-1][:-3]+'wav.txt' #Filename in txt without directory
        gtSong = parseChords(gtFilename)     
        for folder in subDirs:
            filename = baseFolder + folder + '/' + baseFilename
            song = parseChords(filename)
            chordTable = processSong(gtSong, song)
            with open(filename[:-3]+'jsonN', 'w') as f:
                j = json.dumps(chordTable.chordTable)
                print >> f, j

def mergeAllSongs():
    #Merge all individual json files into one
    baseFolder = '../outputs/billboard2012/BillboardTest2012/'
    allChords = ChordTable()
    for filename in glob.iglob(os.path.join(baseFolder, '*', '*.jsonN')):
        tmp = ChordTable(filename)
        allChords.mergeTable(tmp.chordTable)
    with open('allChords.jsonN', 'w') as f:
        j = json.dumps(allChords.chordTable)
        print >> f, j
        
def mergeInFolder(folderPath, outFile = 'allChords.jsonN'):
    #Merge all individual json files into one
    #baseFolder = '../outputs/billboard2012/BillboardTest2012/'
    allChords = ChordTable()
    for filename in glob.glob(folderPath + '*.jsonN'):
        tmp = ChordTable(filename)
        allChords.mergeTable(tmp.chordTable)
    with open(folderPath + outFile, 'w') as f:
        j = json.dumps(allChords.chordTable)
        print >> f, j

def mergeEachSubFolder(baseFolder = '../outputs/billboard2012/BillboardTest2012/'):
    for subFolder in os.walk(baseFolder).next()[1]:     #Take only subs name
        mergeInFolder(baseFolder + subFolder + '/')
        
def topXtopY(chordTable, x, y):
    #Return a dictionary trimmed to top X gtChords and top Y chords from a chordTable
    sortedTable = ChordTable()    
    topX = countByChord(chordTable)[:x]
    
    for i in range(len(topX)):      #Iterate through gtChords
        gtKey = topX[i][0]          #get key
        sortedTable.chordTable[gtKey] = {}
        topY = sorted(chordTable[gtKey].iteritems(), 
                                 key = operator.itemgetter(1), reverse = True)
        topY = topY[:y]
        for i in range(len(topY)):  #Read top chord tags and iterate
            chordKey = topY[i][0]
            sortedTable.chordTable[gtKey][chordKey] = chordTable[gtKey][chordKey]
    return sortedTable
    
def countByChord(chordTable):
    #Cycle through all chords and count the number of occurances.
    counter = 0
    allGT = dict()
    for gtChord in chordTable.keys():
        for chord in chordTable[gtChord]:
            counter += chordTable[gtChord][chord]
        allGT[gtChord] = counter
        counter = 0
    allGT = sorted(allGT.iteritems(), key = operator.itemgetter(1),
                   reverse = True)  #Sort by descending order
    return allGT
    
def save2csv(filename, chordTable):
    #Save in filename the chordTable in csv format, open in json format
    row = ["","","","",""]
    columns = [["GT Label", "GT Notes", "Estim Label", "Estim Notes", "Count"]] #Containing all rows
    for gtChord in chordTable.keys():
        row[1] = gtChord
        for chord in chordTable[gtChord].keys():
            row[0] = chordTable[gtChord][chord][1]
            row[2] = chordTable[gtChord][chord][2]
            row[3] = chord
            row[4] = chordTable[gtChord][chord][0]
            columns.append(copy.deepcopy(row))

    with open(filename, 'w') as f:
        w = csv.writer(f)
        w.writerows(columns)

def csvEachMergedAlgorithm():
    #save each allChords.jsonN into a csv file
    baseFolder = '../outputs/billboard2012/BillboardTest2012/'
    for filename in glob.iglob(os.path.join(baseFolder, '*', 'allChords.jsonN')):
        tmp = ChordTable(filename)
        outFile = filename[:-5]+'csv'   #Tomando que son .jsonN
        save2csv(outFile, tmp.chordTable)
        
            
if __name__ == "__main__":
    song1 = parseChords('../outputs/billboard2012/BillboardTest2012/Ground-truth/1002.lab')
    song2 = parseChords('../outputs/billboard2012/BillboardTest2012/CF2/1002.wav.txt')

        