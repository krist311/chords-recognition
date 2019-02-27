# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 12:17:09 2014

Script to extract excerpts of sonified MIDI algorithms where the chord
was miscalculated.

@author: cuaras
"""

import json
import csv
import random
import os
import essentia.standard as es


def openChord(chordNotes, GTNotes):
    #Open json file showing exceprts where chord-GT combination occurs
    #[i]list of algorithm/song, [i][j] name, algorithm and cases per song
    #[i][2][k][l][m] k = case per song | l = 0,1 (start/end) | m = 1,2 (GT & estimation)
    path = '../scriptDocs/'
    filename = chordNotes + '-' + GTNotes + '.json'
    with open(path+filename, 'r') as f:
        return json.load(f)

def getExcerpt(chPair, chunk, rnd, rnd2):
    #Algorithm to get exceprts and save them
    
    OFFSET = 3  #Seconds before and after chord    
    skipWrite = False    
    
    GTChordStart = chunk[rnd][2][rnd2][0][1]
    GTChordEnd = chunk[rnd][2][rnd2][1][1]
    chordStart = chunk[rnd][2][rnd2][0][2]    
    chordEnd = chunk[rnd][2][rnd2][1][2]
        
    songNumber = chunk[rnd][0][:-4]
    songAlgorithm = chunk[rnd][1]    
    
    path =   '../outputs/billboard2012/BillboardTest2012/' + songAlgorithm + '/mp3/'
    GTpath = '../outputs/billboard2012/BillboardTest2012/Ground-truth/mp3/' 
    outPath = '../excerpts/algorithms/' + chPair + '/'
    outGTPath = '../excerpts/GT/' + chPair + '/'
    
    for directory in [outPath, outGTPath]:
        if not os.path.exists(directory):
            os.makedirs(directory)    
    
    #print 'Opening Estimation File'
    audio = es.MonoLoader(filename = path + songNumber[:-3]+'wav..mp3')()    
    
    #Check if audio +/- 3 secs exisits for chunk to be valid, skip file if not
    if (chordStart - OFFSET < 0) or (chordEnd + OFFSET > len(audio)/44100.0):
        log = 'ALG , ' + songNumber + ' , ' + chPair + ' , ' + str(chordStart) + ' , ' + str(chordEnd)             
        print 'SKIPPING - ' + log          
        with open('../excerpts/skipped.log', 'a') as f:           
            f.write(log + '\n')
        skipWrite = True
    else:
        out = audio[44100*(chordStart - OFFSET):44100*(chordEnd + OFFSET)]
    
    #print 'Writing Estimation File'
    fn = outPath + songNumber+'-'+songAlgorithm+'-'+str(chordStart)+'.wav'
    if not skipWrite: 
        es.MonoWriter(filename = fn)(out)
    else:
        skipWrite = False       #Reset skipwrite for GT file

    #print 'Opening GT File'
    audio = es.MonoLoader(filename = GTpath + songNumber[:-3]+'.mp3')()
    if (chordStart - OFFSET < 0) or (chordEnd + OFFSET > len(audio)/44100.0):
        log = 'GT , ' + songNumber + ' , ' + chPair + ' , ' + str(GTChordStart) + ' , ' + str(GTChordEnd)        
        print 'SKIPPING - ' + log        
        with open('../excerpts/skipped.log', 'a') as f:
            f.write(log + '\n')
        skipWrite = True
    else:    
        out = audio[44100*(GTChordStart - OFFSET):44100*(GTChordEnd + OFFSET)]
    
   #print 'Writing GT File'
    fn = outGTPath + songNumber+'-'+songAlgorithm+'-'+str(GTChordStart)+'.wav'
    if not skipWrite:    
        es.MonoWriter(filename = fn)(out)

if __name__ == "__main__":
    topChordsFile = '../scriptDocs/topChords.csv'
    chordList = [] #Will read Name, notes, GT, GT notes, count
    with open(topChordsFile, 'r') as f:
        rd = csv.reader(f)
        for line in rd:
            chordList.append(line)
    
    for i in range(len(chordList)):
        chPair = chordList[i][0] + '-' + chordList[0][2]
        chunk  = openChord(chordList[i][1], chordList[i][3])    
        
        print 'Getting chunks for ' + chPair    
        for j in range(10):
            print j + 1,
            rnd = random.randint(0, len(chunk)-1)        #Random for song/algorithm
            rnd2 = random.randint(0, len(chunk[rnd][2])-1) #Random for num repetition in song
            getExcerpt(chPair, chunk, rnd, rnd2)
        
        print ""    #Force newline
        

    
    
    