#!/usr/bin/env python
import pickle
import os
print os.getcwd()
import yaafelib2 as yaafelib
import emoClassify as ec
import time

os.chdir("/home/alice/Urop/PythonObjectLM")
f=open("pickledClassifier.p", "rb")
model=pickle.load(f)

afp, engine=ec.createAFP()

def getNum(emo):
    #anger
    if emo=='a':
        return "-20"
    #angst
    elif emo=='ast':
        return "-1"
    #boredom
    elif emo=='b':
        return "0"
    #disgust
    elif emo=='d':
        return "-20"
    #happiness
    elif emo=='h':
        return "20"
    #sadness
    elif emo=='sa':
        return "0"
    elif emo=='su':
        return "1"
    #neutral
    elif emo=='f':
        return "-1"
    elif emo=='n':
        return "0"
    else:
        return emo

while True:
    gotFile=False
    while not gotFile: 
        gotFile=os.path.isfile("sentWavFile.wav")
    print "gotFile=", gotFile
    time.sleep(0.1)
    feats=ec.extractFeatures("sentWavFile.wav", afp, engine)
    os.system("rm 'sentWavFile.wav'")
    emo=model.predict(feats)
    emoNum=getNum(emo)
    outFile=open("emoNumber.txt", "w")
    outFile.write(emoNum)
    print emoNum
    outFile.close()
    
