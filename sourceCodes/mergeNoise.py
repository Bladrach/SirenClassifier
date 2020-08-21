from pydub import AudioSegment
import os
import glob
import shutil
from shutil import copyfile


def merge(source1, source2):
    source1Arr = []
    source2Arr = []
    for ind, wn in enumerate(os.listdir(source1)):
        source1Arr.append(wn)

    for index, wavname in enumerate(os.listdir(source2)):
        source2Arr.append(wavname)
    
    print(source1Arr)
    print(len(source1Arr))
    print(source2Arr)
    print(len(source2Arr))
    #print(source2Arr[134])

    for i in range(len(source1Arr)):
        sound1 = AudioSegment.from_file(source1 + "/" + source1Arr[i])
        print(i)
        sound2 = AudioSegment.from_file(source2 + "/" + source2Arr[i])

        combined = sound1.overlay(sound2)

        combined.export("/home/mehmet/Desktop/SirenDetection/noisyFiretruck/" +  str(source1Arr[i]), format = 'wav')  #/noisyFiretruck is needed to be changed
                                                                                                                    #for ambulance manually and create a folder for it

traffic = "/home/mehmet/Desktop/SirenDetection/dataset/traffic"
ambulance = "/home/mehmet/Desktop/SirenDetection/dataset/ambulance"
firetruck = "/home/mehmet/Desktop/SirenDetection/dataset/firetruck"

merge(firetruck, traffic)
