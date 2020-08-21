
import os
import glob
import shutil
from shutil import copyfile


def split(source):
    for ind, file in enumerate(os.listdir(source)):
       for index, wavname in enumerate(os.listdir(source + "/" + file)):
           copyfile(source + "/" + file + "/" + wavname, source + "/" + str(ind) + "_" + str(file) + "_" + str(index) + ".wav")


if __name__ == '__main__':
    
    data_path = glob.glob("/home/mehmet/Desktop/SirenDetection/noisyDataset/*")
    save_path = "/home/mehmet/Desktop/SirenDetection/3class"
    for i in range(len(data_path)): 
        folderName = data_path[i]
        fn = str(folderName.split("/")[6])
        os.mkdir(save_path + "/test/" + fn)
        os.mkdir(save_path + "/train/" + fn)
        wavFolderName = glob.glob(folderName + "/*")
        print("{}. klasör için işlem başladı. Lütfen bekleyiniz...".format(str(i + 1)))
        for j in range(len(wavFolderName)):
            if(j <= len(wavFolderName)*0.2):
                shutil.copy(wavFolderName[j], save_path + "/test/" + fn)
            else:
                shutil.copy(wavFolderName[j], save_path + "/train/" + fn)
        print("İşlem başarıyla tamamlandı!")

    split("/home/mehmet/Desktop/SirenDetection/3class/test")
    split("/home/mehmet/Desktop/SirenDetection/3class/train")
