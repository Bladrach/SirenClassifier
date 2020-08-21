import torch
import torchaudio
from torch.utils.data import Dataset
import glob
import cv2
from torchaudio import transforms
import numpy as np
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, data_path, train = True):
        if(train == True):
            self.data_path = glob.glob(data_path + "/train/*.wav")
        else:
            self.data_path = glob.glob(data_path + "/test/*.wav")
    


    def __getitem__(self, index):
        filename = self.data_path[index]
        n_fft = 128
        #fbins = n_fft//2 + 1
        spec_transform = transforms.Spectrogram(n_fft = n_fft, normalized = False)

        label = int(filename.split("/")[-1].split("_")[0])
        soundSource = filename.split("/")[-1].split("_")[1]
        number = filename.split("/")[-1].split("_")[2]

        wave, sample_rate = torchaudio.load_wav(filename)

        spec = spec_transform(wave)

        log_spec = (spec + 1e-9).log2()[0, :, :]
        

        width = 65
        height = log_spec.shape[0]
        dim = (width, height)
        log_spec = cv2.resize(log_spec.numpy(), dim, interpolation = cv2.INTER_AREA)
        plt.figure()
        plt.imshow(log_spec)
        plt.show()
        

        return log_spec, label, soundSource

    def __len__(self):
        return len(self.data_path)
