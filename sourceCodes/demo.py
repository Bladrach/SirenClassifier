import model
import torch
import os
import torch
import torchaudio
import glob
import cv2
from torchaudio import transforms
import matplotlib.pyplot as plt


net = model.Net()

net.load_state_dict(torch.load('net_epoch_20.pth'))

n_fft = 128
width = 65
#fbins = n_fft//2 + 1
spec_transform = transforms.Spectrogram(n_fft = n_fft, normalized = False)

data_path = "/home/mehmet/Desktop/SirenDetection/demo/*.wav"
dictionary = {0: 'Trafik gürültüsü', 1: 'İtfaiye sireni', 2: 'Ambulans sireni'}

waveArr = []
logSpecArr = []
predArr = []
for i in range(len(glob.glob(data_path))):
    print(glob.glob(data_path)[i])
    print(i)
    wave, sample_rate = torchaudio.load_wav(glob.glob(data_path)[i])
    #wave = wave * 0.8
    waveArr.append(wave)
    spec = spec_transform(wave)

    log_spec = (spec + 1e-9).log2()[0, :, :]
    height = log_spec.shape[0]
    dim = (width, height)
    log_spec = cv2.resize(log_spec.numpy(), dim, interpolation = cv2.INTER_AREA)
    #print(log_spec.shape)
    logSpecArr.append(log_spec)
    log_spec = torch.from_numpy(log_spec).unsqueeze(0).unsqueeze(1)
    output = net(log_spec)
    predArr.append(torch.max(output, 1)[1].item())
    print(dictionary[torch.max(output, 1)[1].item()])


plt.figure()
plt.subplot(2, 3, 1)
plt.ylabel("Genlik")
plt.xlabel("Örnekleme sayısı")
plt.title("Ambulans sireni")
plt.plot(waveArr[0].t().numpy(), 'b')
plt.subplot(2, 3, 4)
plt.ylabel("Frekans bölmeleri")
plt.xlabel("Zaman dilimi pencereleri")
plt.title("Ambulans sireninin spektrogramı")
plt.text(30, 90, dictionary[predArr[0]], color = 'r', horizontalalignment = 'center')
plt.imshow(logSpecArr[0])
plt.subplot(2, 3, 2)
plt.ylabel("Genlik")
plt.xlabel("Örnekleme sayısı")
plt.title("İtfaiye sireni")
plt.plot(waveArr[1].t().numpy(), 'b')
plt.subplot(2, 3, 5)
plt.ylabel("Frekans bölmeleri")
plt.xlabel("Zaman dilimi pencereleri")
plt.title("İtfaiye sireninin spektrogramı")
plt.text(30, 90, dictionary[predArr[1]], color = 'r', horizontalalignment = 'center')
plt.imshow(logSpecArr[1])
plt.subplot(2, 3, 3)
plt.ylabel("Genlik")
plt.xlabel("Örnekleme sayısı")
plt.title("Trafik gürültüsü")
plt.plot(waveArr[2].t().numpy(), 'b')
plt.subplot(2, 3, 6)
plt.ylabel("Frekans bölmeleri")
plt.xlabel("Zaman dilimi pencereleri")
plt.title("Gürültünün spektrogramı")
plt.text(30, 90, dictionary[predArr[2]], color = 'r', horizontalalignment = 'center')
plt.imshow(logSpecArr[2])
plt.tight_layout(pad = 1.3)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
