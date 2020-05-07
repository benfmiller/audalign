import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt

filename = 'SUB.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
plt.show()
