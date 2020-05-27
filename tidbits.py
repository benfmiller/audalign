from os import path
from pydub import AudioSegment
import pydub

                                                                   
src = "Street.mp3"
dst = "Street.wav"

# convert wav to mp3               
#AudioSegment.converter = "C:\\Users\\benfm\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages"                                             
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")


import moviepy.editor
import moviepy.audio
import numpy as np



def conv_vid_to_aud(vid_name, aud_name):
    # Replace the parameter with the location of the video
    video = moviepy.editor.VideoFileClip(vid_name)
    audio = video.audio
    # Replace the parameter with the location along with filename
    audio.write_audiofile(aud_name)

#conv_vid_to_aud("SUB.mp4", "SUB.wav")

#import soundfile as sf

# Extract audio data and sampling rate from file 
#data, fs = sf.read('SUB.wav') 
# Save as FLAC file at correct sampling rate
#sf.write('myfile.flac', data, fs)  
#print(data)

#import wavio
import librosa

#wavio.write("myfile.wav", my_np_array, fs, sampwidth=2)
"""
audio = moviepy.editor.AudioFileClip('Street.mp3')
audionp = audio.to_soundarray()
print(audionp[10000:11000])"""

"""
from scipy.io.wavfile import read
a = read("Street.wav")
numpy.array(a[1],dtype=float)
array([ 128.,  128.,  128., ...,  128.,  128.,  128.])"""

#import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt

filename = 'SUB.wav'
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
plt.show()

#------------------------------------------------
import numpy as np
import simpleaudio as sa

frequency = 440  # Our played note will be 440 Hz
fs = 44100  # 44100 samples per second
seconds = 3  # Note duration of 3 seconds

# Generate array with seconds*sample_rate steps, ranging between 0 and seconds
t = np.linspace(0, seconds, seconds * fs, False)

# Generate a 440 Hz sine wave
note = np.sin(frequency * t * 2 * np.pi)

# Ensure that highest value is in 16-bit range
audio = note * (2**15 - 1) / np.max(np.abs(note))
# Convert to 16-bit data
audio = audio.astype(np.int16)

# Start playback
play_obj = sa.play_buffer(audio, 1, 2, fs)

# Wait for playback to finish before exiting
play_obj.wait_done()
