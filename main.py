from os import path
from pydub import AudioSegment
import pydub


#asd
# files   "                                                                      
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