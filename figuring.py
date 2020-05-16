import warnings
import json
import molasses as mo
warnings.filterwarnings("ignore")

from dejavu import Dejavu
from dejavu.recognize import FileRecognizer

djv = Dejavu()

djv.fingerprint_file("SUB.mp3")
print(djv.fingerprinted_files)
a = djv.fingerprinted_files
b = a[0][1][0]
print(b)


#djv.fingerprint_directory("mp3", [".mp3"])
#print(djv.fingerprinted_files)
#djv.save_fingerprinted_songs('test_mp3s.json')



"""
song = djv.recognize(FileRecognizer, "mp3/Sean-Fournier--Falling-For-You.mp3")
print(song)
print ("From file we recognized: %s\n" % song)
"""



"""
# Recognize audio from a file
song = djv.recognize(FileRecognizer, "mp3/Sean-Fournier--Falling-For-You.mp3")
print(song)
print ("From file we recognized: %s\n" % song)

# Or recognize audio from your microphone for `secs` seconds
#secs = 5
#song = djv.recognize(MicrophoneRecognizer, seconds=secs)
if song is None:
    print ("Nothing recognized -- did you play the song out loud so your mic could hear it? :)")
else:
    print ("From mic with %d seconds we recognized: %s\n" % (secs, song))

# Or use a recognizer without the shortcut, in anyway you would like
recognizer = FileRecognizer(djv)
song = recognizer.recognize_file("mp3/Josh-Woodward--I-Want-To-Destroy-Something-Beautiful.mp3")
print ("No shortcut, we recognized: %s\n" % song)"""

