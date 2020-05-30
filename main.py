import audalign as ad


#TODO: Rework Fingerprint Directory
#TODO: Multiprocessing
#TODO: Not double add files
#TODO: Handle non audio files
#TODO: Add precision optional adjustments to recognize
#TODO: Document

#Tests:
#TODO: Different Sample rates? Needed?
#TODO: Stretch
#TODO: Transpose
#TODO: wav vs mp3
#TODO: is normalize needed



ada = ad.Audalign()

ada.fingerprint_file("TestAudio/SUB.wav", plot=False, normalize=False)
print(len(ada.fingerprinted_files[0][1]))
#djv.save_fingerprinted_files('Sub.json')
#print(len(djv.fingerprinted_files))

#print(djv.fingerprinted_files[0][0] + "  :  " + djv.fingerprinted_files[0][2])
#a = djv.fingerprinted_files
#b = a[0][1]
#print(b)

#print("\nBeginning Recognizing")
#print(djv.recognize(recognizer='MicrophoneRecognizer', seconds=10))
#print(djv.recognize("SUB.mp3"))

#djv.save_fingerprinted_files('test_mp3s.pickle')



#djv.fingerprint_directory("mp3")
#print(djv.fingerprinted_files)





#filen = djv.recognize("mp3/Sean-Fournier--Falling-For-You.mp3")
#print(filen)
#print ("From file we recognized: %s\n" % filen)

"""
filepath = "SUB.mp3"

filename = audalign.decoder.path_to_filename(filepath)
file_hash = audalign.decoder.unique_hash(filepath)
file_name = filename
# don't refingerprint already fingerprinted files
#file_name, hashes, file_hash = audalign._fingerprint_worker(filepath)
#print(hashes)

channels, Fs, file_hash = audalign.decoder.read(filepath)
channel_amount = len(channels)
result = []

for channeln, channel in enumerate(channels):
    print("Fingerprinting channel %d/%d for %s" % (channeln + 1,
                                                    channel_amount,
                                                    filepath))
    hashes = audalign.fingerprint.fingerprint(channel, Fs=Fs, plot=False)
    print("Finished channel %d/%d for %s" % (channeln + 1, channel_amount,
                                                filepath))
    result += hashes
    print("Length is: {}".format(len(result)))
"""