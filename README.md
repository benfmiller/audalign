# AudioFingerprinting
Package for aligning audio files with audio fingerprinting

Currently under construction

The goal of audalign is to offer a package for aligning many recordings of the same incident.

The package may offer tools to modify audio files by placing empty space before or after a 
target file so that all will automatically be aligned.

This was originally forked from dejavu by Will Drevo, but my implementation
differs from drevo's drastically. Dejavu is written in python 2, and I re-implemented
it in python 3. Audalign also doesn't make use of any databases as keeping a handful of 
fingerprints within memory is no problem, though there is a method for saving and loading the
fingerprints to a json or pickle file.

Audalign will be mainly using the fingerprinting algorithm implemented in Panako, which is an
audio fingerprinting package written in Java.