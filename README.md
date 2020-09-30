# AudioFingerprinting
Python package for aligning audio files using audio fingerprinting.

The goal of audalign is to offer a package for aligning many recordings of the same event.

This package offers tools to modify audio files by placing empty space before or after a 
target file so that all will automatically be aligned.

## Installation

Audalign is not currently listed in PyPI but can be pip installed if the package is downloaded by

```bash
git clone https://github.com/benfmiller/audalign.git
cd audalign/
pip install audalign
```

OR

Download and extract audalign then
```bash
pip install audalign
```
in the directory

Don't forget to install ffmpeg/avlib (Below in the Readme)!

## Fingerprinting

```python
import audalign

ada = audalign.Audalign()
ada.fingerprint_file("test_file.wav")

# or

ada.fingerprint_directory("audio/directory")
```
fingerprints are stored in ada and can be saved by 

```python
ada.save_fingerprinted_files("save_file.json") # or .pickle
```
Most video and audio file formats can be decoded

## Recognizing

```python
print(ada.recognize("matching_file.mp3"))
```
File doesn't have to be fingerprinted already. If it is, the file is not re-fingerprinted

Returns dictionary match time and match info. Match info is a dictionary of each file it recognized with. Each file is a dictionary of match information.

## Aligning

```python
print(ada.align("folder/files"))
```
Returns dictionary of each file recognized and best alignment. Also returns match info dictionary of each recognition in the folder

You can specify a destination folder to write the aligned files with the appropriate length of silence added to the front.

## General use

```python
ada.load_fingerprinted_files("data.json") # or .pickle
ada.plot("file.wav") # Plots spectrogram with peaks overlaid
ada.multiprocessing = False # If you want single threaded fingerprinting
ada.set_accuracy(1) # from 1-4, sets fingerprinting variables for different levels of accuracy
ada.hash_style = "base" #you can use "base" "base_three" "panako" "panako_mod"
ada.convert_audio_file("dont_like_wavs.wav", "like_mp3s.mp3") # Also convert video file to audio file
```


## Getting ffmpeg set up

You may use **ffmpeg or libav**.

Mac (using [homebrew](http://brew.sh)):

```bash
# ffmpeg
brew install ffmpeg --with-libvorbis --with-sdl2 --with-theora

####    OR    #####

# libav
brew install libav --with-libvorbis --with-sdl --with-theora
```

Linux (using aptitude):

```bash
# ffmpeg
apt-get install ffmpeg libavcodec-extra

####    OR    #####

# libav
apt-get install libav-tools libavcodec-extra
```

Windows:

1. Download and extract ffmpeg from [Windows binaries provided here](https://ffmpeg.org/download.html).
2. Add the ffmpeg `/bin` folder to your PATH envvar

OR

1. Download and extract libav from [Windows binaries provided here](http://builds.libav.org/windows/).
2. Add the libav `/bin` folder to your PATH envvar
