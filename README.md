# AudioFingerprinting
Python package for aligning audio files using audio fingerprinting.

The goal of audalign is to offer a package for aligning many recordings of the same event.

This package offers tools to modify audio files by placing empty space before or after a 
target file so that all will automatically be aligned.

## Installation

Audalign is not currently listed in PyPI but can be pip installed if the package is downloaded by

```bash
git clone https://github.com/benfmiller/audalign.git
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

## Recognizing

## Aligning


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

####    OR    #####

1. Download and extract libav from [Windows binaries provided here](http://builds.libav.org/windows/).
2. Add the libav `/bin` folder to your PATH envvar
