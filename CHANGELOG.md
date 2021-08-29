# Change Log

## [0.7.1] 2021 - 08 - 29

### Changed

- Better handling of second match in rankings

### Added

- number of fingerprints print in alignments
- run_align.py script. CLI with argparse to run alignment easily

### Fixed

- runtime divide by zero warning
- butter filter error with 0 threshold
- target aligning prelim checks for files outside against dir
- Alignment cases with no matches

## [0.7.0] 2021 - 07 - 17

### Changed

- alignments are all positive now. You can easily align the files in a DAW by placing the files at the given time mark
- removed list around fingerprints recognition locality_frames_setting
- sped up finding audio files

### Added

- pretty printers for recognitions and alignments in audalign object
- recalc shifts from alignment results
- write_shifts_from_results. For use with recalc shifts or using different source files

## [0.6.1] 2021 - 07 - 13

### Changed

- sped up remove noise by about 80%

### Fixed

- doesn't try to write m4a's

## [0.6.0] 2021 - 07 - 07

### Added

- rank alignment added to all recognitions and alignments
- rank alignment function
- noise remove write extention
- noise remove destination directory support
- can't read ext in writes, so unsupported file types are written to wavs

### Changed

- raises error if destination directory doesn't exist at the start of alignments
- noise remove is prettier
- correcognize and correcognize spectrogram filter_matches defaults to 0
- updated docstrings

### Fixed

- inverted overlap ratio calculator
- Fixed bug with no results in correcognizes and visrecognize

## [0.5.2] 2021 - 06 - 22

### Fixed

- Correcognize offsets
- Correcognize_spectrogram offsets

### Changed

- Added multiprocessing to correcognize
- Added multiprocessing to correcognize_spectrogram

## [0.5.1] 2021 - 06 - 11

### Changed

- Sped up correlation max finding

### Added

- Correlation_Spectrogram: correlation technique, but with the spectrogram
- Better docs for align functions

### Fixed

- Plotter for correlations
- Recognize max_lags locality bug

## [0.5.0] 2021 - 06 - 04

### Changed

- Scaling factor in correlation normalized for length and bit depth
- Default locality_filter_prop lowered to include more results

### Added

- correlation locality

## [0.4.2] 2021 - 05 - 24

### Changed

- Multiprocessing for visual recognition/alignment on Linux works
- Multiprocessing for visual recognition/alignment optimized better for Windows

## [0.4.1] 2021 - 05 - 20

### Added

- Lots more neat tests

### Changed 
- multiprocessing works for Windows. Linux forces single threaded. 
  - By my tests, Windows multiprocessing and Linux were the same speed.

## [0.4.0] 2021 - 05 - 18

### Added

- fine_align: to give a precise alignment after a rough one
  - Lets you specify match_index for which rough alignment match to use
  - Lets you specify width of alignment for fine alignment
  - Lets you specify which alignment technique to use (visual not implemented yet)
- max_lags for recognize and correcognize

### Changed

- visrecognize to reduce pickling between processes
- pytest specifies tests dir; number of processes bumped to 10

## [0.3.1] 2021 - 05 - 12

### Added

- align_files: like align, but takes two or more filenames/paths to align
- align_files tests
- addopts to pytest.ini file

### Changed

- tests: all write to tmpdir except for saving fingerprints

### Fixed

- Pydub linux file reading bug: throws IndexError where it should be CouldntDecodeError

## [0.3.0] 2021 - 05 - 02

### Added

- Locality_filter_prop: filters locality tuples by proportion of highest confidence to tuple confidence within each offset
- Locality tuples include individual confidences at the end of each tuple

## [0.2.2] 2021 - 05 - 02

### Fixed

- Locality, improperly sorted through tuple noise

## [0.2.1] 2021 - 04 - 28

### Fixed

- Locality, wasn't creating correct tuples

## [0.2.0] 2021 - 04 - 14

### Added

- metadata function
- Correlation based alignment and correcognize
- frequency threshold getter and static setter

### Changed

- "use_fingerprints" to "technique" that takes a string
- changed filter fingerprints in align to none to tell if user input is supplied
- Bumped urllib and Pillow

## [0.1.6] 2021 - 02 - 26

### Fixed

- Messed up version nums

## [0.1.5] 2021 - 02 - 26

### Added 

- fingerprinting and alignment windows

### Changed

- Sped up fingerprinting in tests
- Documentation all same style

### Fixed

- write shifted files loophole
- read write destination

## [0.1.4] 2021 - 02 - 05

### Added

- Github Actions
- Locality Fingerprinting

### Fixed

- recognize bug
- tests

## [0.1.3] 2021 - 01 - 27

### Added

- Args and kwargs for noise reduce

### Changed

- Visual Alignment weighting

### Fixed 

- Try except for write in aligns

## [0.1.2] 2020 - 12 - 29

### Fixed

- Writing total track
- target align bug

## [0.1.1] 2020 - 12 - 22

### Added

- Visual Calc_mse
- Visual Image Resizing
- Visual Volume Floor

## [0.1.0] 2020 - 12 - 18

### Added

- Tons of Stuff
- Many Tests
- There are previous versions, but I don't think anybody would care about them
