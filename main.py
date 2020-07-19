def main():
    import audalign as ad
    import time

    # TODO: New fingerprint algorithm
    # TODO: Document
    # TODO: type hints
    # TODO: complete align function
    # TODO: write aligned files function

    ada = ad.Audalign()  # "all_audio.json")
    # ada.write_processed_file("ResearchMaher/FraserSUB.mov", "processed_audio/FraserSUB.wav")
    t = time.time()
    ada.fingerprint_file("audio_files/TestAudio/Street.wav")
    # ada.multiprocessing = False
    # ada.fingerprint_directory("audio_files")
    t = time.time() - t
    print(f"It took {t} seconds to complete.")
    print(f"Total fingerprints: {ada.total_fingerprints}")
    print(
        f"Number of fingerprinted files: {len(ada.fingerprinted_files)} : {len(ada.file_names)}"
    )

    # ada.save_fingerprinted_files("all_audio.json")
    # ada.plot("audio_files/TestAudio/Paige.MOV")
    print(ada.recognize("audio_files/audio_sync/20200602/nr2.wav"))


if __name__ == "__main__":
    main()
