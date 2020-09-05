def main():
    import audalign as ad
    import time

    # TODO: New fingerprint algorithm
    # TODO: Document and type hint
    # TODO: complete align function
    # TODO: write aligned files function
    # TODO: add location information to fingerprint

    ada = ad.Audalign()  # "all_audio.json")
    t = time.time()
    # result = ada.recognize("audio_files/TestAudio/test.wav")
    # ada.fingerprint_file("audio_files/TestAudio/Street.wav")
    # ada.multiprocessing = False
    # ada.fingerprint_directory("audio_files")
    print(ada.align("audio_files/shifts", "test_alignment"))
    t = time.time() - t
    # print(f"It took {t} seconds to complete.")
    # print(f"Total fingerprints: {ada.total_fingerprints}")
    # print(
    # f"Number of fingerprinted files: {len(ada.fingerprinted_files)} : {len(ada.file_names)}"
    # )

    # ada.save_fingerprinted_files("all_audio.json")
    # ada.plot("audio_files/audio_sync/20200602/nr1.wav")
    # print(ada.recognize("audio_files/audio_sync/20200602/nr2.wav"))


if __name__ == "__main__":
    main()
