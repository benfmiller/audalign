def main():
    import audalign as ad
    import time

    # TODO: improve fingerprint algorithm
    # TODO: type hint
    # TODO: improve alignment

    ada = ad.Audalign()  # "all_audio.json")
    t = time.time()
    # result = ada.recognize("audio_files/TestAudio/test.wav")
    # ada.fingerprint_file("audio_files/TestAudio/Street.wav")
    # ada.convert_audio_file("audio_files/shifts/Eigen-song-base.wav", "Eigen-song-base.mp3")
    # ada.fingerprint_directory("audio_files/processed_audio")
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
