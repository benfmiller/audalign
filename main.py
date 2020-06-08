def main():
    import audalign as ad
    import time

    # TODO: New fingerprint algorithm
    # TODO: Not double add files
    # TODO: Document
    # TODO: Add uniquehashes and filename fields

    ada = ad.Audalign("all_audio.json")
    # ada.write_processed_file("ResearchMaher/FraserSUB.mov", "processed_audio/FraserSUB.wav")
    t = time.time()
    # ada.fingerprint_directory("audio_files")
    t = time.time() - t
    print(f"It took {t} seconds to complete.")
    print(f"Total fingerprints: {ada.total_fingerprints}")
    # ada.save_fingerprinted_files("all_audio.json")
    ada.plot("audio_files/TestAdio/Paige.MOV")
    print(ada.recognize("audio_files/TestAudio/Rachel.mp4"))


if __name__ == "__main__":
    main()
