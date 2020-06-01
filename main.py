def main():
    import audalign as ad

    # TODO: Multiprocessing
    # TODO: Not double add files
    # TODO: Add precision optional adjustments to recognize
    # TODO: Document
    # TODO: Add plot function
    # TODO: Add uniquehashes and filename fields

    ada = ad.Audalign("TestAudio.json")

    # ada.save_fingerprinted_files("asdf.")
    # ada.fingerprint_directory("TestAudio")
    # ada.save_fingerprinted_files("TestAudio.json")
    print(ada.recognize("ResearchMaher/BenStreet.mp4"))
    # print(len(ada.fingerprinted_files[0][1]))
    # djv.save_fingerprinted_files('Sub.json')
    # print(len(djv.fingerprinted_files))

    # print(djv.fingerprinted_files[0][0] + "  :  " + djv.fingerprinted_files[0][2])
    # a = djv.fingerprinted_files
    # b = a[0][1]
    # print(b)

    # print("\nBeginning Recognizing")
    # print(djv.recognize(recognizer='MicrophoneRecognizer', seconds=10))
    # print(djv.recognize("SUB.mp3"))

    # djv.save_fingerprinted_files('test_mp3s.pickle')

    # djv.fingerprint_directory("mp3")
    # print(djv.fingerprinted_files)

    # filen = djv.recognize("mp3/Sean-Fournier--Falling-For-You.mp3")
    # print(filen)
    # print ("From file we recognized: %s\n" % filen)


if __name__ == "__main__":
    main()
