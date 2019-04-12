def root_params():
    feparam = {
        # input parameters
        'stereo_to_mono': True,
        # spectorgam parameters
        'fs': 11025,
        'wl': 4096,
        'hop_size': 512,
        'trim_to_len': 11025*60*3,# 3 minutes
        # log-frequency spectrum
        'enlogFreq': True,
        'enCosSim': False,
        'overtoneS': 0.7,
        # tuning selection
        'tuningBefore': 1,
        # noise reduction
        'specRollOn': 0.01,
        'enPeakNoiseRed': 0,
        # tone level gestaltize
        'wgmax': 10,
        'enToneGesComp': 0,
        'enToneGesRed': 0,
        # standardization
        'enSUB': 1,
        'enSTD': 1,
        'stdwr': 18,
        'specWhitening': 1,
        # 3 semitone -> 1 semitone processes
        'enSigifBins': 1,
        'enNNLS': 1,
        'enCenterbin': 0,
        'en3bin': 0,
        # note level gestalize (wgmax is set above)
        'enNoteGesComp': 0,
        'enNoteGesRed': 0,
        # segmentation
        'noSegmentation': 1,
        'useBassOnsetSegment': 0,
        'useBeatSyncSegment': 0,
        # filter selection
        'useMedianFilter': 1,
        'useMeanFilter': 0,
        # bass, treble profiling
        'basstreblechromagram': 1,
        'baseuppergram': 0,
        # window selection
        'enProfileHann': 0,
        'enProfileHamming': 0,
        'enProfileRayleigh': 1,
        # normalization
        'normalization': 1  # Inf change than you'll realize what does it mean
    }
    # ****** Back-end control ****** %

    beparam = {
        # normalization
        'normalization': 1,  # Inf change than you'll realize what does it mean
        'useSIM1': 0,  # work together with 'BassOnsetSegment'
        'useDBN1': 1,
        'useDBN2': 0,
        'enCast2MajMin': 0,
        'enChordGestalt': 1,
        'enCombSameChords': 1,
        'enEliminShortChords': 1,
        'enMergeSimilarChords': 0,
        'grainsize': 1,
        'enBassTrebleCorrect': 1,  # work together with 'noSegmentation'
        'btcVersion': 3
    }
    dbnparam = {
        'muCBass': 1,
        'muNCBass': 1,
        'muTreble': 1,
        'muNoChord': 1,
        'sigma2Treble': 0.2,
        'sigma2CBass': 0.1,
        'sigma2NCBass': 0.5,
        'sigma2NoChord': 0.2,
        'selfTrans': 1e12
    }
    dbn2param = {
        'mu': 1,
        'sigma': 0.1,
        'wTreble': 1,
        'wCBass': 1,
        'wNCBass': 0.5,
        'selfTrans': 1e12
    }
    #avaliable options: MirexRoot, MirexMajMin, MirexMajMinBass, MirexSevenths, MirexSeventhsBass. The same as in MusOOEvaluator.
    mirex_category = "MirexRoot"
    label_size = 13

    return feparam, beparam, dbnparam, dbn2param, mirex_category, label_size


