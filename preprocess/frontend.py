import librosa
import numpy as np
from numpy.ma import sin
from scipy.fftpack import fft
from scipy.interpolate import interp1d
from scipy.signal.windows import hamming

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def preprocess_librosa(audiopath, feparam, n_bins=84, bins_per_octave=12, mod_steps=(0,)):
    x, sr = librosa.load(audiopath, feparam['fs'], mono=feparam['stereo_to_mono'])
    Xs = []
    tuning = librosa.estimate_tuning(y=x, sr=sr)
    for mod_step in mod_steps:
        X_pitched = librosa.effects.pitch_shift(x, sr, n_steps=mod_step)
        X = np.abs(librosa.core.cqt(X_pitched, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, tuning=tuning, window='hamming', norm=2))
        Xs.append(X.T)
    return Xs

def preprocess_mauch(audiopath, feparam, n_bins=84, bins_per_octave=12, mod_steps=(0,)):
    x, sr = librosa.load(audiopath, feparam['fs'], mono=feparam['stereo_to_mono'])
    # we differentiate tone and note in this program
    # by tone we mean 1 / 3 - semitone - wise frequency, by note we mean semitone - wise frequency
    fmin = 27.5  # MIDI note 21, Piano key number 1(A0)
    fmax = 3322  # MIDI note 104, Piano key number nnotes
    fratio = pow(2, 1 / (12 * 3))  # nsemitones = 3
    nnotes = 84
    ntones = nnotes * 3  # nsemitones = 3
    USR = 80  # upsampling rate

    # Needed only for plots
    Ms, Mc = toneProfileGen(feparam['overtoneS'], feparam['wl'], ntones, 3, fmin, fmax, fratio, feparam['fs'])
    # E = nnlsNoteProfile(feparam.overtoneS, nnotes, ntones)

    # note that the size of one col of the spectrogram will be nfft / 2 + 1,
    # where nfft = feparam.wl(the hamming window size)
    LE = logFreqNoteProfile(ntones, fmin, fratio, USR, feparam['fs'], nbins=int(feparam['wl'] / 2 + 1),
                            wl=feparam['wl'])

    # TODO lowcat as done in mySpectogram
    X = abs(librosa.stft(x, n_fft=feparam['wl'], hop_length=feparam['hop_size'], window='hamming'))

    # note that using cosine similarity method there are both simple tone
    # matrix (Ms) and complex tone matrix (Mc), the two are multiplied together
    # to form the salience matrix. Thus CosSim is an additive process.
    # also note that using log frequency spectrum method there is only one
    # salience matrix at this stage, and this is just similar to the simple
    # tone salience matrix. And the nnls process will infer note salience
    # matrix from this later. Thus logFreq is an deductive process.

    if feparam['enCosSim']:
        # calculating cosine similarity
        Ss = cosine_similarity(Ms, X)
        Sc = cosine_similarity(Mc, X)
    elif feparam['enlogFreq']:
        Ss = LE.dot(X)

    if feparam['tuningBefore']:
        Ss, et = phase_tuning(Ss)
        if feparam['enCosSim']:
            Sc = tuning_update(Sc, et)
    return Ss.T


def logFreqNoteProfile(ntones, fmin, fratio, USR, fs, nbins, wl):
    """this matrix can transform original linear space DFT (Hz bin) to log-freq
    space DFT (1/3 semitone bin) by means of cosine interpolations
    this process is implemented as described in M. Mauch's thesis
    first the original 2048-bin DFT is upsampled linearly 40 times
    then the upsampled spectrum is downsampled non-linearly with a constant-Q
    Mauch p.99"""

    df = fs / wl  # not sure that there should be fs/nbins, according to Mauch it's fs/wl
    fi = (fs / 2) * np.linspace(0, 1, nbins)
    ff = (fs / 2) * np.linspace(0, 1, nbins * USR)
    df_f = ff / 51.94  # df(f) = f / Q, where Q = 36 / ln2 = 51.94
    fk = fmin * (np.power(fratio, np.arange(ntones) - 1))

    # matrix to transform linear DFT to upsampled linear DFT
    # size of the matrix is num_bins * (USR * num_bins)
    # therefore by multiplying this matrix, a num_bins DFT can be transformed to
    # an upsampled USR * num_bins bins
    # every row of the matrix is the cosine interpolation of the correponding bin
    h = []
    for i in np.arange(nbins):
        hf = np.zeros(USR * nbins)
        ind_to = np.arange(len(ff))[ff < fi[i] + df]
        ind_from = np.arange(len(ff))[ff > fi[i] - df]
        ind = np.intersect1d(ind_from, ind_to)
        # ind = [j for j, v in enumerate(ff) if fi[i] - df < v < fi[i] + df]
        # in junqi's thesis there is 2*pi instead of pi, but it is wrong in terms of cosine interpolation
        hf[ind] = (np.cos(np.pi * (ff[ind] - fi[i]) / df) / 2) + 0.5
        h.append(hf)

    # matrix to transform upsampled DFT to constant-Q DFT
    # size of the matrix is (USR*numbins) * numtones
    # by multiplying the matrix, a (USR*numbins) DFT can be transformed to a
    # numtones bin
    hl = []
    for i in range(USR * nbins):
        hlf = np.zeros(ntones)
        ind_to = np.arange(len(fk))[fk < ff[i] + df_f[i]]
        ind_from = np.arange(len(fk))[fk > ff[i] - df_f[i]]
        ind = np.intersect1d(ind_from, ind_to)
        # ind = [j for j, v in enumerate(fk) if ff[i] - df_f[i] < v < ff[i] + df_f[i]]
        hlf[ind] = np.cos(np.pi * (ff[i] - fk[ind]) / df_f[i]) / 2 + 0.5
        hl.append(hlf)

    # the final matrix is the product of the above two matrice,
    # which will be a matrix of size numbins * numtones
    # this matrix is a transformer to transform a vector of numbins into a
    # vector of numtones, which equals to linear freq to log freq.
    LE = (np.array(h).dot(np.array(hl))).T
    # normalize this transformation matrix row wise (use L1 norm)
    LE = normalize(LE, axis=0, norm='l1')
    return LE


def sin_in_window(wl, f_tone, fs):
    return sin(2 * np.pi * np.arange(1, wl + 1) * f_tone / fs)


def toneProfileGen(s, wl, numtones, numsemitones, fmin, fmax, fratio, fs):
    """build the tone profiles for calculating note salience matrix
    each sinusoidal at frequency 'ftone' is generated via sin(2*pi*n*f/fs)
    the true frequency of the tone is supposed to lie on bin notenum*numsemitones-1,
    e.g. A4 is bin 49*numsemitones-1 = 146, C4 is bin 40*numsemitones-1 = 119 (note that notenum is
    not midinum, note num is the index of the key on an 88 key piano with A0 = 1)
    Mauch p.57"""
    w = hamming(wl)
    Ms, Mc = [], []  # simple and complex tone profiles
    for tone_id in range(numtones):
        f_tone = fmin * (pow(fratio, (tone_id - 1)))  # frequency of current tone
        s_tone = sin_in_window(wl, f_tone, fs) * w.T
        c_tone = (pow(s, 0) * sin_in_window(wl, f_tone, fs) + pow(s, 1) * sin_in_window(wl, 2 * f_tone, fs)
                  + pow(s, 2) * sin_in_window(wl, 3 * f_tone, fs) + pow(s, 3) * sin_in_window(wl, 4 * f_tone, fs)) * w.T

        fft_tone = fft(s_tone)[:int(wl / 2)]
        fft_ctone = fft(c_tone)[:int(wl / 2)]

        Ms.append(fft_tone)
        Mc.append(fft_ctone)
    return Ms, Mc


def phase_tuning(S):
    """tuning based on phase information
    assuming nsemitones = 3"""
    nslices = S.shape[1]
    ntones = S.shape[0]

    Sbar = sum(S, 2) / nslices  # make the length of Sbar divisable by 3?
    dftSbar = fft(Sbar)

    tp = len(dftSbar) * (1 / 3) + 1
    phiSbar = np.angle(dftSbar)

    phi = np.interp(tp, range(1, len(phiSbar) + 1), phiSbar)

    delta = wrapd(- phi - 2 * np.pi / 3) / (2 * np.pi)

    st = 440
    et = st * pow(2, delta / 12)

    # then interpolate the original S at every x position throughout
    # nslices * ntones, the edge values are just interpolated as zeros
    S = tuning_update(S, et)
    return S, et


def tuning_update(S, et):
    """interpolate the original S at every x position throughout
    nslices*ntones, the edge values are just interpolated as zeros

    now modify S based on the new tuning of the song, so that
    the middle bin of each semitone corresponds to the semitone frequency in
    the estimated tuning
    let's treat the problem in this way:
    st = standard tuning = 440Hz
    et = estimated tuning
    now we wonder what's the position of the et related to st in the axis of
    3 semitones per bin in terms of bin
    so we have st * (2 ^ (p / 36)) = et
    so p = (log(et / st) / log(2)) * 36"""
    st = 440
    p = (np.log(et / st) / np.log(2)) * 36

    ntones, nslices = S.shape
    for j in range(nslices):
        # insert two zeros on both side to serve the possible p = +/- 1.5
        # thus the orginal index x = [-1,0,1,2,...,ntones, ntones + 1, ntones + 2];
        # and interpolation index xi = 1:ntones + p;
        # y = [0;0;S(:,j);0;0];
        # x = (-1:ntones+2)';
        y = S[:, j]
        x = np.arange(ntones).T
        xi = np.arange(ntones).T + p
        yi = interp1d(x, y, kind='linear', fill_value='extrapolate')(xi)
        S[:, j] = yi
    return S


def wrapd(angle):
    """Wrapped phase means that all phase points are constrained to the range
     -180 degrees ? Phase Offset < 180 degrees. When the actual phase is outside this range,
     the phase value is increased or decreased by a multiple of 360 degrees to put the phase value
     within +/- 180 degrees of the Phase Offset value."""
    while not -np.pi <= angle < np.pi:
        if angle < -np.pi:
            angle = angle + 2 * np.pi
        elif angle >= np.pi:
            angle = angle - 2 * np.pi
    return angle
