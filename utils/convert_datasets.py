import numpy as np
import re
import os


class Chord:
    # Data struct to include start, end, label.
    pass


def chord_to_num(chord_name):
    NOTES = {'N': 0, 'Silence': 0, 'X': 0, 'C': 1, 'C#': 2, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 4,
             'E': 5, 'Fb': 5, 'F': 6, 'F#': 7, 'Gb': 7, 'G': 8, 'G#': 9, 'Ab': 9, 'A': 10, 'A#': 11, 'Bb': 11, 'B': 12,
             'Cb': 12}
    return NOTES[chord_name]


def parse_chords(filename):
    # Opens a mirex chord file and returns a list with chord objects
    song = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                start, end, label = line.strip().split('\t')
            except:
                try:
                    start, end, label = line.split(' ')
                except:
                    pass
            label = label.strip('\n')
            label = re.split(':', label)[0]  # take only first root note of chord
            label = re.split('/', label)[0]
            chord = Chord()
            chord.start = float(start)
            chord.end = float(end)
            chord.label = chord_to_num(label)
            song.append(chord)
    return song


def audio_to_cqt(filename, n_bins=24):
    import librosa
    import numpy as np
    y, sr = librosa.load(filename)
    # I am not sure yet how to choose hop_size if samplerate is different from 44100, thus there will be magic numbers
    hop_length = int((sr / 44100) * 2048)
    C = np.abs(librosa.cqt(y, sr=sr, n_bins=n_bins, hop_length=hop_length))
    # Transform C to make it the same dimensions as in billboard dataset (each row contains bins for current sample)
    return C.T, sr, hop_length


def match_chords_and_cqt(cqt, chords_filename, sample_rate, cqt_hop_length, out_file=None):
    converted_dataset = []
    chords = parse_chords(chords_filename)
    sample_time = cqt_hop_length / sample_rate
    for chord in chords:
        chord_start_in_samples = round(chord.start / sample_time)
        chord_end_in_samples = round(chord.end / sample_time)
        samples_per_chord = cqt[chord_start_in_samples:chord_end_in_samples]
        converted_dataset.extend(np.c_[samples_per_chord, len(samples_per_chord) * [chord.label]])
    if out_file:
        np.savetxt(out_file, converted_dataset, delimiter=",", fmt='%s')
    return converted_dataset


def convert_song(audio_file, chords_file):
    cqt, sr, hop_length = audio_to_cqt(audio_file, n_bins=24)
    return match_chords_and_cqt(cqt, chords_file, sr, hop_length)


def convert_billboard_dataset(folder_name):
    converted_dataset = []
    for song_name in os.listdir(folder_name + "chromabins"):
        chords_filename = folder_name + 'chords/' + song_name + '/full.lab'
        chroma_filename = folder_name + 'chromabins/' + song_name + '/bothchroma.csv'
        cqt = np.genfromtxt(chroma_filename, delimiter=',')[:, 2:]
        converted_dataset.extend(
            match_chords_and_cqt(cqt, chords_filename, sample_rate=44100, cqt_hop_length=2048))
    np.savetxt(folder_name + "billboard_converted.csv", converted_dataset, delimiter=",", fmt='%s')


def convert_isophonics_dataset(folder_name, *bands):
    for band in bands:
        converted_dataset = []
        for album in os.listdir(folder_name + band + '/audio/'):
            print(band + album)
            for song in os.listdir(folder_name + band + '/audio/' + album):
                converted_dataset.extend(
                    convert_song(folder_name + '/' + band + '/audio/' + album + '/' + song,
                                 folder_name + '/' + band + '/chords/' + album + '/' + os.path.splitext(song)[0] + '.lab'))
        np.savetxt(folder_name + '/' + band + '/' + band + '_isophonic_converted.csv', converted_dataset,
                   delimiter=",", fmt='%s')


if __name__ == '__main__':
    #billboard_path = "C:/Users/Daniil/Documents/Git/vkr/data/billboard/"
    isophonic_path = "C:/Users/Daniil/Documents/Git/vkr/data/isophonic/"
    #convert_billboard_dataset(billboard_path)
    convert_isophonics_dataset(isophonic_path,'The Beatles')
