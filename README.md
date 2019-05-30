# Datasets
- [TheBeatles180 Isophonics dataset](http://www.isophonics.net/content/reference-annotations-beatles) 
  - [audio](https://drive.google.com/open?id=1WzdcHeLeFHrYu_2_NDTEEglfrKLLKc1c) - extract in <i>data/audio</i>
  - [gt](https://drive.google.com/open?id=1EK59lTqt6iXN7ykdZXJyrYLuHmws36lh) - extract in <i>data</i>
  - [converted](https://drive.google.com/open?id=1Yh4dBJqtYkN7Hy5qp8E-dKeITX9D38I_) - extract in <i>data/converted</i>
  - [converted with librosa](https://drive.google.com/open?id=1hiTi_CPKxu9Qpli-zch1vINa4iY5iE9s)
  - [converted_list](https://drive.google.com/open?id=1E-TVqZvlFIJ2KzxmkkdhPxlKXxzQAcZJ) - extract to root folder
- [CaroleKingQueen Isophonic dataset](http://www.isophonics.net/content/reference-annotations-carole-king)
  - [audio](https://drive.google.com/open?id=1GVBNRwZ_YFHD9aroqP_NaI57H07_R3GR)
  - [converted with librosa](https://drive.google.com/open?id=1WbyWA4UcYuMHw7QvXMpb-i1PJrpKDKiZ)
  - [converted list](https://drive.google.com/open?id=1vMFUvgJrAzCsO4PjZ98pYkirGRGNt3Sk)
- [USPop2002](https://labrosa.ee.columbia.edu/projects/musicsim/uspop2002.html)
  - [audio](https://drive.google.com/open?id=161eEk-o1ulujRh_n-hYmQxlwdmbhWWja)
  - [converted with librosa](https://drive.google.com/open?id=1w8Mo2r6ml1v76SiU3MTjgoWiNjFLOH9a)
  - [converted list](https://drive.google.com/open?id=1V5zdvcB50YLfnTlsKyZspvTuVbWghrLE)
- JayChou29
  - [audio](https://drive.google.com/open?id=1s55LgFKyybeSueruV8Xvtwh6TE4yJnAb)
- Full datasets
  - [Full list](https://drive.google.com/open?id=1m8wC0vAc4p-HbNx68PH1gOfSKv2FE_EU)
  - [Full list with modulation](https://drive.google.com/open?id=1HSWo6Wv1fWmWjViN13TERxIgUpCMxoHe)
  - [Full dataset mod -1, -3](https://drive.google.com/file/d/19kqa5sZ7YwWd4KHZ8DdyUSP1eDkKnhiL)
  - [Full dataset mod 2, 4](https://drive.google.com/file/d/19kqa5sZ7YwWd4KHZ8DdyUSP1eDkKnhiL)
# Structure of repo:
- <b>data</b>
  - <b>audio</b> - default folder for raw audio files
  - <b>converted</b> - default folder for the saving preprocessed data in csv format
  - <b>gt</b> - contains .lab files with chords
  - <b>tracklists</b> - contains lists of paths to audio files starting with 'audio_root' parameter
  - <b>predicted</b> - contains predictions for TheBeatles180 and JayChou29 datasets, reports generated by MusOOEvaluator and pretrained models
# Installation
Download raw and converted datasets
```
bash ./data/download.sh
```
# Preprocessing
System computes notegramms (252 bins per sample) as described in Mauch 2010 (p.98) with <b>hop_length</b>=512, <b>sample_rate</b>=11025, <b>window_size</b>=4096
### How to use:
```
python preprocess.py --songs_list data/tracklists/TheBeatles180List
```
#### Parameters:
--songs_list<br>
--audio_root, default: data/audio/<br>
--gt_root, default: data/gt/<br>
--conv_root, default: data/converted/, determines folder where converted datasets will be saved<br>
--subsong_len, default: 40, length of song part in seconds to be splitted during preprocess<br> 
--song_len, default: 180 if <i>subsong_len</i> is not specified, song will be cutted or zeropaded to <i>song_len</i><br> 
--use_librosa, default: True, by default librosa.cqt will be used for preprocessing<br>
--songs_list, required, examples could be found in data/tracklists<br>
--num_bins, default: 84, defines number of bins during cqt<br>
--modulation_steps, list, default: [0], defines amount of modulation steps during CQT-preprocessing <br>
# Models:
## LSTM
  <br>Accuracy on test-set: 65%
### How to use:
```
python train_rnn.py --model LSTM --conv_list TheBeatles180List_converted_librosa.txt
```
#### Optional parameters:
--num_epochs, default: 2<br>
--learning_rate, default: 0.01<br>
--weight_decay, default: 1e-5<br>
--songs_list, default: data/tracklists/TheBeatles180List<br>
--audio_root, default: data/audio/<br>
--gt_root, default: data/gt/<br>
--conv_root, default: data/converted/, determins folder where converted datasets will be saved<br>
--conv_list, if specified, converted audio from list will be used for fitting model and convertation process will be skipped<br>
--category, default: MirexRoot<br>
--subsong_len, default: 40, length of song part in seconds to be splitted during preprocess<br> 
--song_len, default: 180 if <i>subsong_len</i> is not specified, song will be cutted or zeropaded to <i>song_len</i><br> 
--hidden_dim, default: 200<br>
--num_layers, default: 2<br>
--batch_size, default: 4<br>
--sch_step_size, default:100, scheduler step size<br>
--sch_gamma, default:100, scheduler's gamma<br>
--, default:10, test model on train and val datasets every n iterations <br>
--use_librosa, default: True <br>
--save_model_as, if specified, model will be saved in <i>pretrained</i> folder 
#### Test
```
python test_nn.py --model pretrained/LSTM_MirexRoot_TheBeatles180_librosa.pkl --conv_root data/converted/librosa --conv_list TheBeatles180List_converted_librosa.txt
```
## Random forest
  <br>Accuracy on test-set: Mirex_Root:55%<br>
### How to use:
#### Train
```
python train_rf.py
```
##### Optional parameters:
--songs_list, default: data/tracklists/TheBeatles180List<br>
--audio_root, default: data/audio/<br>
--gt_root, default: data/gt/<br>
--conv_root, default: data/converted/, determins folder where converted datasets will be saved<br>
--conv_list, if specified, converted audio from list will be used for fitting model and convertation process will be skipped<br>
--category, default: MirexRoot<br>
--subsong_len, default: 40, length of song part in seconds to be splitted during preprocess<br>
--song_len, default: 180 if <i>subsong_len</i> is not specified, song will be cutted or zeropaded to <i>song_len</i><br> 
--criterion, default: entropy<br>
--max_features, default: log2<br>
--n_estimators, default: 1<br>
#### Test
```
python test_rf.py --model pretrained/RF_MirexRoot_TheBeatles180_librosa.pkl --conv_root data/converted/librosa --conv_list TheBeatles180List_converted_librosa.txt
```
##### Optional parameters:
--songs_list, default: data/tracklists/TheBeatles180List<br>
--audio_root, default: data/audio/<br>
--gt_root, default: data/gt/<br>
--conv_root, default: data/converted/, determins folder where converted datasets will be saved<br>
--conv_list, if specified, converted audio from list will be used for fitting model and convertation process will be skipped<br>
--category, default: MirexRoot<br>
--subsong_len, default: 40, length of song part in seconds to be splitted during preprocess<br>
--song_len, default: 180 if <i>subsong_len</i> is not specified, song will be cutted or zeropaded to <i>song_len</i><br> 
