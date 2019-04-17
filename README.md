# Todo
- [ ] demo
# Datasets
- [TheBeatles180 Isophonics dataset](http://www.isophonics.net/content/reference-annotations-beatles) 
  - [audio](https://drive.google.com/open?id=1WzdcHeLeFHrYu_2_NDTEEglfrKLLKc1c) - extract in <i>data/audio</i>
  - [gt](https://drive.google.com/open?id=1EK59lTqt6iXN7ykdZXJyrYLuHmws36lh) - extract in <i>data</i>
  - [converted](https://drive.google.com/open?id=1Yh4dBJqtYkN7Hy5qp8E-dKeITX9D38I_) - extract in <i>data/converted</i>
  - [converted_list](https://drive.google.com/open?id=1E-TVqZvlFIJ2KzxmkkdhPxlKXxzQAcZJ) - extract to root folder
# Structure of repo:
- <b>data</b>
  - <b>audio</b> - default folder for raw audio files
  - <b>converted</b> - default folder for the saving preprocessed data in csv format
  - <b>gt</b> - contains .lab files with chords
  - <b>tracklists</b> - contains lists of paths to audio files starting with 'audio_root' parameter
# Preprocessing
System computes notegramms (252 bins per sample) as described in Mauch 2010 (p.98) with <b>hop_length</b>=512, <b>sample_rate</b>=11025, <b>window_size</b>=4096
### How to use:
preprocess.py 
#### Optional parameters:
--songs_list, default: data/tracklists/TheBeatles180List<br>
--audio_root, default: data/audio/<br>
--gt_root, default: data/gt/<br>
--conv_root, default: data/converted/, determins folder where converted datasets will be saved<br>
--category, default: MirexRoot<br>
--subsong_len, default: 40, length of song part in seconds to be splitted during preprocess<br> 
--song_len, default: 180 if <i>subsong_len</i> is not specified, song will be cutted or zeropaded to <i>song_len</i><br> 
# Models:
## LSTM
  <br>Accuracy on test-set: to be determined
### How to use:
train_nn.py --model LSTM
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
## Random forest
  <br>Accuracy on test-set: Mirex_Root:55% (learned 30% of TheBeatles180 dataset)<br>
### How to use:
#### Train
train_rf.py
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
test_rf.py --model [Pretrained model in cPickle format]
##### Optional parameters:
--model default: pretrained/RF_MirexRoot_TheBeatles180.pkl, pretrained model in cPickle format<br>
--songs_list, default: data/tracklists/TheBeatles180List<br>
--audio_root, default: data/audio/<br>
--gt_root, default: data/gt/<br>
--conv_root, default: data/converted/, determins folder where converted datasets will be saved<br>
--conv_list, if specified, converted audio from list will be used for fitting model and convertation process will be skipped<br>
--category, default: MirexRoot<br>
--subsong_len, default: 40, length of song part in seconds to be splitted during preprocess<br>
--song_len, default: 180 if <i>subsong_len</i> is not specified, song will be cutted or zeropaded to <i>song_len</i><br> 
