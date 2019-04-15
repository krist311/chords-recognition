Currently model fitted songwise, sequence length 3876
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
--conv_list, if specified, converted audio from list will be used for model and convertation process will be skipped<br>
--category, default: MirexRoot<br>
--subsong_len, default: 40, length of song part in seconds to be splitted during preprocess<br> 
--song_len, default: 180<br>
--hidden_dim, default: 200<br>
--num_layers, default: 2<br>
## Random forest
  <br>Accuracy on test-set: Mirex_Root:55% (learned 30% of TheBeatles180 dataset)<br>
### How to use:
train_rf.py
#### Optional parameters:
--songs_list, default: data/tracklists/TheBeatles180List<br>
--audio_root, default: data/audio/<br>
--gt_root, default: data/gt/<br>
--conv_root, default: data/converted/, determins folder where converted datasets will be saved<br>
--conv_list, if specified, converted audio from list will be used for model and convertation process will be skipped<br>
--category, default: MirexRoot<br>
--subsong_len, default: 40, length of song part in seconds to be splitted during preprocess<br>
--song_len, default: 180<br>
--criterion, default: entropy<br>
--max_features, default: log2<br>
--n_estimators, default: 1<br>
