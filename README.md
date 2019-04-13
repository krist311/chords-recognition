Currently model fitted songwise, sequence length 3876
## Models:
- One layer LSTM
  <br>Accuracy on test-set: to be determined
## How to use:
train.py --model LSTM
#### Optional parameters:
--num_epochs, default: 2<br>
--learning_rate, default: 0.01<br>
--weight_decay, default: 1e-5
--songs_list, default: data/tracklists/TheBeatles180List
--audio_root, default: data/audio/
--gt_root, default: data/gt/
--conv_root, default: data/converted/, determins folder where converted datasets will be saved
--conv_list, if specified, converted audio from list will be used for model and convertation process will be skipped
--category, default: MirexRoot
