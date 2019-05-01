#!/usr/bin/env bash
bash ./data/download.sh

#GRU
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4


python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4



python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
#lstm SGD
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4




python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4



python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

#BIGRU

python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4


python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4



python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
#GRU SGD
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4




python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4



python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model GRU --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4







#lstm
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4


python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4



python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
#lstm SGD
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4




python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4



python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional False --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

#BILSTM
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4


python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4



python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
#LSTM SGD
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 3 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4




python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 5 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4



python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 1e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-3

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 40 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 60 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4

python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.05 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.01 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.005 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4
python train_rnn.py --model LSTM --opt SGD --bidirectional True --hidden_dim 84 --val_step 200 --num_layers 1 --lr 0.001 --conv_list FullList386_converted_librosa.txt --num_epochs 100 --sch_step_size 30 --sch_gamma 0.1 --save_model True --weight_decay 5e-4



