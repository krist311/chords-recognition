import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import glog as log
from pprint import pformat

from preprocess.chords import TypesConverter, chord_nums_to_inds

from models import LSTMClassifier, GRUClassifier
from dataloader import get_train_val_seq_dataloader
from preprocess.generators import gen_train_data
import sys

from utils.parser import get_train_rnn_parser
from utils.utils import get_params_by_category

torch.set_default_dtype(torch.float64)


def val_muiltiple_nets(root_model, type_model, val_loader, category, bass_model=None):
    root_model.eval()
    type_model.eval()
    correct, total, acc = 0, 0, 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels, lengths = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            root_outputs = root_model(inputs, lengths)
            root_predicted = root_outputs.topk(1, dim=2)[1].squeeze().view(-1)
            type_outputs = type_model(inputs, lengths)
            # make final prediction
            chord_nums = []
            if bass_model:
                type_predicted = type_outputs.topk(2, dim=2)[1].view(-1, 2)
                bass_model.eval()
                bass_outputs = bass_model(inputs, lengths)
                bass_predicted = bass_outputs.topk(2, dim=2)[1].view(-1, 2)
                for root, chord_type, bass in zip(root_predicted, type_predicted, bass_predicted):
                    if not root:
                        chord_nums.append('0')
                    else:
                        if not chord_type[0]:
                            chord_type[0] = chord_type[1]
                        chord_type_conv = TypesConverter.ind_to_type(chord_type[0].item())
                        bass_conv = TypesConverter.ind_to_bass(bass[0].item())
                        if bass_conv == '7' and '7' not in chord_type_conv:
                            if '7' in TypesConverter.ind_to_type(chord_type[1].item()):
                                chord_type_conv = TypesConverter.ind_to_type(chord_type[1].item())
                            else:
                                bass_conv = TypesConverter.ind_to_bass(bass[1].item())
                        if ('min' in chord_type_conv and (bass_conv == '3' or bass_conv == '7')) or (
                                chord_type_conv == '7' and bass_conv == '7'):
                            bass_conv = f"b{bass_conv}"
                        chord_nums.append(f"{root}:{chord_type_conv}{f'/{bass_conv}' if bass_conv else ''}")
            else:
                type_predicted = type_outputs.topk(1, dim=2)[1].squeeze().view(-1)
                for root, chord_type in zip(root_predicted, type_predicted):
                    if not root or not chord_type:
                        chord_nums.append('0')
                    else:
                        chord_nums.append(f"{root}:{TypesConverter.ind_to_type(chord_type.item())}")
            predicted = chord_nums_to_inds(chord_nums, category)

            labels = labels.view(-1)
            predicted = torch.tensor(predicted)[labels >= 0]
            labels = labels[labels >= 0]

            total += len(labels)
            correct += (predicted == labels).sum().item()
    if total:
        acc = 100 * correct / total
        print(f'Total acc: {acc}')


def train_nets(args):
    if args.category == 'MirexRoot':
        train(args)
    elif args.category == 'MirexMajMin':
        if args.multiple_nets:
            root_model = train(args, 'MirexRoot')
            maj_min_model = train(args, 'maj_min')
            params, num_classes, y_ind = get_params_by_category(args.category)
            _, val_loader = get_train_val_seq_dataloader(args.conv_list, args.batch_size, y_ind)
            val_muiltiple_nets(root_model, maj_min_model, val_loader, args.category)
        else:
            train(args)
    elif args.category == 'MirexMajMinBass':
        if args.multiple_nets:
            root_model = train(args, 'MirexRoot')
            maj_min_model = train(args, 'maj_min')
            bass_model = train(args, 'bass')
            params, num_classes, y_ind = get_params_by_category(args.category)
            _, val_loader = get_train_val_seq_dataloader(args.conv_list, args.batch_size, y_ind)
            val_muiltiple_nets(root_model, maj_min_model, val_loader, args.category, bass_model)
        else:
            train(args)
    elif args.category == 'MirexSevenths':
        if args.multiple_nets:
            root_model = train(args, 'MirexRoot')
            maj_min_7_model = train(args, 'maj_min_7')
            params, num_classes, y_ind = get_params_by_category(args.category)
            _, val_loader = get_train_val_seq_dataloader(args.conv_list, args.batch_size, y_ind)
            val_muiltiple_nets(root_model, maj_min_7_model, val_loader, args.category)
        else:
            train(args)
    elif args.category == 'MirexSeventhsBass':
        if args.multiple_nets:
            root_model = train(args, 'MirexRoot')
            maj_min_7_model = train(args, 'maj_min_7')
            bass_model = train(args, 'bass7')
            params, num_classes, y_ind = get_params_by_category(args.category)
            _, val_loader = get_train_val_seq_dataloader(args.conv_list, args.batch_size, y_ind)
            val_muiltiple_nets(root_model, maj_min_7_model, val_loader, args.category, bass_model)
        else:
            train(args)


def train(args, category=None):
    # prepare train dataset
    params, num_classes, y_ind = get_params_by_category(category if category else args.category)
    conv_root = args.conv_root
    if args.use_librosa:
        conv_root = conv_root + '/librosa/'
        input_size = 84
    else:
        conv_root = conv_root + '/mauch/'
        input_size = 252
    conv_list = args.conv_list
    # generate train data from audio dataset if not specified
    if not conv_list:
        conv_list = gen_train_data(args.songs_list, args.audio_root, args.gt_root, params, conv_root,
                                   args.subsong_len, args.song_len, args.use_librosa)

    # check if gpu available
    use_gpu = torch.cuda.is_available()
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
    # create model
    if args.model == 'LSTM':
        model = LSTMClassifier(input_size=input_size, hidden_dim=args.hidden_dim, output_size=num_classes,
                               num_layers=args.num_layers,
                               use_gpu=use_gpu, bidirectional=args.bidirectional, dropout=args.dropout)
        if args.checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
    elif args.model == 'GRU':
        model = GRUClassifier(input_size=input_size, hidden_dim=args.hidden_dim, output_size=num_classes,
                              num_layers=args.num_layers,
                              use_gpu=use_gpu, bidirectional=args.bidirectional, dropout=args.dropout)
        if args.checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
    if use_gpu:
        model = model.cuda()



    train_loader, val_loader = get_train_val_seq_dataloader(conv_list, args.batch_size, y_ind)
    log_path = './logs/{:%Y_%m_%d_%H_%M}_{}'.format(datetime.datetime.now(), args.model)
    writer = SummaryWriter(log_path)
    loss_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    if args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] if checkpoint else 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sch_step_size, gamma=args.sch_gamma)
    if checkpoint:
        scheduler.load_state_dict(checkpoint['sch_state_dict'])
    model.train()
    with tqdm(total=len(train_loader) * (args.num_epochs-start_epoch)) as pbar:
        running_loss = 0.0
        for epoch in range(start_epoch+1,args.num_epochs):
            scheduler.step()
            for i, data in enumerate(train_loader, 1):
                iteration = epoch * len(train_loader) + i
                inputs, labels, lengths = data
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs, lengths)
                outputs = outputs.view(-1, outputs.size(2))
                labels = labels.view(-1)
                loss = loss_criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if iteration % args.val_step == 0:
                    # print statistics
                    train_acc = val_model(model, train_loader, num_classes)
                    val_acc = val_model(model, val_loader, num_classes)
                    av_loss = running_loss / args.val_step
                    if writer:
                        write_results(writer, av_loss, iteration, model, train_acc, val_acc)
                    print_results(i, epoch, av_loss, train_acc, val_acc)
                    running_loss = 0.0
                    model.train()
                pbar.update()
                # save checkpoint every 10 epochs, necessary because cloud could be disconnected any time
            if epoch %10 ==0:
                import numpy as np
                rand = np.random.randint(50)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'sch_state_dict': scheduler.state_dict()
                }, f'checkpoint{rand}.tar')
                print(f'saving as checkpoint{rand}.tar')

            # disable dropout on last 10 epochs
            if args.num_epochs - epoch == 10:
                model.disable_dropout()

    log.info('Finished Training')
    acc = val_model(model, val_loader, num_classes, print_results=True)

    # save pretrained model
    if args.save_model:
        torch.save(model.state_dict(),
                   f"pretrained/{args.model}_bi_{args.bidirectional}_{args.category}_{'librosa' if args.use_librosa else 'mauch'}_acc_"
                   f"{acc}_lr_{args.lr}_wd_{args.weight_decay}_nl_{args.num_layers}_hd_{args.hidden_dim}_ne_{args.num_epochs}"
                   f"_sss_{args.sch_step_size}_sg_{args.sch_gamma}_opt_{args.opt}")
    return model


def val_model(model, test_loader, num_classes, print_results=False):
    model.eval()
    correct, total, acc = 0, 0, 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, lengths = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs, lengths)
            predicted = outputs.topk(1, dim=2)[1].squeeze().view(-1)
            labels = labels.view(-1)
            predicted = predicted[labels >= 0]
            labels = labels[labels >= 0]
            total += len(labels)
            correct += (predicted == labels).sum().item()
    if total:
        acc = 100 * correct / total
    if print_results:
        log.info(f'Val acc: {acc}')
    return acc


def print_results(iter, epoch, loss, train_acc, val_acc):
    log.info('[%d, %5d] loss: %f train_acc: %.3f, val_acc: %.3f' %
             (epoch + 1, iter + 1, loss, train_acc, val_acc))


def write_results(tensorboard_writer, loss, i, model, train_acc, test_acc):
    tensorboard_writer.add_scalar('data/loss', loss, i)
    tensorboard_writer.add_scalar('data/train_acc', train_acc, i)
    tensorboard_writer.add_scalar('data/val_acc', test_acc, i)
    for name, param in model.named_parameters():
        tensorboard_writer.add_histogram(name, param.clone().cpu().data.numpy(), i)


if __name__ == '__main__':
    parser = get_train_rnn_parser()
    args = parser.parse_args(sys.argv[1:])
    log.info('Arguments:\n' + pformat(args.__dict__))
    train_nets(args)
