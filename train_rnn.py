import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import glog as log
from pprint import pformat

from models import LSTMClassifier, GRUClassifier
from dataloader import get_train_val_seq_dataloader
from preprocess.chords import preds_to_lab
from preprocess.generators import gen_test_data, gen_train_data
import sys

from utils.parser import get_train_rnn_parser
from utils.utils import get_params_by_category, save_model

torch.set_default_dtype(torch.float64)


def train(args):
    # prepare train dataset
    params, num_classes = get_params_by_category(args.category)
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

    # create model
    if args.model == 'LSTM':
        model = LSTMClassifier(input_size=input_size, hidden_dim=args.hidden_dim, output_size=num_classes,
                               num_layers=args.num_layers,
                               use_gpu=use_gpu, bidirectional=args.bidirectional, dropout=args.dropout)
    elif args.model == 'GRU':
        model = GRUClassifier(input_size=input_size, hidden_dim=args.hidden_dim, output_size=num_classes,
                              num_layers=args.num_layers,
                              use_gpu=use_gpu, bidirectional=args.bidirectional)
    if use_gpu:
        model = model.cuda()
    train_loader, val_loader = get_train_val_seq_dataloader(conv_list, args.batch_size)
    log_path = './logs/{:%Y_%m_%d_%H_%M}_{}'.format(datetime.datetime.now(), args.model)
    writer = SummaryWriter(log_path)
    loss_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    if args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sch_step_size, gamma=args.sch_gamma)
    model.train()
    with tqdm(total=len(train_loader) * args.num_epochs) as pbar:
        for epoch in range(args.num_epochs):
            running_loss = 0.0
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
            scheduler.step()
            # disable dropout on last 10 epochs
            if args.num_epochs - epoch == 10:
                model.disable_dropout()

    log.info('Finished Training')
    acc = val_model(model, val_loader, num_classes, print_results=True)
    # save pretrained model
    if args.save_model:
        torch.save(model,
                   f"pretrained/{args.model}_bi_{args.bidirectional}_{args.category}_{'librosa' if args.use_librosa else 'mauch'}_acc:"
                   f"{acc}_lr:{args.lr}_wd:{args.weight_decay}_nl:{args.num_layers}_hd:{args.hidden_dim}_ne:{args.num_epochs}"
                   f"_sss:{args.sch_step_size}_sg:{args.sch_gamma}_opt_{args.opt}")


def val_model(model, test_loader, num_classes, print_results=False):
    def count_scores(pred, corr, scores):
        pred = pred.view(-1)
        corr = corr.view(-1)
        for c, p in zip(corr, pred):
            scores[c][p] += 1

    def draw_results():
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        # Normalize by dividing every row by its sum
        for i in range(num_classes):
            scores[i] = scores[i] / scores[i].sum()

        # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(scores.numpy())
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + list(range(num_classes)), rotation=90)
        ax.set_yticklabels([''] + list(range(num_classes)))

        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # sphinx_gallery_thumbnail_number = 2
        plt.show()

    model.eval()
    correct, total, acc = 0, 0, 0
    scores = torch.zeros(13, 13)
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
            if print_results:
                count_scores(predicted, labels, scores)
            total += len(labels)
            correct += (predicted == labels).sum().item()
    if total:
        acc = 100 * correct / total
    if print_results:
        log.info(f'Val acc: {acc}')
        draw_results()
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


def t(model, songs_list, audio_root, params, save_path):
    param, _, _, _, category = params()
    for song_name, X in gen_test_data(songs_list, audio_root, param):
        y = model.predict(X)
        preds_to_lab(y, param['hop_size'], param['fs'], category, save_path, song_name)


if __name__ == '__main__':
    parser = get_train_rnn_parser()
    args = parser.parse_args(sys.argv[1:])
    log.info('Arguments:\n' + pformat(args.__dict__))
    train(args)
