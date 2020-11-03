import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim

from loader import BatchLoader
from model import RelationModel
from utils import scorer, constant, helper
from utils.vocab import Vocab

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)

parser.add_argument('--gcn', dest='gcn', action='store_true', help='Vanilla GCN.')
parser.set_defaults(gcn=False)
parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', type=float, default=0, help='L2-penalty for all pooling output.')
parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
parser.add_argument('--prune_k', default=-1, type=int, help='Prune the dependency tree to <= K distance off the dependency path; set to -1 for no pruning.')

parser.add_argument('--sgcn', dest='sgcn', action='store_true', help='Syn-GCN.')
parser.set_defaults(sgcn=False)
parser.add_argument('--d_attn_dim', type=int, default=50, help='Attention size.')
parser.add_argument('--deprel_dim', type=int, default=100, help='Deprel embedding dimension.')

parser.add_argument('--rgcn', dest='rgcn', action='store_true', help='R-GCN.')
parser.set_defaults(rgcn=False)

parser.add_argument('--gat', dest='gat', action='store_true', help='R-GCN.')
parser.set_defaults(gat=False)

parser.add_argument('--e_attn', dest='e_attn', action='store_true', help='Entity Attention.')
parser.set_defaults(sgcn=False)

parser.add_argument('--ee', dest='ee', action='store_true', help='Encode Entity')
parser.set_defaults(ee=False)

parser.add_argument('--pattn', dest='pattn', action='store_true', help='Use attention layer.')
parser.set_defaults(pattn=False)
parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
parser.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')

parser.add_argument('--trans', dest='trans', action='store_true', help='Transformer')
parser.set_defaults(sgcn=False)


parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# make opt
opt = vars(args)
opt['num_class'] = len(constant.LABEL_TO_ID)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
opt['rule_size'] = len(vocab.id2rule)
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = BatchLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, 'tacred/mappings_train.txt', evaluation=False)
dev_batch = BatchLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, 'tacred/mappings_dev.txt', evaluation=True)
model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_f1")

# print model info
helper.print_config(opt)

# model
model = RelationModel(opt, emb_matrix=emb_matrix)

id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
dev_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    epoch_start_time = time.time()
    for batch in train_batch.data:
        start_time = time.time()
        global_step += 1
        loss = model.update(batch, False)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))
    for batch in train_batch.data_r:
        start_time = time.time()
        global_step += 1
        loss = model.update(batch, True)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    dev_loss = 0
    references = []
    candidates = []
    for batch in dev_batch.data:
        preds, _, _, loss = model.predict(batch, False)
        predictions += preds
        dev_loss += loss
    for batch in dev_batch.data_r:
        preds, _, outputs, loss = model.predict(batch, True)
        predictions += preds
        dev_loss += loss
        batch_size = len(preds)
        rules = batch.rule.view(batch_size, -1)
        for i in range(batch_size):
            output = outputs.transpose(0, 1)[i]
            reference = [[vocab.id2rule[int(r)] for r in rules[i].tolist()[1:] if r not in [0,3]]]
            candidate = []
            for r in output.tolist()[1:]:
                if int(r) == 3:
                    break
                else:
                    candidate.append(vocab.id2rule[int(r)])
            # print (reference)
            # print (candidate)
            references.append(reference)
            candidates.append(candidate)
    predictions = [id2label[p] for p in predictions]
    dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)
    bleu = corpus_bleu(references, candidates)
    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']
    epoch_duration = time.time() - epoch_start_time
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}, bleu = {:.4f}, time = {:.3f} sec".format(epoch,\
            train_loss, dev_loss, dev_f1, bleu, epoch_duration))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1))
    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    model.save(model_file, epoch)
    if epoch == 1 or dev_f1 > max(dev_f1_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)
    # lr schedule
    if len(dev_f1_history) > 5 and dev_f1 <= dev_f1_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad']:
        current_lr *= opt['lr_decay']
        model.update_lr(current_lr)

    dev_f1_history += [dev_f1]
    print("")

print("Training ended with {} epochs.".format(epoch))