"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from loader import BatchLoader
from model import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
model = RelationModel(opt)
model.load(model_file)

# load vocab
vocab_file = 'vocab/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
eval_batch = BatchLoader(data_file, opt['batch_size'], opt, vocab, 'tacred/mappings_dev.txt', evaluation=True)

helper.print_config(opt)
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

predictions = []
all_probs = []
references = []
candidates = []
# for batch in eval_batch.data:
#     preds, probs, _ = model.predict(batch, False)
#     predictions += preds
#     all_probs += probs
for batch in eval_batch.data_r:
    preds, probs, outputs, loss = model.predict(batch, True)
    predictions += preds
    all_probs += probs
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
        print (reference[0])
        print (candidate)
        print ()
        references.append(reference)
        candidates.append(candidate)
predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(eval_batch.gold(), predictions, verbose=True)
bleu = corpus_bleu(references, candidates)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}\t{:.4f}".format(args.dataset,p,r,f1,bleu))

print("Evaluation ended.")

