"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from utils import constant, torch_utils

# from torch.autograd import Variable

from torch_geometric.nn import GCNConv

class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.classifier = SynGCN(opt, emb_matrix)
        self.decoder = Decoder(opt)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_d = nn.NLLLoss(ignore_index=constant.PAD_ID)
        self.parameters = [p for p in self.classifier.parameters() if p.requires_grad]
        if opt['cuda']:
            self.classifier.cuda()
            self.decoder.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
    
    def update(self, batch):
        """ Run a step of forward and backward model update. """
        inputs = list()
        if self.opt['cuda']:
            for b in constant.KEYS:
                inputs += [batch[b].cuda()]
            labels = batch.rel.cuda()
            rules  = batch.rule.cuda()
        else:
            for b in constant.KEYS:
                inputs += [batch[b]]
            labels = batch.rel
            rules  = batch.rule
        batch_size = labels.size(0)
        # step forward
        self.classifier.train()
        self.decoder.train()
        self.optimizer.zero_grad()


        logits, hidden, encoder_outputs = self.classifier(inputs, batch_size)
        loss = self.criterion(logits, labels)
        
        #DECODER PART
        rules = rules.view(batch_size, -1)
        masks = inputs[1]
        max_len = rules.size(1)
        rules = rules.transpose(1,0)
        output = rules.data[0, :] # sos
        # outputs = torch.zeros(max_len, batch_size, self.opt['rule_size'])
        # if self.opt['cuda']:
        #         outputs = outputs.cuda()
        loss_d = 0
        h0 = hidden[0].view(self.opt['num_layers'], 2, batch_size, -1).transpose(1, 2).sum(2)
        c0 = hidden[1].view(self.opt['num_layers'], 2, batch_size, -1).transpose(1, 2).sum(2)
        decoder_hidden = (h0, c0)
        for t in range(1, max_len):
            output, decoder_hidden, attn_weights = self.decoder(
                    output, masks, decoder_hidden, encoder_outputs)
            loss_d += self.criterion_d(output, rules[t])
            # outputs[t] = output
            # top1 = output.data.max(1)[1]
            output = rules.data[t]
            if self.opt['cuda']:
                output = output.cuda()
        loss += loss_d
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.opt['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, batch):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        inputs = list()
        if self.opt['cuda']:
            for b in constant.KEYS:
                inputs += [batch[b].cuda()]
            labels = batch.rel.cuda()
        else:
            for b in constant.KEYS:
                inputs += [batch[b]]
            labels = batch.rel

        batch_size = labels.size(0)

        # forward
        self.classifier.eval()
        logits, hidden, encoder_outputs = self.classifier(inputs, batch_size)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()

        return predictions, probs, loss.data.item()

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

class SynGCN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(SynGCN, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        
        input_size = opt['emb_dim']
        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,\
                dropout=opt['dropout'], bidirectional=True)

        if opt['gcn']:
            self.deprel_emb = nn.Embedding(len(constant.DEPREL_TO_ID), opt['deprel_dim'],
                    padding_idx=constant.PAD_ID)
            self.attn = Attention(opt['deprel_dim'], 2*opt['hidden_dim'], opt['attn_dim'])
            self.gcn = GCNConv(2*opt['hidden_dim'], opt['hidden_dim'])

            self.entity_attn = Attention(opt['hidden_dim'], opt['hidden_dim'], opt['hidden_dim'])

        self.linear = nn.Linear(2*opt['hidden_dim'], opt['num_class'])

        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()
    
    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        if self.opt['gcn'] and self.opt['deprel_dim'] > 0:
            self.deprel_emb.weight.data[1:,:].uniform_(-1.0, 1.0)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer

        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")

    def zero_state(self, batch_size): 
        state_shape = (2*self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0
    
    def forward(self, inputs, batch_size):
        for i in range(len(inputs)):
            if i != 6:
                inputs[i] = inputs[i].view(batch_size, -1)
        words, masks, deprel, d_masks, subj_mask, obj_mask, edge_index = inputs # unpack
        s_len = words.size(1)
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        # embedding lookup
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        inputs = self.drop(torch.cat(inputs, dim=2)) # add dropout to input
        input_size = inputs.size(2)
        
        # rnn
        h0, c0 = self.zero_state(batch_size)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # hidden = self.drop(ht[-1,:,:]) # get the outmost layer h_n
        outputs = self.drop(outputs)

        if self.opt['gcn']:
            deprel = self.deprel_emb(deprel)
            weights = self.attn(deprel, d_masks, outputs[:,0,:]).view(-1)
            weights = weights[weights.nonzero()].squeeze(1)
            outputs = outputs.reshape(s_len*batch_size, -1)
            outputs = self.gcn(outputs, edge_index)#, weights)
            outputs = outputs.reshape(batch_size, s_len, -1)

            subj_weights = self.entity_attn(outputs, subj_mask, outputs[:,0,:])
            obj_weights  = self.entity_attn(outputs, obj_mask, outputs[:,0,:])

            subj = subj_weights.unsqueeze(1).bmm(outputs).squeeze(1)
            obj  = obj_weights.unsqueeze(1).bmm(outputs).squeeze(1)

            final_hidden = self.drop(torch.cat([subj, obj] , dim=1))
        
        else:
            final_hidden = outputs[:,0,:]

        logits = self.linear(final_hidden)
        return logits, (ht, ct), outputs

class Attention(nn.Module):
    """
    A GCN layer with attention on deprel as edge weights.
    """
    
    def __init__(self, input_size, query_size, attn_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        self.vlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_() # use zero to give uniform attention at the beginning
    
    def forward(self, x, x_mask, q):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size)
        q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(
            batch_size, self.attn_size).unsqueeze(1).expand(
                batch_size, seq_len, self.attn_size)
        projs = [x_proj, q_proj]
        scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len)

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores, dim=1)

        return weights
    
class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.embed_size = opt['emb_dim']
        self.hidden_size = opt['hidden_dim']
        self.output_size = opt['rule_size']
        self.n_layers = opt['num_layers']

        self.embed = nn.Embedding(self.output_size, self.embed_size)
        self.dropout = nn.Dropout(opt['dropout'], inplace=True)
        self.attention = Attention(self.hidden_size, opt['hidden_dim'], opt['attn_dim'])
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size,
                          self.n_layers, dropout=opt['dropout'])
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, masks, last_hidden, encoder_outputs):

        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)

        # batch_size = encoder_outputs.size(0)
        # # Calculate attention weights and apply to encoder outputs
        # attn_weights = self.attention(encoder_outputs, masks, last_hidden[0].view(2, batch_size,-1)[-1])
        # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        # context = context.transpose(0, 1)  # (1,B,N)
        # # Combine embedded input word and attended context, run through RNN
        rnn_input = embedded #torch.cat([embedded, context], 2)
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        # context = context.squeeze(0)
        output = self.out(output) #torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, None #, attn_weights


