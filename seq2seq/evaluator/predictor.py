import torch
from torch.autograd import Variable
import random

class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab,use_cuda=True,sample_bais = 1,sample_lossrate = float('Inf')):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        self.use_cuda = use_cuda
        self.sample_bais = sample_bais
        self.sample_lossrate =sample_lossrate
        if torch.cuda.is_available() and use_cuda:
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab


    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
                              volatile=True).view(1, -1)
        if torch.cuda.is_available() and self.use_cuda:
            src_id_seq = src_id_seq.cuda()

        softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])
        if self.sample_bais != 1 and self.sample_bais > len(other['topk_length'][0]):
            self.sample_bais = len(other['topk_length'][0])
        id = random.randint(0,self.sample_bais-1)
        while other['score'][0][0] * self.sample_lossrate > other['score'][0][id]:
            id = random.randint(0,self.sample_bais-1)
        length = other['topk_length'][0][id]
        #length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][id].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq
