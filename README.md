# seq2seq模型改进，增加多样性
+ 根据论文[A Simple, Fast Diverse Decoding Algorithm for Neural Generation在seq2seq](https://arxiv.org/pdf/1611.08562.pdf),在beam search基础上进行修改
+ 更改输出方式，可以通过设置偏差方式，来随机输出最好结果附近的结果

## 使用方式：
    topk_decoder = (decoder_rnn, k ,use_cuda=True,use_diverse = False,diverse_rate = 1.0)
    seq2seq = Seq2seq(encoder, topk_decoder)
    predictor = Predictor(seq2seq, input_vocab, output_vocab,use_cuda=True,bias = 1, max_diff_rate = float('inf') )
### use_cuda：是否使用gpu，use_diverse：多样化率γ，bias：再最好的bias个结果中随机输出一个，max_diff_rate：允许最差结果与最好结果相差多少倍
### 可以参照sample/sample.py

+ 对比原beam search和修改后的beam search，在beam search算法中增加惩罚因子![]( https://github.com/dark1412myj/IMageBase/blob/master/seq2seq_3.jpg )
+ γ 为 多样化率，k 为当前节点在父节点中的排名

![]( https://github.com/dark1412myj/IMageBase/blob/master/seq2seq_1.jpg "对于原beam search和改进后的")
