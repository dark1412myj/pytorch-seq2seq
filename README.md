# seq2seq模型改进，增加多样性
+ 根据论文[A Simple, Fast Diverse Decoding Algorithm for Neural Generation在seq2seq](https://arxiv.org/pdf/1611.08562.pdf),在beam search基础上进行修改



+ 对比原beam search和修改后的beam search，在beam search算法中增加惩罚因子![]( https://github.com/dark1412myj/IMageBase/blob/master/seq2seq_3.jpg )
+ γ 为 多样化率，k 为当前节点在父节点中的排名

![]( https://github.com/dark1412myj/IMageBase/blob/master/seq2seq_1.jpg "对于原beam search和改进后的")
