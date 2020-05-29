# 基于多模态的属性抽取

完整文档：https://zhuanlan.zhihu.com/p/144268850

* 论文：Multimodal Attribute Extraction. Logan, I. V., et al. NIPS 2017. https://arxiv.org/pdf/1711.11118.pdf
* 数据集：MAE-dataset https://rloganiv.github.io/mae/
* 环境：tensorflow 1.12

最近做属性抽取，并且基于多模态信息（文本+图片）那种，然后发现了一个比较经典的论文“Multimodal Attribute Extraction”。正好就顺着这个论文的思路，捋一下这个任务，复现一下，再记录一下实践过程和想法

论文在几年前，提出了这么一个做多模态任务的框架，在今天来看感觉还挺经典的，但是具体方法在今天来看就非常普通，没什么高端操作，比不起现在的论文里各种眼花缭乱一通乱秀。但是出发总要从原点开始，中间有些想法和总结，有点什么就记点什么

