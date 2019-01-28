## English Chinese Translate 2019-1

#### 1.preprocess

clean() 删去无用字符并分词，prepare() 将数据保存为 (en_text, zh_text) 格式

#### 2.represent

add_flag() 添加控制符，shift() 对 zh_text 分别删去 bos、eos 得到 zh_sent、label

tokenize() 分别通过 en_sent、flag_zh_text 建立词索引，构造 embed_mat

align() 对非测试数据 en_sent、zh_sent、label 头部，填充或截取为定长序列

#### 3.build

通过 dnn 的 trm 构建翻译模型，分别对解码器词特征 x、编码器词特征 y 多头

线性映射得到 q、k、v，使用点积注意力分别得到语境向量、再进行交互决定输出

#### 4.translate

先对输入进行编码、再通过搜索进行解码，check() 忽略无效词，plot_att() 可视化