## English Chinese Translate 2018-12

#### 1.preprocess

clean() 删去无用字符并分词，prepare() 将数据保存为 (en_text, zh_text) 格式

#### 2.represent

add_flag() 添加控制符，shift() 对 zh_text 分别删去 bos、eos 得到 zh_sent、label

tokenize() 分别通过 en_sent、flag_zh_text 建立词索引，构造 embed_mat

align() 对训练数据 en_sent 头部，zh_sent、label 尾部，填充或截取为定长序列

#### 3.build

通过 rnn 的 s2s、att 构建翻译模型，分别对解码器词特征 h2、编码器词特征 h1

线性映射得到 q、k、v，使用点积注意力得到语境向量 c，h2_i 与 c_i 共同决定输出

#### 4.translate

先对输入进行编码、再通过搜索进行解码，check() 忽略无效词，plot_att() 可视化