# HEDA
利用基于可解释性数据的混合数据增强方案缓解生成式大语言模型的捷径学习(Mitigate Shortcut Learning for LLMs with Hybrid Explainable Data Augmentation)

## 环境依赖
请确保你有正常的CUDA环境，并安装以下依赖:

```
pip install fairseq
pip install fairscale
```

如果要训练模型，请根据 sentencepiece [官方文档](https://github.com/google/sentencepiece) 来处理数据。

```
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
make install
```

## 数据准备部分
- 下载相关数据集
- 随机采样, 并根据特定提示模板处理数据：本文从 PromptSource [官方库](https://github.com/bigscience-workshop/promptsource) 选择多个提示模板。
- 利用ChatGPT基于可控提示模板生成可解释性数据：
```
python gpt.py
```
- 将部分可解释数据和原始数据混合为最终训练数据，将训练数据提示问句部分和标签部分分别放到source和target的文件中。

## 数据处理部分
- 将数据处理为训练所需要的二进制文件：
```
bash scripts/proprecess.sh
```

## 训练模型
- 使用LoRA高效微调LLaMA(7B)模型
```
bash scripts/train.sh
```

## 推理预测
- 使用微调后的模型测试
```
bash scripts/resgen.sh
```

