# 模式识别与机器学习 2024 春夏学期 神经网络实践作业

### 作业目标

根据课上所学神经网络的基础理论和数学公式，用Python代码实现简易2层神经网络，尝试解决复杂的分类任务。该作业需要你将以下知识点用于实践：

**Python基础操作**
- Numpy矩阵运算、Matplotlib绘图操作
- 交互式编辑器NoteBook操作（你的训练可视化结果将存储在Jupyter Notebook文件中）

**网络结构知识点**
- 理解神经网络参数（权重以及偏置）的存储形式
- 理解神经网络如何通过矩阵运算实现前向传播
- 理解非线性激活函数的重要意义
- 理解如何计算神经网络的损失函数

**数据处理知识点**
- 合理划分训练和测试集
- 理解训练过程为何要将大量数据划分成mini batch

**训练与验证知识点**
- 将神经网络的向量输出用于多类别分类
- 理解学习率的大小对训练过程的影响
- 学会观察训练损失和验证损失以判断过拟合
- 在测试集上观察模型性能，并分析错误案例

**更高要求**
- 理解损失函数中正则化项（Regularization terms）对网络训练和分类性能的影响
- 理解神经网络隐藏层所学到的特征模板
- 调节超参数（隐藏层神经元数目、激活函数、学习率、正则化）以训练获得更优神经网络参数

### 准备工作
下载安装Anaconda以准备好Python环境
1. 参考[这篇文章](https://zhuanlan.zhihu.com/p/32925500)配置好Anaconda，该步让配置Python环境更加方便，且自带Jupyter Notebook
2. 运行Anaconda Power Shell
3. 参考[这篇文章](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)设置好Python库镜像源。在你刚打开的终端中运行以下两行代码：
```
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
4. 你需要安装的库：
pip install numpy matplotlib imageio

5. 下载VSCode或其他你已经习惯的代码编辑器。若使用VSCode，可参考[这篇文章](https://www.jianshu.com/p/12ab3f05a43f)安装Python插件。同时也会支持Jupyter notebook

6. 选择Python Kernel为Anaconda默认提供的解释器，就可以运行Jupyter里的代码块

### 代码介绍
```main.ipynb``` 是搭建、训练和评估网络的脚本。你训练网络以及可视化结果会随着脚本一起存储，作为作业打分的目标。

```net_utils/neural_net.py``` 是神经网络的功能实现

```net_utils/gradient_check.py``` 是梯度反向传播代码实现

```net_utils/data_utils.py``` 是用于加载分类数据集的代码

```net_utils/vis_utils.py``` 是用于可视化神经网络所学的特征模式的代码

### 作业要求

1. （15分）实现神经网络前馈过程。补全```net_utils/neural_net.py```的TwoLayerNet.loss方法中的计算代码。请在forward pass的代码中实现**ReLU激活函数**以赋予网络非线性分类能力。

2. （10分）实现神经网络训练过程。补全```net_utils/neural_net.py```的TwoLayerNet.train方法中的代码。

3. （10分）实现神经网络分类过程。补全```net_utils/neural_net.py```的TwoLayerNet.predict方法中的代码。

4. （10分）利用```main.ipynb```中Toy example来验证你的网络是否实现正确。

5. （15分）验证通过后，在更难的CIFAR-10数据集上训练并测试你的神经网络。

6. （15分）请绘制```main.ipynb```中Toy example和CIFAR-10两个分类任务的训练过程（可包含训练损失、验证损失、验证准确率等指标的曲线。据此判断是否过拟合，以调节你的训练时长。

7. （25分）调节网络超参数组合，观察不同超参数对网络在CIFAR-10上分类准确率的影响，通过尝试不同的组合，获得你的最佳超参数。你的网络分类准确率至少达到**48%**。最后可视化网络所学的特征，观察和调节超参数之前相比有什么变化。


### 附加题
请学有余力的同学尝试以下任务：

1. （15分）在```net_utils/neural_net.py```的TwoLayerNet.loss方法中实现其他种类的激活函数（如Sigmoid、Tanh等），并编写相应的梯度计算和反向传播代码

2. （15分）尝试添加Dropout、PCA降维或添加更多特征以实现更高的CIFAR-10分类准确率（>=52%）。



