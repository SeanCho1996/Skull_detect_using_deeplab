# 基于deeplab v3+的超声图像头围测量

## Deeplab v3+的结构和实现
### Deeplab v3+的详细分析
请参阅[这篇博文](https://zhangbin0917.github.io/2018/06/03/Encoder-Decoder-with-Atrous-Separable-Convolution-for-Semantic-Image-Segmentation/)
### Deeplab v3+的简单理解
Deeplab v3+相较于v3，主要的提升在于引入了类似U-Net的Encoder-Decoder结构，一方面利用ASPP提取金字塔多尺度的上下文特征，而Encoder-Decoder结构可以强化边缘的细节。</br>
**Encoder**的前向卷积结构直接引用了Xception模型，经过4个大的卷积层（包含pooling），最后输出的feature map的尺寸比原图缩小了16倍（2^4)，最后将feature map做分别4个空洞卷积（空洞率分别为1，6，12，18），再做一次全局的均值池化，将这5个结果concatenate到一起，最后经过一个1x1的卷积调整channel数</br>
**Decoder**的结构相对简单，先将Encoder得到的feature map做双线性插值上采样到4倍尺寸（尺寸为原图的1/4），与Encoder前向网络第二层卷积层的输出（尺寸为原图的1/4）concatenate，再做4倍上采样，过3x3卷积提取最终特征，对每个像素做分类得到结果

## 数据库的建立
* 1.解压后的生数据（训练+验证）储存在一个[整体的文件夹](https://drive.google.com/open?id=1rWuVtATwO_Oe8vPIXIbwIkIeaB84zZr3)中，运行`mk_train_data.py`可以自动把原图和标签图分开储存在`training_set`和`training_label`文件夹中，并生成标签的mask图像，最后生成训练及验证文件的索引txt，储存在`Segmentation`文件夹中。数据库中共有999张图片，我选择了899张做训练，其余的100张做验证。
* 2.运行`remove_gt_color.py`文件将mask标签从24位转移到8位，并生成一个粗边框（这一步是对原deeplab原来的数据集pascal voc 2012做的，我们的数据集本身就是8位而且不需要粗边框），生成的文件保存在`SegmentationClassRaw`文件夹中。
* 3.已有`training_set`, `Segmentation`以及`SegmentationClassRaw`后，就可以运行`build_skull_data.py`将图像转译成tfrecord文件了，生成的文件在`tfrecord`文件夹中。

## 图像前处理
在训练以前，deeplab允许先对图像做一个简单的预处理，即crop切割操作，原因是对于部分设备，若输入图像尺寸过大，显卡计算资源不足会导致训练无法开始，crop的意义在于从图像中随机切取一块指定大小的区域并只针对这一小块做训练，对于pascal voc 2012数据集，其中的图片本身尺寸都小于512x512，所以用默认的切割大小（513x513）就可以包含整张图片且不会超出显卡的内存上限。
但对于我们的数据集，大多数图像尺寸为（800x540），这个大小是超过我的显卡的运算能力的，所以仿照pascal voc 2012数据集，我们做了一个513x513的crop操作。（这个操作在`train.py`里实现）

## 训练
运行`train.py`即可开始训练过程，但是训练需要大量的输入参数，虽然已经将默认的数值调整为适应我们的数据的数值，但仍建议运行`x-train.bat`自动开始训练（只针对windows系统），训练后的checkpoint模型保存在`model`文件夹里。

## 验证
运行`eval.py`即可开始验证，但同样地，推荐运行`x-eval.bat`自动运行，验证这一步有一点***需要特别注意***：在deeplab项目中，[通常采用全尺寸的图片做inference](https://github.com/tensorflow/models/issues/3939)，这意味着，在训练时采用的513x513的crop在验证这一步是不可行的，验证需要的crop尺寸必须要大于等于原图的全尺寸（我选择了801x801）。验证后会在cmd界面显示各类准确率miou。

## 可视化
`vis.py`(`x-vis.bat`)允许将验证结果的图片生成并保存在`.\vis_res\segmentation_results`中，如下图所示</br>
<img src="https://i.ibb.co/SBtx15S/Snipaste-2020-03-25-07-39-58.jpg" alt="Snipaste-2020-03-25-07-39-58" border="0"></br>
