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
但对于我们的数据集，大多数图像尺寸为800x540，这个大小是超过我的显卡的运算能力的，所以仿照pascal voc 2012数据集，我们最初做了一个513x513的crop操作。（这个操作在`train.py`里实现），但是发现这样最后输出的精度很低，查阅代码后发现并不是切割了512x512的部分，而是将原图调整大小，resize到了512x512，所以会严重影响精度，所以最后还是选择了801x801的大小。

## 训练
运行`train.py`即可开始训练过程，但是训练需要大量的输入参数，虽然已经将默认的数值调整为适应我们的数据的数值，但仍建议运行`x-train.bat`自动开始训练（只针对windows系统），训练后的checkpoint模型保存在`model`文件夹里。

## 验证
运行`eval.py`即可开始验证，但同样地，推荐运行`x-eval.bat`自动运行，验证这一步有一点***需要特别注意***：在deeplab项目中，[通常采用全尺寸的图片做inference](https://github.com/tensorflow/models/issues/3939)，这意味着，在训练时采用的513x513的crop在验证这一步是不可行的，验证需要的crop尺寸必须要大于等于原图的全尺寸（我选择了801x801）。验证后会在cmd界面显示各类准确率miou。

## 可视化
`vis.py`(`x-vis.bat`)允许将验证结果的图片生成并保存在`.\vis_res\segmentation_results`中，如下图所示</br>
<img src="https://i.ibb.co/SBtx15S/Snipaste-2020-03-25-07-39-58.jpg" alt="Snipaste-2020-03-25-07-39-58" border="0"></br>

## 提取模型并测试
运行`export_model.py`（`x-export_model.bat`）即可导出训练好的[模型](https://drive.google.com/open?id=1aoTMrdQW2ogQ9hxaU0F1O9Fr0AlT4TWl)，储存在`Xout`文件夹中，运行`post_process.py`（`x-post_process.bat`）允许读取这个模型并在一张测试图片上运行，得到的结果如下图所示</br>
<img src="https://i.ibb.co/zN9LXdG/Snipaste-2020-03-28-23-51-24.jpg" alt="Snipaste-2020-03-28-23-51-24" border="0"></br>

`post_process.py`中引用了一些来自`x-final.py`文件中的操作，事实上`x-final.py`中的函数绝大多数是基于模型的运算，比如利用模型求出一张图片的预测(run_image函数），由标签图片生成合适显示的colormap（label_to_color_image函数）等，而`post_process.py`中的函数则是对生成图像的后处理，包括求形状轮廓，求mask的近似椭圆，求近似周长以及可视化等。

## 结果分析
仅从分类性能来看，deeplab v3+在验证集的准确率还是非常高的，对于当前这个二分类的分割问题，总体的准确率可以达到95%，而且这只是在训练集上训练了1000个epoch的粗结果，如果增加训练数，理论上应该可以获得更好的结果。</br>
<img src="https://i.ibb.co/XywDYP8/Snipaste-2020-03-27-16-11-51.jpg" alt="Snipaste-2020-03-27-16-11-51" border="0"></br>
但是在测试更多图片后发现，对于目标边缘的噪声处理的仍然不好，如下图所示</br>
<img src="https://i.ibb.co/SRZshLs/Figure-2.png" alt="Figure-2" border="0"></br>
而且整体的运算速度也并不理想，对于简单的图片处理时间在30s左右，而相对比较复杂的图片则需要50s-60s

## 文件结构
<img src="https://i.ibb.co/nfgfGkp/Snipaste-2020-03-29-00-18-07.jpg" alt="Snipaste-2020-03-29-00-18-07" border="0">
