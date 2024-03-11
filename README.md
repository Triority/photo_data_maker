## 文件和程序说明
+ `backs`文件夹是我随机找到的图片，可作为随机背景
+ `images`和`images_marked`为在18届智能车讯飞组比赛中需要的数据和绿色椭圆形标注
+ `video2img.py`程序可以提取视频帧
+ `photos2video.py`是用于延时摄影照片合成视频的程序(别问我为什么放在这个仓库)
+ `rename_all.py`用于给图片批量重命名，一些奇怪符号好似导致yolov4-tiny在建立索引出错
+ `pytorch_model_resave.py`用于转换模型到旧版本pytorch的格式
+ `auto_marking.py`用于生成图片数据集，只解释一下这个程序

## 生成原理
我的需求是使用yolo识别几十个带有不同图案的白色背景的矩形泡沫板(用于智能车竞赛讯飞组的识别任务)。
实在是懒得手动拉框标注数据集，所以写了这个一劳永逸的程序。已经在比赛里用了两届了，非常好用，板子只要出现在摄像头范围内就是0.99或者1.00的置信度识别到。

因为是矩形的泡沫板，我们首先正对板子拍一张图，裁剪掉背景只留板子，然后对这个图像叠加到随机背景上并做各种随机变换，包括亮度颜色位置大小等等等，得到一堆图片。
在标注时，如果需要标注的内容是整个矩形板子，那么只要用绿色矩形覆盖整个图片就好；如果是图片的一部分，我一般使用绿色椭圆形标注其中的某一内容。
只要把标注的图片和原图做一样的位置变换，就可以得到带有纯绿色标注的数据集图片，识别纯绿色就很容易了不说了。

这样就做到了标注几十张图片得到几十万张图片数据集的目的。这些图片的差异程度可以通过配置文件的参数配置，实现不同拍摄环境下的识别，包括场地灯光强度颜色等等等

## 生成步骤
### total
首先定义`total`，这决定了生成数据集的总数，你可以设置为`1`来测试效果。

### processes
使用的进程数，如果不想让电脑在数据生成时候过于卡顿，设置成`multiprocessing.cpu_count() - 1`就好。当然如果想更快或者更慢你可以根据自己的硬件自定义(不要为0就好)

### dir_path
定义4个图片读取和输出的路径

#### backs_dir_path
这个文件夹中应该包含一堆图片，你所使用的随机背景图片，尺寸不要太小，这个尺寸就是输出图像的尺寸

#### images_dir_path
这个文件夹中应该包含一堆文件夹，每个文件夹名称都是一个物品类别，并在其中保存这一类物品的图片

#### marked_dir_path
同样的，这个文件夹中应该包含一堆文件夹，每个文件夹名称都是一个物品类别，并在其中保存这一类物品标注后的图片，同时名称与`images_dir_path`中的对应图片名称相同。
标注方法是在目标物体上绘制一个RGB颜色为(0,255,0)的绿色椭圆。当然不一定是椭圆，对目标定位时取这个绿色区域的最小外接矩形。

#### output_dir_path
输出路径。该路径下会自动创建三个文件夹，分别保存生成的图片数据，使用绿色方框标注后的图片数据，以及标注文件。

### run
配置好以上内容后就可以开始生成数据了

## 参数配置
如果你对默认变换参数不满意或者有特殊需求可以修改变换参数，参数的作用在`funcs.py`的函数注释中已经做了详细解释。
如果需要修改可以直接修改`config.yaml`配置文件，里面也做了注释
