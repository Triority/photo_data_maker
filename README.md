## 程序说明
仓库包含两个程序，一个是可以提取视频帧的`video2img.py`程序，这个不多解释了。

另一个程序`auto_marking.py`用于生成图片数据集。
## 生成步骤
### total
首先定义`total`，这决定了生成数据集的总数，你可以设置为`1`来测试效果。

> 注意
> 这个变量是每张图片在每个cpu核心里生成的总数。如果你有100张图片，使用了15个cpu核心，`total`设置为`10`，那么将生成100*15*10=15000张图片。

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
