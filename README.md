## 说明
代码链接：*https://github.com/zhanghlgithub/pytorch-backbone-cifar10*

为了方便快速学习和测试不同网络结构的性能表现，本模块封装好了接口。公开数据集基于cifar10数据测试。方便做实验和学习最新的方法。
##### 使用方法
在使用时，只需把自己学到的最新网络结构添加到backbone模块即可，然后修改训练脚本`train.sh`中的backbone参数，使得模型训练时调用你写的backbone。简单好用！
## 训练
##### 训练脚本
```python
bash train.sh
```
##### 关键参数说明

 - lr：初试学习率
 - backbones：选择不同的backbones
 - st_epoch：开始epoch
 - end_epoch：结束epoch
 - optim：选择优化器
 注：具体参数含义，查看train.parse_args()参数说明
 ##### 训练过程
 **adam算法：** 
 
| lr             | epoch          |
|:--------:      | :-------------:|
| 0.001 或 0.003 | 0 ~ 50 epoch   |
| 0.0001         |  50~ 80 epoch  |
| 0.00001        | 80~100 epoch   |

 **SGD**

| lr       | epoch           |
|:--------:| :-------------: |
| 0.1      | 0 ~ 200 epoch   |
| 0.01     | 200 ~ 300 epoch |
| 0.001    | 300 ~ 400 epoch |
注：**adam算法收敛快，但是最后精度不如SGD， 训练过程查看log日志**
## DONE
| model | 精度     |   
|:--------:| :-------------:|
| [mobileNetV2](https://arxiv.org/abs/1801.04381) | 93.13%  |
| [DPN26](https://arxiv.org/abs/1707.01629) | 94.91%  |
预训练模型下载：链接：（未上传）
## TODO
 - [ ] [MobileNet-V1](https://arxiv.org/abs/1704.04861)
 - [ ] [MobileNet-V3](https://arxiv.org/abs/1905.02244)
 - [ ] [ShuffleNet-V1](https://arxiv.org/abs/1707.01083)
 - [ ] [ShuffleNet-V2](https://arxiv.org/abs/1807.11164)
 - [ ] [DenseNet](https://arxiv.org/pdf/1608.06993.pdf )
 - [ ] [EfficientNetB0-B7](https://arxiv.org/abs/1905.11946)
 - [ ] [MixNet](https://arxiv.org/abs/1907.09595)
 - [ ] [MNASNet B1, A1(Squeeze-Excite)](https://arxiv.org/abs/1807.11626)
 - [ ] [ChamNet](https://arxiv.org/abs/1812.08934)
 - [ ] [FBNet-C](https://arxiv.org/abs/1812.03443)
 - [ ] [Sigle-path NAS](https://arxiv.org/abs/1904.02877)
 - [ ] [ResNet](https://arxiv.org/abs/1512.03385)
 - [ ] [ResNext](https://arxiv.org/abs/1611.05431)
注：以上网络都是非常先进的，常用的网络结构，由于硬件环境+时间原因。不能全部train出模型。有时间会继续完善。

