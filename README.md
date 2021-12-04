# 软件工程 - 小组项目
该项目用于实现渔船作业方式的识别，可以根据csv文件里的渔船作业经纬度、速度、方向以及时间判断渔船的作业方式。
计划将制成一个网页版，能够实现下列功能：
- 能够实时根据用户上传的csv文件，判别渔船的作业方式
- 能够根据用户上传的csv文件，动态显示用户在海域内的作业的位置和运动方向
- 若时间充足，可以基于源数据集，生成全部数据的作业方式，实现海域位置、运动可视化，即可以在地图上看见各片海域的船只作业方式，方便渔民在适合的海域使用适当的作业方式

1、目前已实现模型的建立：正确率如下

![image-20211204235932664](https://gitee.com/wuzhengqian/my-copy-picture/raw/master/img/202112042359985.png)

2、实现了第一阶段的网页功能，能够实时根据用户上传的csv文件，判别渔船的作业方式

![图片1](https://gitee.com/wuzhengqian/my-copy-picture/raw/master/img/202112041549175.jpeg)

![图片2](https://gitee.com/wuzhengqian/my-copy-picture/raw/master/img/202112041549089.jpeg)

![image-20211204153414170](https://gitee.com/wuzhengqian/my-copy-picture/raw/master/img/202112041549262.png)



3、第二阶段可视化部分，实现了用户注册登录后识别可视化的限制访问，作业轨迹可视化尚未实现。

![image-20211204153812057](https://gitee.com/wuzhengqian/my-copy-picture/raw/master/img/202112041549202.png)
