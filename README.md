# pbnn-ModelZoo
## 1 编译环境
下载docker。包括模型量化、编译、芯片运行所需依赖。https://github.com/bjxx-algo/pbnn-toolkit/
进入docker，在/app/ 目录下执行以下命令
```mkdir -p build
cmake ..
make -j 64
cd .. 
```

## 2 运行（以DA04核心板1为例）
slaver端启动`./pb_infer_coserver `服务
```
cd pbnn-ModelZoo/pb_infer&& ./pb_infer_coserver
```
master启动`./pb_infer_server`服务
```
cd pbnn-ModelZoo/pb_infer&& ./pb_infer_server
```
运行代码
```
root@ubuntu2004-arm64:/data/fwj# ./build/yolov8_demo  /data/fwj/model/yolov8s.pbnn
Running preprocess...
Preprocess OK.
Running execute...
execute OK.
Running postprocess...
conf 0.755859 cls 18 label ok 0.75 box x1 483.97 box y1 137.667 box x2 613.099 box y2 284.939
✅ output saved: ./results/yolov8s_output_20251224_120632_264.jpg
Postprocess OK.
```
注*：/data/fwj/model/yolov8s.pbnn模型路径为绝对路径

```
转换模型和推理脚本 ./build/yolov8_demo 运行成功后，推理的图片结果保存在 ./results/yolov8s_output_20251224_120632_264.jpg
```
