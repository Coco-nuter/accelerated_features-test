1. 按照quant文件夹内的说明完成量化和编译，获得.xmoel
2. 将.xmoel拷贝到model_data目录下
3. 准备测试图片，将其拷贝纸data目录
4. 将整个xilinx文件夹通过U盘活ssh拷贝至板子上
5. 进入开发板，进入xilinx目录
6. 执行：python megadepth1500.py