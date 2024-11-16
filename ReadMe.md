# DDPM_simple_pytorch

pytorch版ddpm的简单实现

参考：[chunyu-li/ddpm: 扩散模型的简易 PyTorch 实现 (github.com)](https://github.com/chunyu-li/ddpm)

## 数据集的简历

1. 由于torchvision.datasets.StanfordCars已无法下载，你可以在[Stanford Cars Dataset (kaggle.com)](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)下载数据集

2. 数据集的文件结构如下：

   ```
   ddpm_simple_pytorch(master)
   ├── README.md
   ├── stanford_cars
   	├──train
   		├── default
   			├──1.png 
   	├──test
   	├── default
   			├──1.png 
   └── ...
   ```

   ## 各文件的展示

   1. 查看数据集

      ```
      python dataloader.py
      ```

   2. 模拟前向扩散

      ```
      python forward_noising.py 
      ```

   3. 展示Unet的结构

      ```
      python unet.py
      ```

   4. 进行模型的训练

      ```
      python training_model.py
      ```

   5. 生成一张图片

      ```
      python sampling.py
      ```