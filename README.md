# flower_classification_vai_tf_numpy_array
This is a simple example about how to train a ConNet model from labeled dataset with TensorFlow and then use Vitis AI 1.2 tools to deploy the model into ZCU102 board. <br /><br />
To make it easier I just make my model ovefit the dataset. All the training/validation/calibration data are just from the same dataset. <br /> 
And I just use the numpy array as data input and OpenCV functions to open images during model training. Please find project using tf.data.Dataset here: [flower_classification_vai_tf_dataset](https://github.com/gewuek/flower_classification_vai_tf_dataset).<br />
The dataset is downloaded from: https://www.kaggle.com/alxmamaev/flowers-recognition <br />
And you may find the dataset from Keras tutorial: https://www.kaggle.com/alxmamaev/flowers-recognition <br />
The ARM deploy code is modified from the DNNDK resnet50 example code. <br />

The whole design is trained and deployed using Ubuntu 18.04 + Vitis AI 1.2 + TensorFlow 1.15 + PetaLinux 2020.1. <br />
<br />

***To be notices:*** Although this design uses Xilinx tools to deploy design on Xilinx developboard this is just personal release. No gurantee can be made here. :-) Pease feel free to contact me or post your questions on:  <br />
https://forums.xilinx.com/t5/Machine-Learning/bd-p/Deephi <br />


### TensorFlow Training and Vitis AI Quantization Flow<br />
Please install the Vitis AI 1.2 according to https://github.com/Xilinx/Vitis-AI/ before starting the custom model flow.<br />
Make sure you can run Vitis AI DNNDK examples.<br />

1. git clone the repository inside the Vitis-AI folder so that when launching the docker you can see the ***flower_classification_vai_tf_numpy_array*** folder inside the docker workspace.<br />
2. Download kaggle flower dataset from https://www.kaggle.com/alxmamaev/flowers-recognition <br />
3. unzip the folder and copy the files into ```flower_classification_vai_tf_numpy_array/x86/flowers``` folder. So that the directory would like below: <br />
![directory.PNG](/pic_for_readme/directory.PNG) <br />
4. Launch the docker, call ```conda activate vitis-ai-tensorflow``` to set the TensorFlow environment  and then navigate into the ```flower_classification_vai_tf_numpy_array/x86/``` folder <br />
5. Load images and labels into dataset <br />
```python3 ./load_data.py``` <br />
6. Train data <br />
```python3 ./train_data.py``` <br />
7. Evaluate the trained model(Opitional) <br />
```python3 ./evaluate_trained_model.py``` <br />
8. Freeze the model <br />
```python3 ./freeze_model.py``` <br />
9. Evaluate the frozen model(Opitional) <br />
```python3 ./evaluate_frozen_model.py``` <br />
10. Quantize the graph, using ```chmod u+x ./decent_q.sh``` if necessary <br />
```./decent_q.sh``` <br />
11. Evaluate quantized graph (Optional) <br />
```python3 ./evaluate_quantized_graph.py``` <br />
12. Compile the quantized model into elf using DNNC, use ```chmod u+x ./dnnc.sh``` if necessary <br />
```./dnnc.sh``` <br />
13. Now you should get the ELF file at ```flower_classification_vai_tf_numpy_array/x86/flower_classification/dpu_flower_classification_0.elf```. Copy the file into the ```flower_classification_vai_tf_numpy_array/arm/flower_classification/model``` for further usage <br />

### Test on ZCU102 board
For build and deploy the example on ZCU102 board, there are two additional requirement <br />
1. DNNDK Libs <br />
  a) Refer to [DNNDK example](https://github.com/Xilinx/Vitis-AI/tree/v1.2/mpsoc) to set up the on board DNNDK libs. <br />
  b) Refer to [Vitis Ai Library guide](https://github.com/Xilinx/Vitis-AI/tree/v1.2/Vitis-AI-Library) to set up the on board VAi libs. <br />
  c) I was told that the Vitis AI Libs are optional if you are using pure DNNDK libs, but don't have time to have a try.<br /><br />

2. Compile Environment <br />
  a) If you are using [ZCU102 VAI 1.2 release image](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-zcu102-dpu-v2020.1-v1.2.0.img.gz) the compiling environment is preinstalled at the image. So you just need to copy the ***flower_classification_vai_tf_numpy_array/arm/flower_classification*** folder into the ZCU102 board(SD card or DDR) and when the board boot up go to the flower_classification folder and run ***make*** to compile the application. <br /><br />
  b) For custom platform please refer to the configuration from [Vitis AI custom platform creation flow](https://github.com/gewuek/vitis_ai_custom_platform_flow). If it is not up-to-date please try with this [VAI 1.2 tmp rep](https://github.com/gewuek/vitis_ai_custom_platform_v1.2_tmp) to configure the rootfs. Both the Vitis AI GUI compilation and on board compilation should work. <br />

The running result on ZCU102 would like below:
![classification_flower.PNG](/pic_for_readme/classification_flower.PNG)

### Reference

https://www.youtube.com/watch?v=VwVg9jCtqaU&t=112s

https://www.kaggle.com/alxmamaev/flowers-recognition

https://www.youtube.com/watch?v=j-3vuBynnOE

https://github.com/tensorflow/docs/blob/r1.12/site/en/tutorials/load_data/images.ipynb

https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf

https://github.com/Xilinx/Edge-AI-Platform-Tutorials/tree/3.1/docs/DPU-Integration

https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb

https://stackoverflow.com/questions/51278213/what-is-the-use-of-a-pb-file-in-tensorflow-and-how-does-it-work/51281809#51281809
