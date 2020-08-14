Updating to Vitis AI 1.2

# flower_classification_dnndk_v1
This is a simple example about how to train a ConNet model from labeled dataset with TensorFlow and then use DNNDK tools to deploy the model into ZCU102 board. <br /><br />
To make it easier I just make my model ovefit the dataset. All the training/validation/calibration data are just from the same dataset. <br /> 
And I just use the numpy array as data input and OpenCV functions to open images during model training. Please find project using tf.data.Dataset here: [flower_classification_dnndk_v2](https://github.com/gewuek/flower_classification_dnndk_v2).<br />
The dataset is downloaded from: https://www.kaggle.com/alxmamaev/flowers-recognition <br />
And you may find the dataset from Keras tutorial: https://www.kaggle.com/alxmamaev/flowers-recognition <br />
The ARM deploy code is modified from the DNNDK resnet50 example code. <br />

The whole design is trained and deployed using Ubuntu 18.04 + Vitis AI 1.2 + TensorFlow 1.15 + PetaLinux 2020.1. <br />
<br />

***To be notices:*** Although this design uses Xilinx tools to deploy design on Xilinx developboard this is just personal release. No gurantee can be made here. :-) Pease feel free to contact me or post your questions on:  <br />
https://forums.xilinx.com/t5/Machine-Learning/bd-p/Deephi <br />


### TensorFlow Training and DNNDK Quantization Flow<br />
Please install the Vitis AI 1.2 according to https://github.com/Xilinx/Vitis-AI/ before starting the custom model flow.<br />
Make sure you can run Vitis AI DNNDK examples.<br />

1. git clone the repository inside the Vitis-AI folder so that when launching the docker you can see the ***flower_classification_dnndk_v1*** folder inside the docker workspace.<br />
2. Download kaggle flower dataset from https://www.kaggle.com/alxmamaev/flowers-recognition <br />
3. unzip the folder and copy the files into ```flower_classification_dnndk_v1/x86/flowers``` folder. So that the directory would like below: <br />
![directory.PNG](/pic_for_readme/directory.PNG) <br />
4. Launch the docker, call ```conda activate vitis-ai-tensorflow``` to set the TensorFlow environment  and navigate into the ```flower_classification_dnndk_v1/x86/``` folder <br />
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
13. Now you should get the ELF file at ```flower_classification_v1/x86/flower_classification/dpu_flower_classification_0.elf```. Copy the file into the ```flower_classification_v1/arm/flower_classification/model``` for further usage <br />

### Test on ZCU102 board
For build and deploy the example on ZCU102 board, there are two additional requirement <br />
1. DNNDK Libs <br />
a) If you are using the [DPU Intergration flow](https://github.com/Xilinx/Edge-AI-Platform-Tutorials/tree/master/docs/DPU-Integration) after you enable the dnndk user package in PetaLinux the libs would be automatically installed into the rootfs. <br /><br />
b) Otherwise you can try to install the libs according to [DNNDK user guide](https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf) page.21 ~ page.22 <br />

2. Compile Environment <br />
a) If you are using [prebuild DPU TRD image](https://www.xilinx.com/member/forms/download/zcu102-image-license-xef.html?filename=petalinux-user-image-zcu102-zynqmp-sd-20190802.img.gz) or if you enable the compile tool chain in PetaLinux package: ```petalinux package groups -> packagegroup-petalinux-self-hosted```the compiling environment is preinstalled at the image. So you just need to copy the ```flower_classification_v1/arm/flower_classification``` folder into the ZCU102 board(SD card or DDR) and when the board boot up go to the flower_classification folder and run ***make*** to compile the application. <br /><br />
b) If you work with [DPU Intergration flow](https://github.com/Xilinx/Edge-AI-Platform-Tutorials/tree/master/docs/DPU-Integration). Then I would suggest you to use the SDK cross compile flow mentioned in that webpage and copy the compiled ELF into ```flower_classification``` and run the ELF directly. <br />

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
