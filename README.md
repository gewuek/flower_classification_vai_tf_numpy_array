# flower_classification_dnndk_v1
This is a simple example about how to train a ConNet model from labeled dataset with TensorFlow and then use DNNDK tools to deploy the model into ZCU102 board.

The whole design is trained and deployed using Ubuntu 18.04 + TensorFlow 1.12 + DNNDK 3.1 + PetaLinux 2019.1
To make it easier I just make my model ovefit the dataset. All the training/validation/calibration data are just from the same dataset.

# Training and DNNDK tools Test flow
Please install the DNNDK 3.1 develop environment according to https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf before starting the custom model flow.
Make sure you can run DNNDK examples.

1. Download kaggle flower dataset from https://www.kaggle.com/alxmamaev/flowers-recognition
2. unzip the folder and copy the files into flower_classification_dnndk_v1/x86/flowers folder. So that the directory would like below:

3. Navigate into the flower_classification_dnndk_v1/x86/ folder
4. Load images and labels into dataset
python3 ./load_data.py
5. Train data
python3 ./train_data.py
6. Evaluate the trained model(Opitional)
python3 ./evaluate_trained_model.py
7. Freeze the model
python3 ./freeze_model.py
8. Evaluate the frozen model(Opitional)
python3 ./evaluate_frozen_model.py
9. Quantize the graph, using chmod u+x ./decent_q.sh if necessary
./decent_q.sh
10. Evaluate quantized graph
python3 ./evaluate_quantized_graph.py (Optional)
11. Compile the quantized model into elf using DNNC, use chmod u+x ./dnnc.sh if necessary
./dnnc.sh


# Reference

https://www.youtube.com/watch?v=VwVg9jCtqaU&t=112s

https://www.kaggle.com/alxmamaev/flowers-recognition

https://www.youtube.com/watch?v=j-3vuBynnOE

https://github.com/tensorflow/docs/blob/r1.12/site/en/tutorials/load_data/images.ipynb

https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf

https://github.com/Xilinx/Edge-AI-Platform-Tutorials/tree/3.1/docs/DPU-Integration

https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb

https://stackoverflow.com/questions/51278213/what-is-the-use-of-a-pb-file-in-tensorflow-and-how-does-it-work/51281809#51281809
