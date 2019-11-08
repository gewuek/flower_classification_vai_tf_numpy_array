# flower_classification_dnndk_v1
This is a simple example about how to train a ConNet model from labeled dataset and then use DNNDK tools to deploy the model into ZCU102 board.

The whole design is trained and deployed using Ubuntu 18.04 + TensorFlow 1.12 + DNNDK 3.1 + PetaLinux 2019.1
To make it easier I just make my model ovefit the dataset

# Reference

https://www.youtube.com/watch?v=VwVg9jCtqaU&t=112s

https://www.kaggle.com/alxmamaev/flowers-recognition

https://www.youtube.com/watch?v=j-3vuBynnOE

https://github.com/tensorflow/docs/blob/r1.12/site/en/tutorials/load_data/images.ipynb

https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf

https://github.com/Xilinx/Edge-AI-Platform-Tutorials/tree/3.1/docs/DPU-Integration

https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb

https://stackoverflow.com/questions/51278213/what-is-the-use-of-a-pb-file-in-tensorflow-and-how-does-it-work/51281809#51281809
