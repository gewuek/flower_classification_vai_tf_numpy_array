/*
-- (c) Copyright 2019 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

/* header file for DNNDK APIs */
#include <dnndk/dnndk.h>

using namespace std;
using namespace cv;

/* 7.71 GOP MAdds for ResNet50 */
//#define RESNET50_WORKLOAD (7.71f)
/* DPU Kernel name for ResNet50 */
#define KRENEL_FLOWER "flower_classification_0"
/* Input Node for Kernel ResNet50 */
#define INPUT_NODE      "conv2d_Conv2D"
/* Output Node for Kernel ResNet50 */
#define OUTPUT_NODE     "dense_1_MatMul"

const string baseImagePath = "./image/";

/*
* Software normalization
* normalize_image
*/
void normalize_image(const Mat& image, int8_t* data, float scale)
{
  for(int i = 0; i < 3; ++i) {
    for(int j = 0; j < image.rows; ++j) {
      for(int k = 0; k < image.cols; ++k) {
	       //data[j*image.rows*3+k*3+2-i] = (float(image.at<Vec3b>(j,k)[i])/255) * scale;
		   data[j*image.rows*3+k*3+i] = (float(image.at<Vec3b>(j,k)[i])/255) * scale;
		   //data[j*image.rows*3+k*3+2-i] = 64;
		   //printf("DATA BeforeTEST = %d\n\r", image.at<Vec3b>(j,k)[i]);
		   //printf("DATA AfterTEST = %d\n\r", data[j*image.rows*3+k*3+2-i]);
      }
     }
   }
}

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, vector<string> &images) {
    images.clear();
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
                (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
                images.push_back(name);
            }
        }
    }

    closedir(dir);
    sort(images.begin(), images.end());
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kinds file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const &path, vector<string> &kinds) {
    kinds.clear();
    fstream fkinds(path);
    if (fkinds.fail()) {
        fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
        exit(1);
    }
    string kind;
    while (getline(fkinds, kind)) {
        kinds.push_back(kind);
    }

    fkinds.close();
}

/**
 * @brief calculate softmax
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void CPUCalcSoftmax(const float *data, size_t size, float *result) {
    assert(data && result);
    double sum = 0.0f;

    for (size_t i = 0; i < size; i++) {
		// printf("data = %f\n\r", data[i]);
        result[i] = exp(data[i]);
        sum += result[i];
    }

    for (size_t i = 0; i < size; i++) {
        result[i] /= sum;
    }
}

/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float *d, int size, int k, vector<string> &vkinds) {
    assert(d && size > 0 && k > 0);
    priority_queue<pair<float, int>> q;

    for (auto i = 0; i < size; ++i) {
        q.push(pair<float, int>(d[i], i));
    }

    for (auto i = 0; i < k; ++i) {
        pair<float, int> ki = q.top();
        printf("top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
        vkinds[ki.second].c_str());
        q.pop();
    }
}

/**
 * @brief Run DPU Task for FlowerClassification
 *
 * @param taskFlowerClassification - pointer to Flower Classification Task
 *
 * @return none
 */
void runFlowerClassification(DPUTask *taskFlowerClassification) {
    assert(taskFlowerClassification);

    /* Mean value for Flower Classification specified in Caffe prototxt */
    vector<string> kinds, images;

    /* Load all image names.*/
    ListImages(baseImagePath, images);
    if (images.size() == 0) {
        cerr << "\nError: No images existing under " << baseImagePath << endl;
        return;
    }

    /* Load all kinds words.*/
    LoadWords(baseImagePath + "word_list.txt", kinds);
    if (kinds.size() == 0) {
        cerr << "\nError: No words exist in file words.txt." << endl;
        return;
    }

    /* Get channel count of the output Tensor for Flower Classification Task  */
    int channel = dpuGetOutputTensorChannel(taskFlowerClassification, OUTPUT_NODE);
    float *softmax = new float[channel];
    float *FCResult = new float[channel];
	float scale_input = dpuGetInputTensorScale(taskFlowerClassification, INPUT_NODE);
	printf("scale_input = %f;\n\r", scale_input);	
	DPUTensor *dpu_in = dpuGetInputTensor(taskFlowerClassification, INPUT_NODE);
	int8_t *data = dpuGetTensorAddress(dpu_in);

    for (auto &imageName : images) {
        cout << "\nLoad image : " << imageName << endl;
        /* Load image and Set image into DPU Task for Flower Classification */
        Mat image = imread(baseImagePath + imageName);
		normalize_image(image, data, scale_input);
		
        //dpuSetInputImage2(taskFlowerClassification, INPUT_NODE, image);
		//dpuSetInputImageWithScale(taskConv, CONV_INPUT_NODE, img, jw_mean, jw_scale);

        /* Launch RetNet50 Task */
        cout << "\nRun DPU Task for Flower Classification ..." << endl;
        dpuRunTask(taskFlowerClassification);

        /* Get DPU execution time (in us) of DPU Task */
        //long long timeProf = dpuGetTaskProfile(taskFlowerClassification);
        //cout << "  DPU Task Execution time: " << (timeProf * 1.0f) << "us\n";
        //float prof = (RESNET50_WORKLOAD / timeProf) * 1000000.0f;
        //cout << "  DPU Task Performance: " << prof << "GOPS\n";

        /* Get FC result and convert from INT8 to FP32 format */
        dpuGetOutputTensorInHWCFP32(taskFlowerClassification, OUTPUT_NODE, FCResult, channel);

        /* Calculate softmax on CPU and display TOP-5 classification results */
        CPUCalcSoftmax(FCResult, channel, softmax);
        TopK(softmax, channel, 5, kinds);
		//break;

        /* Display the impage */
        cv::imshow("Classification of Flowers", image);
        cv::waitKey(1000);
    }

    delete[] softmax;
    delete[] FCResult;
}

/**
 * @brief Entry for runing Flower Classification neural network
 *
 * @note DNNDK APIs prefixed with "dpu" are used to easily program &
 *       deploy Flower Classification on DPU platform.
 *
 */
int main(void) {
    /* DPU Kernel/Task for running Flower Classification */
    DPUKernel *kernelFlowerClassification;
    DPUTask *taskFlowerClassification;

    /* Attach to DPU driver and prepare for running */
    dpuOpen();

    /* Load DPU Kernel for Flower Classification */
    kernelFlowerClassification = dpuLoadKernel(KRENEL_FLOWER);

    /* Create DPU Task for Flower Classification */
    taskFlowerClassification = dpuCreateTask(kernelFlowerClassification, 0);

    /* Run Flower Classification Task */
    runFlowerClassification(taskFlowerClassification);

    /* Destroy DPU Task & free resources */
    dpuDestroyTask(taskFlowerClassification);

    /* Destroy DPU Kernel & free resources */
    dpuDestroyKernel(kernelFlowerClassification);

    /* Dettach from DPU driver & free resources */
    dpuClose();

    return 0;
}
