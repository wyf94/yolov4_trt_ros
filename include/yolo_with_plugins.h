#ifndef YOLO_WITH_PLUGINS_H
#define YOLO_WITH_PLUGINS_H
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "yolo_layer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <numeric>
#include "common.h"
#ifndef CUDA_CHECK

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#endif

namespace util {

class TrtYOLO {
private:
    int h;
    int w;
    int category_num = 80;
    std::string model;
    CUcontext ctx;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t trtCudaStream;
    // const char* INPUT_BLOB_NAME = "data";
    //  const char* OUTPUT_BLOB_NAME = "prob";
    void* testBuffer;

    int inputCount=0;
    int intputIndex;
    int outputIndex;
    void init_cuda();
    void load_engine();
    void* buffers[4];
    std::vector<void*> inputCudaBuffer;
    std::vector<void*> outputCudaBuffer;
    std::vector<void*> inputHostBuffer;
    std::vector<void*> outputHostBuffer;
    std::vector<int64_t> bindInputBufferSize;
    std::vector<int64_t> bindOutputBufferSize;
    void doInference();
void iou(std::vector<Box>& box_result, float nms_threshold);
std::vector<Box> nms_boxes(std::vector<Box>& box_result,float nms_threshold);
void postProcessing(std::vector<Box>& box_result,int width, int hegiht,float conf_th, float nms_threshold = 0.5);

public:
    void detect(cv::Mat& img,std::vector<Box>& box_result,float conf_th=0.3);
/*
    TrtYOLO(std::string path, int height, int width,int num): model(path),h(height),w(width) {
        category_num = num;
        init_cuda();
        load_engine();
        context = engine->createExecutionContext();
    }
*/
    void trt_init(std::string path, int height, int width,int num){
       model = path;
       h=height;
       w= width;
        category_num = num;
        init_cuda();
        load_engine();
        context = engine->createExecutionContext();
    }
    inline int64_t volume(const nvinfer1::Dims& d)
    {
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    }


    TrtYOLO() {}
    ~TrtYOLO() {
        cuCtxDestroy(ctx);
    }

    inline unsigned int getElementSize(nvinfer1::DataType t)
    {
        switch (t)
        {
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        }
        throw std::runtime_error("Invalid DataType.");
        return 0;
    }
    inline void* safeCudaMalloc(size_t memSize)
    {
        void* deviceMem;
        CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
        if (deviceMem == nullptr)
        {
            std::cerr << "Out of memory" << std::endl;
            exit(1);
        }
        return deviceMem;
    }


    inline void* safeCudaMallocHost(size_t memSize) {
        void* hostMem;
        CUDA_CHECK(cudaMallocHost(&hostMem, memSize));
        if (hostMem == nullptr)
        {
            std::cerr << "Out of memory" << std::endl;
            exit(1);
        }
        return hostMem;

    }

    //void detect(cv::Mat& img,float conf_th) {
//        cv::Mat img_resized = preprocess_yolo(img,);


    //}
};




}
#endif
