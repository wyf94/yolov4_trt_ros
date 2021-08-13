#include "yolo_with_plugins.h"
#include "logging.h"
#include <map>
namespace util
{
    static Logger gLogger;
    bool comp(const Box& a, const Box& b)
    {
        return a.score > b.score;
    }
    bool comp_conf(const Box& a, const Box& b)
    {
        return a.box_conf > b.box_conf;
    }
    //similar to clamp(),within the range of [0,standard]
    int withinStandard(int value, int standard)
    {
        if (standard < 0)
        {
            std::cout << "standard should be greater than 0 " << std::endl;
            exit(1);
        }
        if (value < 0)
        {
            return 0;
        }
        if (value > standard)
        {
            return standard;
        }
        return value;
    }
    void preprocessed_yolo(void* inputPtr, cv::Mat& img, float height, float width)
    {
        cv::resize(img, img, cv::Size(width, height));
        cv::Mat rgb;
        cv::cvtColor(img, rgb, CV_BGR2RGB);
        cv::Mat img_float;
        rgb.convertTo(img_float, CV_32FC3, 1 / 255.0);
        //NCHW to NHWC
        std::vector<cv::Mat> input_channels(3);
        cv::split(img_float, input_channels);
        std::vector<float> result(height * width * 3);
        int channelLength = height * width;
        auto data = result.data();
        int totalLength = 0;
        for (int i = 0; i < 3; ++i)
        {
            memcpy(data, input_channels[i].data, channelLength * sizeof(float));
            totalLength += channelLength;
            data += channelLength;
        }
        memcpy(inputPtr, result.data(), totalLength * sizeof(float));
    }

    void getYoloGridSize(std::string model, int height, int width, std::vector<int>& result)
    {
        if (model.find("yolov3") != std::string::npos)
        {
            if (model.find("tiny") != std::string::npos)
            {
                result.push_back(floor(height / 32) * floor(width / 32));
                result.push_back(floor(height / 16) * floor(width / 16));
            }
            else
            {
                result.push_back(floor(height / 32) * floor(width / 32));
                result.push_back(floor(height / 16) * floor(width / 16));
                result.push_back(floor(height / 8) * floor(width / 8));
            }
        }
        else if (model.find("yolov4") != std::string::npos)
        {
            if (model.find("tiny") != std::string::npos)
            {
                result.push_back(floor(height / 32) * floor(width / 32));
                result.push_back(floor(height / 16) * floor(width / 16));
            }
            else
            {
                result.push_back(floor(height / 8) * floor(width / 8));
                result.push_back(floor(height / 16) * floor(width / 16));
                result.push_back(floor(height / 32) * floor(width / 32));
            }

        }

    }

    void TrtYOLO::init_cuda()
    {
        CUresult rc;
        CUdevice dev;
        rc = cuInit(0);
        if (rc != CUDA_SUCCESS)
        {
            std::cout << "CUDA not initialized!" << std::endl;
            exit(1);
        }
        rc = cuDeviceGet(&dev, 0);
        if (rc != CUDA_SUCCESS)
        {
            std::cout << "There is error on cuDeviceGet " << rc << std::endl;
            exit(1);
        }
        rc = cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, dev);
        if (rc != CUDA_SUCCESS)
        {
            std::cout << "There is error on cuCtxCreate " << rc << std::endl;
            exit(1);
        }
    }

    void TrtYOLO::load_engine()
    {
        //load engine
        std::string TRTbin = model + ".trt";
        std::ifstream fin(TRTbin);
        std::string cached_engine = "";
        while (fin.peek() != EOF)
        {
            std::stringstream buffer;
            buffer << fin.rdbuf();
            cached_engine.append(buffer.str());
        }
        fin.close();
        if (TRTbin.size())
        {
            nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
            assert(runtime != nullptr);
            nvinfer1::YoloPluginCreator mycreate;
            int numcreators = 0;
            nvinfer1::IPluginCreator* const* tmpList = getPluginRegistry()->getPluginCreatorList(&numcreators);
            engine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
        }
        if (!engine)
        {
            std::cout << "TRTEngine load fail" << std::endl;
            exit(1);
        }
        std::vector<int> grid_sizes;
        getYoloGridSize(model, h, w, grid_sizes);
        const int maxBatchSize = 1;
        int output_idx = 0;
        context = engine->createExecutionContext();
        assert(context != nullptr);
        int nbBindings = engine->getNbBindings();

        CUDA_CHECK(cudaStreamCreate(&trtCudaStream));
        int myDevice;
        CUDA_CHECK(cudaGetDevice(&myDevice));
        //allocate device and host buffer
        for (int i = 0; i < nbBindings; ++i)
        {
            nvinfer1::Dims dims = engine->getBindingDimensions(i);
            nvinfer1::DataType dtype = engine->getBindingDataType(i);
            int64_t size = volume(dims) * maxBatchSize;
            int64_t totalSize = size * getElementSize(dtype);
            if (engine->bindingIsInput(i))
            {
                CUDA_CHECK(cudaMalloc(&buffers[i], totalSize))
                    bindInputBufferSize.push_back(totalSize);
                void* hostPtr = nullptr;
                CUDA_CHECK(cudaMallocHost(&hostPtr, totalSize));
                inputHostBuffer.push_back(hostPtr);
            }
            else
            {
                assert(size == (grid_sizes[output_idx] * 3 * 7 * maxBatchSize));
                bindOutputBufferSize.push_back(totalSize);
                output_idx = output_idx + 1;
                CUDA_CHECK(cudaMalloc(&buffers[i], totalSize));
                void* hostPtr = nullptr;
                CUDA_CHECK(cudaMallocHost(&hostPtr, totalSize));
                outputHostBuffer.push_back(hostPtr);
            }
        }

    }

    void TrtYOLO::doInference()
    {
        int myDevice;
        cudaError_t cudaError;
        cudaError = cudaGetDevice(&myDevice);
        void* device0Mem;
        cudaStream_t tempStream;
        CUDA_CHECK(cudaStreamCreate(&tempStream));
        int inputIndex = 0;
        assert(inputHostBuffer.size() == 1);
        //copy preprocessed data to cuda
        CUDA_CHECK(cudaMemcpyAsync(buffers[0], inputHostBuffer[0], bindInputBufferSize[0], cudaMemcpyHostToDevice, trtCudaStream));
        context->enqueue(1, buffers, trtCudaStream, nullptr);
        //fetch detection result
        for (int i = 0; i < outputHostBuffer.size(); i++)
        {
            CUDA_CHECK(cudaMemcpyAsync(outputHostBuffer[i], buffers[i + 1], bindOutputBufferSize[i], cudaMemcpyDeviceToHost, trtCudaStream));
        }
        cudaStreamSynchronize(trtCudaStream);
    }

    void TrtYOLO::iou(std::vector<Box>& box_result, float nms_threshold)
    {
        std::vector<float> box_x1;
        std::vector<float> box_y1;
        std::vector<float> box_x2;
        std::vector<float> box_y2;
        std::vector<float> width1;
        std::vector<float> height1;
        std::vector<float> intersection;
        std::vector<float> areas;
        Box box1 = box_result[0];
        float area_0 = box1.end_x * box1.end_y;
        for (int i = 1; i < box_result.size(); i++)
        {
            areas.push_back(box_result[i].end_x * box_result[i].end_y);
            box_x1.push_back(std::max(box1.start_x, box_result[i].start_x));
            box_y1.push_back(std::max(box1.start_y, box_result[i].start_y));
            box_x2.push_back(std::min(box1.start_x + box1.end_x, box_result[i].start_x + box_result[i].end_x));
            box_y2.push_back(std::min(box1.start_y + box1.end_y, box_result[i].start_y + box_result[i].end_y));
        }
        float zero = 0.0;
        for (int i = 0; i < box_x1.size(); i++)
        {
            width1.push_back(std::max(zero, box_x2[i] - box_x1[i] + 1));
            height1.push_back(std::max(zero, box_y2[i] - box_y1[i] + 1));
            intersection.push_back(width1[i] * height1[i]);
        }

        int del_num = 0;
        for (int i = 0; i < box_x1.size(); i++)
        {
            float iou_num = intersection[i] / (area_0 + areas[i] - intersection[i]);
            if (iou_num > nms_threshold)
            {
                box_result.erase(box_result.begin() + i + 1 - del_num);
                del_num++;
            }
        }

    }

    std::vector<Box> TrtYOLO::nms_boxes(std::vector<Box>& box_result, float nms_threshold)
    {
        std::vector<Box> returnVec;
        sort(box_result.begin(), box_result.end(), comp_conf);
        while (box_result.size() > 0)
        {
            Box tempBox = box_result[0];
            returnVec.push_back(tempBox);
            iou(box_result, nms_threshold);
            box_result.erase(box_result.begin());

        }
        return returnVec;

    }

    void TrtYOLO::postProcessing(std::vector<Box>& box_result, int width, int height, float conf_th, float nms_threshold)
    {
        std::vector<std::vector<Box>> tempBoxResult;
        std::vector<int> total_class(80, -1);
        int total_cls_num = -1;

        for (int i = 0; i < bindOutputBufferSize.size(); i++) {
            int times = bindOutputBufferSize[i] / (7 * sizeof(float));
            assert(bindOutputBufferSize[i] % (7 * sizeof(float)) == 0);
            for (int j = 0; j < times; j++)
            {
                float col_4 = *((float*)outputHostBuffer[i] + 7 * j + 4);
                float col_6 = *((float*)outputHostBuffer[i] + 7 * j + 6);
                float temp_score = col_4 * col_6;
                if (temp_score >= conf_th)
                {
                    Box tempBox;
                    float col_0 = *((float*)outputHostBuffer[i] + 7 * j + 0) * width;
                    float col_1 = *((float*)outputHostBuffer[i] + 7 * j + 1) * height;
                    float col_2 = *((float*)outputHostBuffer[i] + 7 * j + 2) * width;
                    float col_3 = *((float*)outputHostBuffer[i] + 7 * j + 3) * height;
                    float col_5 = *((float*)outputHostBuffer[i] + 7 * j + 5);
                    tempBox.start_x = col_0;
                    tempBox.start_y = col_1;
                    tempBox.end_x = col_2;
                    tempBox.end_y = col_3;
                    tempBox.score = temp_score;
                    tempBox.box_conf = col_4;
                    tempBox.box_class = (int)(col_5);
                    int b_class = (int)(col_5);
                    if (total_class[b_class] == -1)
                    {
                        std::vector<Box> t_box;
                        t_box.push_back(tempBox);
                        tempBoxResult.push_back(t_box);
                        total_cls_num++;
                        total_class[b_class] = total_cls_num;
                    }
                    else
                    {
                        tempBoxResult[total_class[b_class]].push_back(tempBox);
                    }
                }
            }
        }
        std::vector<Box> finalBox;
        for (int i = 0; i < tempBoxResult.size(); i++)
        {
            std::vector<Box> clasBox = tempBoxResult[i];
            std::vector<Box> tempReturn = nms_boxes(clasBox, nms_threshold);
            finalBox.insert(finalBox.end(), tempReturn.begin(), tempReturn.end());
        }
        for (int i = 0; i < finalBox.size(); i++)
        {
            finalBox[i].start_x = (int)(finalBox[i].start_x + 0.5);
            finalBox[i].start_y = (int)(finalBox[i].start_y + 0.5);
            finalBox[i].end_x = (int)(finalBox[i].start_x + finalBox[i].end_x + 0.5);
            finalBox[i].end_y = (int)(finalBox[i].start_y + finalBox[i].end_y + 0.5);
        }
        box_result = finalBox;
    }
    void TrtYOLO::detect(cv::Mat& img, std::vector<Box>& box_result, float conf_th)
    {
        cv::Mat imageDetect = img;
        preprocessed_yolo(TrtYOLO::inputHostBuffer[0], imageDetect, TrtYOLO::h, TrtYOLO::w);

        CUresult a;

        if (ctx)
        {
            a = cuCtxPushCurrent(ctx);
            assert(a == CUDA_SUCCESS);
        }
        doInference();

        a = cuCtxPopCurrent(&ctx);
        assert(a == CUDA_SUCCESS);
        postProcessing(box_result, img.cols, img.rows, conf_th);

        for (int i = 0; i < box_result.size(); i++)
        {
            box_result[i].start_x = withinStandard(box_result[i].start_x, img.cols - 1);
            box_result[i].start_y = withinStandard(box_result[i].start_y, img.rows - 1);
            box_result[i].end_x = withinStandard(box_result[i].end_x, img.cols - 1);
            box_result[i].end_y = withinStandard(box_result[i].end_y, img.rows - 1);
        }

    }
}

