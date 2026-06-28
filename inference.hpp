#include<iostream>
#include<NvInfer.h>
#include<cuda_runtime_api.h>
#include<cuda_fp16.h>
#include<vector>
#include<opencv2/opencv.hpp>

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class Engine
{
    private:
        TRTLogger logger;
        nvinfer1::IRuntime* runtime;
        nvinfer1::ICudaEngine* engine;
        nvinfer1::IExecutionContext* context;
        size_t inputBytes, outputBytes, chwSize;
        unsigned char* inputDevice;
        unsigned char* outputDevice;
        __half* inputDeviceFP16;
        std::vector<__half> inputHost;
        cudaStream_t stream;
        

    public:
        Engine(const std::string& enginePath, const std::string& onnxPath);
        ~Engine();
        void modelInitialize(const std::string& enginePath, const std::string& onnxPath);
        void allocateBuffers();
        void preprocess(const cv::Mat& image);
};

