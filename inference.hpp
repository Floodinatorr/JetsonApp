#include<iostream>
#include<NvInfer.h>
#include<cuda_runtime_api.h>

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
    public:
        Engine(const std::string& enginePath, const std::string& onnxPath);
        ~Engine();
        void modelInitialize(const std::string& enginePath, const std::string& onnxPath);
};

