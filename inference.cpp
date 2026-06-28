#include "inference.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include <vector>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "preprocess.cuh"


Engine::Engine(const std::string& enginePath, const std::string& onnxPath) {
    modelInitialize(enginePath, onnxPath);
    allocateBuffers();
    cudaStreamCreate(&stream);
}

Engine::~Engine() {
    std::cout<<"Program kapatıldı."<<std::endl;
    delete(context);
    delete(engine);
    delete(runtime);
}

void Engine::modelInitialize(const std::string& enginePath, const std::string& onnxPath){
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);

    if(file.good()){
        //Engine File has found
        size_t size = file.tellg();
        std::vector<char> buffer(size);
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), size);
        file.close();

        runtime = nvinfer1::createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size());
        context = engine->createExecutionContext();

        std::cout<<"Engine model başarıyla yüklendi."<<std::endl;

    } else {
        //Engine File has not found so we have to initialize ONNX model

        std::cout<<"Engine model bulunamadı, ONNX model üzerinden yeni bir Engine model oluşturuluyor. Bu işlem 2-3dk sürebilir."<<std::endl;
        auto builder = nvinfer1::createInferBuilder(logger);
        auto network = builder->createNetworkV2(0);
        auto config = builder->createBuilderConfig();
        auto parser = nvonnxparser::createParser(*network, logger);

        if(!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))){
            std::cerr<<"ONNX Parse has failed."<<std::endl;
        }

        engine = builder->buildEngineWithConfig(*network, *config);
        auto serialized = engine->serialize();

        std::ofstream e_file(enginePath, std::ios::binary);
        e_file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
        e_file.close();

        delete(parser);
        delete(network);
        delete(config);
        delete(builder);

        std::cout<<"Engine file has created and saved to directory."<<std::endl;
    }
}

void Engine::allocateBuffers(){
    //Bu adımda bellekten gerekli boyutlarda yer alacağım
    auto inputDims = engine->getTensorShape("images");
    auto outputDims = engine->getTensorShape("output0");

    inputBytes = sizeof(uint16_t);
    for(int i = 0; i<inputDims.nbDims; i++) inputBytes *= inputDims.d[i];
    
    outputBytes = sizeof(uint16_t);
    for(int i = 0; i<outputDims.nbDims; i++) outputBytes *= outputDims.d[i];

    chwSize  = 640 * 640 * 3 * sizeof(__half);

    cudaMalloc((void**) &inputDevice, inputBytes);
    cudaMalloc((void**) &inputDeviceFP16, chwSize);
    cudaMalloc((void**) &outputDevice, outputBytes);

    context->setTensorAddress("images", inputDevice);
    context->setTensorAddress("output0", outputDevice);

}

void Engine::preprocess(const cv::Mat& image) {
    //Bu adımda OpenCV aracılığıyla aldığımız Frame objelerinin NvInfer in anlayabileceği bir hale getireceğim

    /*
     *     OPENCV GÖRÜNTÜSÜ             NVINFER INPUT SHAPE
     *     1920*1080                    640*640
     *     BGR                          RGB
     *     uint8_t                      float32
     *     HWC (Height-Width-Color)     CHW (Color-Height-Width)
     *
     */
    //
    // cv::Mat resized;
    // cv::resize(image, resized, cv::Size(640, 640));
    // cv::cvtColor(resized, resized, CV_BGR2RGB);
    // resized.convertTo(resized, CV_32FC3, 1.0f/255.0f);

    cv::cuda::GpuMat gpuFrame, gpuResized, gpuRgb;
    gpuFrame.upload(image);
    cv::cuda::resize(gpuFrame, gpuResized, cv::Size(640, 640));
    cv::cuda::cvtColor(gpuResized, gpuRgb, cv::COLOR_BGR2RGB);

    launchHwc2Chw(gpuRgb.ptr<unsigned char>(), reinterpret_cast<__half*>(inputDevice), 640, 640);
    //Burada özellikle böyle karmaşık bir kod yazdım aksi takdirde 3*640*640 adet işlem yapmam gerekecekti HWC --> CHW dönüşümü için ancak CPU'mu seviyorum, GPU ile işlemek daha mantıklı

    if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess)
    {
        std::cout << "CUDA preprocess error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}