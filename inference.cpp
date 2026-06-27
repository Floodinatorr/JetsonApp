#include "inference.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include <vector>

Engine::Engine(const std::string& enginePath, const std::string& onnxPath) {
    modelInitialize(enginePath, onnxPath);
}

Engine::~Engine() {
    std::cout<<"Program kapatıldı."<<std::endl;
    delete(engine);
}

void Engine::modelInitialize(const std::string& enginePath, const std::string& onnxPath){
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);

    if(file.good()){
        //Engine File has found
        size_t size = file.tellg();
        std::vector<char> buffer(size);
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), size);

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

        std::ofstream file(enginePath, std::ios::binary);
        file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
        file.close();

        delete(parser);
        delete(network);
        delete(config);
        delete(builder);

        std::cout<<"Engine file has created and saved to directory."<<std::endl;
    }
}