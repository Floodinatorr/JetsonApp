#include "inference.hpp"
#include <fstream>
#include "NvOnnxParser.h"


Engine::Engine(const std::string& enginePath, const std::string& onnxPath) {
    modelInitialize(enginePath, onnxPath);
}

Engine::~Engine() {
    std::cout<<"Program kapatıldı."<<std::endl;
}

void Engine::modelInitialize(const std::string& enginePath, const std::string& onnxPath){
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);

    if(file.good()){
        //Engine File has found

    } else {
        //Engine File has not found so we need to initialize ONNX model

    }
}