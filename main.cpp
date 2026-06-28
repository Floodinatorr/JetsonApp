#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "inference.hpp"

int main(){
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    if(cuda_devices <= 0) {
        std::cerr<< "Hata: Cihazınızda CUDA destekli bir GPU bulunamadı."<<std::endl;
        return -1;
    }

    Engine engine("../models/yolo11s_fp16.engine", "../models/yolo11s.onnx");

    cv::VideoCapture cap("../video.mkv");
    if(!cap.isOpened()) {
        std::cerr<<"Hata: Video açılamadı. "<<std::endl;
        return 0;
    }
    
    cv::Mat frame;


    while (true)
    {
        cap >> frame;
        if(frame.empty()) break;

        engine.preprocess(frame);
        cv::imshow("JetsonApp", frame);

        if(cv::waitKey(1) == 27) break;
    }
    
    cap.release();
    cv::destroyAllWindows();


    /*
    // Görüntüyü ilk aşamada CPU'ya okutup RAM'e yükledim
    cv::Mat cpu_image = cv::imread("test.jpg", cv::IMREAD_COLOR);
    if(cpu_image.empty()){
        std::cerr<<"Hata: Resim dosyası yüklenemedi. "<<std::endl;
        return -1;
    }

    //Şimdiyse RAM üzerindeki bilgiyi GPU belleğine (VRAM) aktaracağım, bu sayede işlemleri daha hızlı gerçekleştirebilirim
    cv::cuda::GpuMat gpu_image, gpu_gray;
    gpu_image.upload(cpu_image);
    cv::cuda::cvtColor(gpu_image, gpu_gray, cv::COLOR_BGR2GRAY);

    //Son olarak GPU belleğindeki görüntüyü tekrardan RAM'e çekeceğim
    cv::Mat res;
    gpu_gray.download(res);
    
    cv::imshow("JetsonApp", res);
    cv::waitKey(0);
    */
    return 0;
    
}