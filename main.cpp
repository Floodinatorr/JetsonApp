#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

// Burada yapmış olduğum RAM --> VRAM --> RAM işlemleri CPU ve GPU belleğinin ortak olmadığı cihazlar için
// Eğer Jetson gibi modülleriniz varsa (bu modüller CPU ve GPU için RAM'lerini ortak kullanır) bu işlemler, sıkılınca duvara pinpon topu fırlatıp durmaktan farksızdır
// Jetson üzerinde GPU'yu kullanarak hesaplama yapmak için gereken kodlar sonrasında commit geçmişinde belirtilecektir


int main(){
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    if(cuda_devices <= 0) {
        std::cerr<< "Hata: Cihazınızda CUDA destekli bir GPU bulunamadı."<<std::endl;
        return -1;
    } 

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

    return 0;
}