#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

int main(){
    std::string imagePath = "test.jpg";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::imshow("Jetson App", image);
    cv::waitKey(0);

    std::cout<<"Hello World!"<<std::endl;
    std::cout<<"Hello World2"<<std::endl;
    return 0;
}