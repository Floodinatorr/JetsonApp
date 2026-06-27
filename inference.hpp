#include<iostream>

class Engine
{
    private:
        /* data */
    public:
        Engine(const std::string& enginePath, const std::string& onnxPath);
        ~Engine();
        void modelInitialize(const std::string& enginePath, const std::string& onnxPath);
};

