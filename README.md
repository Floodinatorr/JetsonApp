# JetsonApp

Python 3.11.15
CUDA 13.3 && TensorRT 11.1.0.6

You need to setup TensorRT in this directory:
```
/usr/local/tensortt/
```

WARNING: Engine files that created by NvInfer, is not cross platform so you can't use this models on another devices. You have to create a new engine file - but don't worry, it just takes a few minutes.