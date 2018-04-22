// Config.h
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

class Config {
public:
    int IMAGE_MIN_DIM;
    int IMAGE_MAX_DIM;
    float MEAN_PIXEL[3];
    int NUM_CLASSES;
    int RPN_ANCHOR_SCALES[5];
    float RPN_ANCHOR_RATIOS[3];
    int BACKBONE_STRIDES[5];
    int RPN_ANCHOR_STRIDE;

    Config()
    {
        IMAGE_MIN_DIM = 800;
        IMAGE_MAX_DIM = 1024;
        MEAN_PIXEL[0] = 123.7; MEAN_PIXEL[1] = 116.8; MEAN_PIXEL[2] = 103.9;
        NUM_CLASSES = 81;
        RPN_ANCHOR_SCALES[0] = 32; RPN_ANCHOR_SCALES[1] = 64; RPN_ANCHOR_SCALES[2] = 128;
        RPN_ANCHOR_SCALES[3] = 256; RPN_ANCHOR_SCALES[4] = 512;
        RPN_ANCHOR_RATIOS[0] = 0.5; RPN_ANCHOR_RATIOS[1] = 1; RPN_ANCHOR_RATIOS[2] = 2;
        BACKBONE_STRIDES[0] = 4; BACKBONE_STRIDES[1] = 8; BACKBONE_STRIDES[2] = 16;
        BACKBONE_STRIDES[3] = 32; BACKBONE_STRIDES[4] = 64;
        RPN_ANCHOR_STRIDE = 1;
    }

};
