// utils.cpp
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

class Config;

pair<float, vector<int> > resize_image(Mat& image, int min_dim, int max_dim){
    int h = image.size().height, w = image.size().width;
    float scale = MAX(1, min_dim / (float) MIN(h, w));
    if (round(MAX(h, w) * scale) > max_dim)
        scale = max_dim / (float) MAX(h, w);
    if (scale!=1.0)
    {
        Size size(round(scale * w), round(scale *h));
        resize(image, image, size);
    }
    h = image.size().height, w = image.size().width;
    int top_pad = (max_dim - h) / 2;
    int bottom_pad = max_dim - h - top_pad;
    int left_pad = (max_dim - w) / 2;
    int right_pad = max_dim - w - left_pad;
    copyMakeBorder(image, image, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, 0);
    int arr[] = {top_pad, left_pad, h + top_pad, w + left_pad};
    vector<int> window(arr, arr + sizeof(arr) / sizeof(arr[0]));
    return make_pair(scale, window);
}

vector<float> mold_image(Mat& image, Config config)
{
    float orig_dim[] = {0, image.rows, image.cols, image.channels()};
    pair<float, vector<int> > scale_window = resize_image(image, config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM);
    Scalar s(config.MEAN_PIXEL[0], config.MEAN_PIXEL[1], config.MEAN_PIXEL[2]);
    subtract(image, s, image);
    vector<float> image_meta(orig_dim, orig_dim + sizeof(orig_dim) / sizeof(orig_dim[0]));
    image_meta.push_back(image.rows); image_meta.push_back(image.cols); image_meta.push_back(image.channels());
    image_meta.insert(image_meta.end(), scale_window.second.begin(), scale_window.second.end());
    image_meta.push_back(scale_window.first);
    vector<float> active_classes(config.NUM_CLASSES, 0);
    image_meta.insert(image_meta.end(), active_classes.begin(), active_classes.end());
    return image_meta;
}

void get_anchors(int img_height, int img_width, Config config)
{
    vector<pair<int, int> > backbone_shapes;
    for (int i = 0; i < sizeof(config.BACKBONE_STRIDES)/ sizeof(int); ++i)
    {
        backbone_shapes.push_back(make_pair(ceil((float) img_height/config.BACKBONE_STRIDES[i]),
                                            ceil((float) img_height/config.BACKBONE_STRIDES[i])));
    }
}