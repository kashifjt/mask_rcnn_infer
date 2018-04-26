// utils.cpp
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc.hpp"
#include "Config.h"

using namespace cv;
using namespace std;

template<typename T>
vector<T> arange(T start, T stop, T step = 1) {
    vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}

pair<Mat, Mat> meshgrid(Mat& x,Mat& y)
{
    Mat X = x.clone();
    X = X.reshape(0, 1);
    Mat Y = y.clone();
    Y = Y.reshape(0, 1);
    int numRows = Y.size().width, numCols = X.size().width;
    for (int i = 0; i < numRows-1; ++i)
    X.push_back(X.row(i));
    for (int i = 0; i < numCols-1; ++i)
    Y.push_back(Y.row(i));
    transpose(Y, Y);
    return make_pair(X, Y);
}

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
    vector<int> window;
    window.push_back(top_pad); window.push_back(left_pad); window.push_back(h + top_pad); window.push_back(w + left_pad);
    return make_pair(scale, window);
}

pair<Mat, vector<int> > mold_image(Mat& image, Config config)
{
    float orig_dim[] = {0.0, (float) image.rows, (float) image.cols, (float) image.channels()};
    pair<float, vector<int> > scale_window = resize_image(image, config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM);
    Scalar ch_mean(config.MEAN_PIXEL[0], config.MEAN_PIXEL[1], config.MEAN_PIXEL[2]);
    subtract(image, ch_mean, image);
    vector<float> image_meta(orig_dim, orig_dim + sizeof(orig_dim) / sizeof(orig_dim[0]));
    image_meta.push_back(image.rows); image_meta.push_back(image.cols); image_meta.push_back(image.channels());
    image_meta.insert(image_meta.end(), scale_window.second.begin(), scale_window.second.end());
    image_meta.push_back(scale_window.first);
    vector<float> active_classes(config.NUM_CLASSES, 0);
    image_meta.insert(image_meta.end(), active_classes.begin(), active_classes.end());
    return make_pair(Mat (1,image_meta.size(), CV_32FC1,image_meta.data()).clone(), scale_window.second);
}

Mat generate_anchors(int scale, pair<int, int> shape,
                     int feature_stride, Config config)
{
    Mat scales(1, sizeof(config.RPN_ANCHOR_RATIOS)/ sizeof(int),
             CV_32FC1,Scalar(scale));
    Mat ratios(1, sizeof(config.RPN_ANCHOR_RATIOS)/ sizeof(int),
             CV_32FC1, &config.RPN_ANCHOR_RATIOS);
    Mat sqrt_ratios;
    sqrt(ratios, sqrt_ratios);
    Mat heights;
    divide(scales, sqrt_ratios, heights);
    Mat widths;
    multiply(scales, sqrt_ratios, widths);

    //Enumerate shifts in feature space
    vector<float> range_y = arange((float) 0.0,(float) shape.first, (float) config.RPN_ANCHOR_STRIDE);
    Mat mat_range_y(1, range_y.size(), CV_32FC1, range_y.data());
    multiply(mat_range_y, Scalar(feature_stride), mat_range_y);
    vector<float> range_x = arange((float) 0.0,(float) shape.second, (float) config.RPN_ANCHOR_STRIDE);
    Mat mat_range_x(1, range_x.size(), CV_32FC1, range_x.data());
    multiply(mat_range_x, Scalar(feature_stride), mat_range_x);
    pair<Mat, Mat> shifts_mesh = meshgrid(mat_range_x, mat_range_y);

    //Enumerate combinations of shifts, widths, and heights
    pair<Mat, Mat> box_widths_centers_x = meshgrid(widths, shifts_mesh.first);
    pair<Mat, Mat> box_heights_centers_y = meshgrid(heights, shifts_mesh.second);

    // //Reshape to get a list of (y, x) and a list of (h, w)
    Mat box_centers;
    vconcat(box_heights_centers_y.second.reshape(1,1),
          box_widths_centers_x.second.reshape(1,1), box_centers);
    Mat box_sizes;
    vconcat(box_heights_centers_y.first.reshape(1,1),
          box_widths_centers_x.first.reshape(1,1), box_sizes);

    // Convert to corner coordinates (y1, x1, y2, x2)
    Mat boxes;
    vconcat(box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes, boxes);
    return boxes;
}

Mat generate_pyramid_anchors(Config config, vector<pair<int, int> > backbone_shapes)
{
    Mat anchors;
    for (int i = 0; i < sizeof(config.RPN_ANCHOR_SCALES)/ sizeof(int); ++i)
    {
        Mat anchor = generate_anchors(config.RPN_ANCHOR_SCALES[i], backbone_shapes[i],
                         config.BACKBONE_STRIDES[i], config);
        if (i == 0) anchors = anchor.clone();
        else hconcat(anchors, anchor, anchors);
    }
    return anchors;
}

Mat get_anchors(int img_height, int img_width, Config config)
{
    vector<pair<int, int> > backbone_shapes;
    for (int i = 0; i < sizeof(config.BACKBONE_STRIDES)/ sizeof(int); ++i)
    {
        backbone_shapes.push_back(make_pair(ceil((float) img_height/config.BACKBONE_STRIDES[i]),
                                            ceil((float) img_width/config.BACKBONE_STRIDES[i])));
    }
    Mat a = generate_pyramid_anchors(config,backbone_shapes);
    transpose(a, a); //Correct reshape(1,1) affect

    //Normalize Box
    float scale[4] = {(float)img_height - 1, (float)img_width - 1, (float)img_height - 1, (float)img_width - 1};
    float shift[4] = {0, 0, 1, 1};
    Mat mat_shift = Mat(1, 4, CV_32FC1, &shift);
    Mat mat_scale = Mat(1, 4, CV_32FC1, &scale);
    for (int i = 0; i < a.rows; ++i)
    {
      a.row(i) = (a.row(i) - mat_shift) / mat_scale;
    }

    return a;
}

void unmold_detections(Mat detections, Mat mrcnn_mask, Size original_image_shape,
                      Size image_shape, vector<int> window)
{
    Mat nonZeros;
    findNonZero(detections.col(4), nonZeros);
    cout<<nonZeros;
}
