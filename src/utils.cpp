// utils.cpp
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

class Config;

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
    //shifts_y = np.arange(0, shape.first, anchor_stride) * feature_stride
    //shifts_x = np.arange(0, shape.second, anchor_stride) * feature_stride
    //shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    vector<float> range_y = arange((float) 0.0,(float) shape.first, (float) config.RPN_ANCHOR_STRIDE);
    Mat mat_range_y(1, range_y.size(), CV_32FC1, range_y.data());
    vector<float> range_x = arange((float) 0.0,(float) shape.second, (float) config.RPN_ANCHOR_STRIDE);
    Mat mat_range_x(1, range_x.size(), CV_32FC1, range_x.data());
    pair<Mat, Mat> shifts_mesh = meshgrid(mat_range_x, mat_range_y);
    // cout<<"shifts_x size: "<<shifts_mesh.first.size()<<endl;
    // cout<<"shifts_y size: "<<shifts_mesh.second.size()<<endl;
    // float a[3][4]={{1,2,3,4},{5,6,7,8},{9,10,11,12}};
    // float b[3]={1,2,3};
    // Mat mat_a(3, 4, CV_32FC1, &a);
    // Mat mat_b(1, 3, CV_32FC1, &b);
    // cout<<"***check begin***";
    // auto mesh_check = meshgrid(mat_b, mat_a);
    // cout<<mesh_check.first<<endl;
    // cout<<mesh_check.second<<endl;
    //Enumerate combinations of shifts, widths, and heights
    //box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    //box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    pair<Mat, Mat> box_widths_centers_x = meshgrid(widths, shifts_mesh.first);
    pair<Mat, Mat> box_heights_centers_y = meshgrid(heights, shifts_mesh.second);
    // cout<<"widths size: "<<widths.size()<<endl;
    // cout<<"shifts_x size: "<<shifts_mesh.first.size()<<endl;
    // cout<<"box_widths size: "<<box_widths_centers_x.first.size()<<endl;
    // cout<<"box_centers_x size: "<<box_widths_centers_x.second.size()<<endl;
    // cout<<"box_heights size: "<<box_heights_centers_y.first.size()<<endl;
    // cout<<"box_centers_y size: "<<box_heights_centers_y.second.size()<<endl;
    //Reshape to get a list of (y, x) and a list of (h, w)
    //box_centers = np.stack(
    //    [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    //box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    Mat box_centers;
    vconcat(box_heights_centers_y.second.reshape(1,1),
          box_widths_centers_x.second.reshape(1,1), box_centers);
    Mat box_sizes;
    vconcat(box_heights_centers_y.first.reshape(1,1),
          box_widths_centers_x.first.reshape(1,1), box_sizes);
    cout<<"box_centers shape: "<<box_centers.size()
      <<"box_sizes shape: "<<box_sizes.size()<<endl;

    // Convert to corner coordinates (y1, x1, y2, x2)
    // boxes = np.concatenate([box_centers - 0.5 * box_sizes,
    //                         box_centers + 0.5 * box_sizes], axis=1)
    Mat boxes;
    vconcat(box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes, boxes);
    cout<<"boxes shape: "<<boxes.size()<<endl;

    return boxes;
}

Mat generate_pyramid_anchors(Config config, vector<pair<int, int> > backbone_shapes)
{
    Mat anchors;
    cout<<sizeof(config.RPN_ANCHOR_SCALES)<<endl;
    for (int i = 0; i < sizeof(config.RPN_ANCHOR_SCALES)/ sizeof(int); ++i)
    {
        Mat anchor = generate_anchors(config.RPN_ANCHOR_SCALES[i], backbone_shapes[i],
                         config.BACKBONE_STRIDES[i], config);
        cout<<"Anchors Size: "<<anchors.size()<<endl;
        cout<<"Anchor Size: "<<anchor.size()<<endl;
        if (i == 0) anchors = anchor.clone();
        else hconcat(anchors, anchor, anchors);
        //anchors.push_back(anchor);
    }
    return anchors;
}

/*
def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes
*/

Mat get_anchors(int img_height, int img_width, Config config)
{
    vector<pair<int, int> > backbone_shapes;
    for (int i = 0; i < sizeof(config.BACKBONE_STRIDES)/ sizeof(int); ++i)
    {
        backbone_shapes.push_back(make_pair(ceil((float) img_height/config.BACKBONE_STRIDES[i]),
                                            ceil((float) img_width/config.BACKBONE_STRIDES[i])));
    }
    Mat a = generate_pyramid_anchors(config,backbone_shapes);
    return a;
}
