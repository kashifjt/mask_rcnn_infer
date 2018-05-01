#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string.h>
#include "Config.h"
#include "utils.cpp"
#include "MaskInfer.h"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    string imageName("data/test01.jpg"); // by default
    Config myConfig;

    Mat image;
    image = imread(imageName.c_str(), IMREAD_COLOR); // Read the file
    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
//    Mat chn[3];
//    split(image, chn);
//    Mat subchn = chn[0](Range(100, 200), Range(100, 200));
//    cv::threshold(subchn, subchn, 100, 255, THRESH_BINARY);
//    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
//    imshow( "Display window", chn[0]);                // Show our image inside it.
//    waitKey(0); // Wait for a keystroke in the window

    cvtColor(image, image, COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3);

    Mat molded_image = image.clone();
    auto image_meta = mold_image(molded_image, myConfig);

    Mat image_anchors = get_anchors(molded_image.rows, molded_image.cols, myConfig);
    //cout<<"anchor[1000]: "<<molded_image.row(100).colRange(0,10)<<endl;
    //cout<<"anchor[2000]: "<<molded_image.row(200).colRange(0,10)<<endl;
    //cout<<"anchor[3000]: "<<molded_image.row(300).colRange(0,10)<<endl;
    //cout<<"anchor[4000]: "<<molded_image.row(400).colRange(0,10)<<endl;
    //cout<<"anchor[5000]: "<<molded_image.row(500).colRange(0,10)<<endl;
    //cout<<"anchor[6000]: "<<molded_image.row(600).colRange(0,10)<<endl;
    //cout<<"anchor[7000]: "<<molded_image.row(700).colRange(0,10)<<endl;
    //cout<<"anchor[8000]: "<<molded_image.row(800).colRange(0,10)<<endl;
    //cout<<"anchor[9000]: "<<molded_image.row(900).colRange(0,10)<<endl;
    //cvtColor(molded_image, molded_image, COLOR_RGB2BGR);

    //namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    //imshow( "Display window", molded_image);                // Show our image inside it.

    //waitKey(0); // Wait for a keystroke in the window

    MaskInfer infer(myConfig);
    string labels_path = "labels/coco.txt";
    string model_path = "model/mrcnn_model.pb";
    infer.ReadLabelsFile(labels_path);
    infer.LoadGraph(model_path);

    auto out = infer.infer(molded_image, image_meta.first, image_anchors);
//    infer.PrintLabels(out[1]);
    cout<<"Tensor(0): "<<endl;
    Mat det = infer.tensor_to_cvmat(out[0], CV_32FC1, true);
    Mat mask = infer.tensor_to_cvmat(out[3], CV_32FC(81), true, true);
    unmold_detections(det, mask, image.size(), molded_image.size(), image_meta.second);
    //Matx<Vec<float, 81>, 28, 28> mask2 = mask.at<Matx<Vec<float, 81>, 28, 28> >(7);
    //cout<<Mat(28, 28, CV_32FC(81), &mask2).channels()<<endl;
    //Mat bgr[81];
    //split(Mat(28, 28, CV_32FC(81), &mask2), bgr);
    //namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    //imshow( "Display window", bgr[25]);                // Show our image inside it.

    //waitKey(0); // Wait for a keystroke in the window
    //cout<<<<endl;
    //unmold_detections(det, )
/*  infer.tensor_to_cvmat(out[1], CV_32FC1, true);
    cout<<"Tensor(2): "<<endl;
    infer.tensor_to_cvmat(out[2], CV_32FC1, true);
    cout<<"Tensor(3): "<<endl;
    infer.tensor_to_cvmat(out[3], CV_32FC1, true);*/
    return 0;
}
