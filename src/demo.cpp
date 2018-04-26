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
    cvtColor(image, image, COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3);

    FileStorage file("data/test.yml", FileStorage::READ);

// Write to file!
    Mat someMatrixOfAnyType = file.getFirstTopLevelNode().mat();
    someMatrixOfAnyType.convertTo(someMatrixOfAnyType, CV_16SC3);
    cout<<"read mat size: "<<someMatrixOfAnyType.size();

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

    Mat image2;
    image2 = imread("data/saved.jpg", IMREAD_COLOR); // Read the file
    cvtColor(image2, image2, COLOR_BGR2RGB);
    //cout<<someMatrixOfAnyType.row(200).colRange(0,10)<<endl;

    cout<<image_meta.first<<endl;
    auto out = infer.infer(molded_image, image_meta.first, image_anchors);
//    infer.PrintLabels(out[1]);
    cout<<"Tensor(0): "<<endl;
    Mat det = infer.tensor_to_cvmat(out[0], CV_32FC1, true);
    Mat mask = infer.tensor_to_cvmat(out[3], CV_32FC(81), true, true);
    cout<<det<<endl;
    cout<<mask.row(0).rows<<endl;
    cout<<mask.row(0).cols<<endl;
    cout<<mask.row(0).size()<<endl;
    //unmold_detections(det, )
/*    infer.tensor_to_cvmat(out[1], CV_32FC1, true);
    cout<<"Tensor(2): "<<endl;
    infer.tensor_to_cvmat(out[2], CV_32FC1, true);
    cout<<"Tensor(3): "<<endl;
    infer.tensor_to_cvmat(out[3], CV_32FC1, true);*/
    return 0;
}
