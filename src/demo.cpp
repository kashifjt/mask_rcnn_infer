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
    image.convertTo(image, 5);

    Mat molded_image = image.clone();
    auto image_meta = mold_image(molded_image, myConfig);
    Mat image_anchors = get_anchors(molded_image.rows, molded_image.cols, myConfig);
    cout<<"anchors rows: "<<image_anchors.rows<<", anchors cols: "<<image_anchors.cols<<endl;
    cout<<"anchor[1000]: "<<image_anchors.row(1000)<<endl;
    cout<<"anchor[2000]: "<<image_anchors.row(2000)<<endl;
    cout<<"anchor[3000]: "<<image_anchors.row(3000)<<endl;
    cout<<"anchor[4000]: "<<image_anchors.row(4000)<<endl;
    cout<<"anchor[5000]: "<<image_anchors.row(5000)<<endl;
    cout<<"anchor[6000]: "<<image_anchors.row(6000)<<endl;
    cout<<"anchor[7000]: "<<image_anchors.row(7000)<<endl;
    cout<<"anchor[8000]: "<<image_anchors.row(8000)<<endl;
    cout<<"anchor[9000]: "<<image_anchors.row(9000)<<endl;
    //cvtColor(image, image, COLOR_RGB2BGR);

    //namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    //imshow( "Display window", image);                // Show our image inside it.

    //waitKey(0); // Wait for a keystroke in the window
    MaskInfer infer(myConfig);
    string labels_path = "labels/coco.txt";
    string model_path = "model/mrcnn_model.pb";
    infer.ReadLabelsFile(labels_path);
    infer.LoadGraph(model_path);
    auto out = infer.infer(molded_image, image_meta.first, image_anchors);
    infer.PrintLabels(out[1]);
    cout<<"Tensor(0): "<<endl;
    infer.tensor_to_cvmat(out[0], CV_32FC1, true);
    cout<<"Tensor(1): "<<endl;
    infer.tensor_to_cvmat(out[1], CV_32FC1, true);
    cout<<"Tensor(2): "<<endl;
    infer.tensor_to_cvmat(out[2], CV_32FC1, true);
    cout<<"Tensor(3): "<<endl;
    infer.tensor_to_cvmat(out[3], CV_32FC1, true);
    return 0;
}
