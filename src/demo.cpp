#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string.h>
#include "Config.h"
#include "utils.cpp"

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
    mold_image(molded_image, myConfig);
    get_anchors(molded_image.rows, molded_image.cols, myConfig);
    cvtColor(image, image, COLOR_RGB2BGR);

    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image);                // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}