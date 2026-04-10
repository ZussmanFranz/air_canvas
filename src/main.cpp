#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

int main(int, char**){
    VideoCapture cap(0);
    if(!cap.isOpened()) return -1;
 
    Mat frame;
    namedWindow("capture", WINDOW_NORMAL);
    while(true)
    {
        cap >> frame;

        // We flip the image to better control our drawing.
        // 1 is a flip flag. Positive value means "flip around Y-axis".
        flip(frame, frame, 1);
        
        imshow("capture", frame);
        if(waitKey(30) >= 0) break;
    }
    return 0;
}