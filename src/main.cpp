#include <iostream>
#include <vector>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

const int HUE_SLIDER_MAX = 179;
int hue_min_slider = 114;
int hue_max_slider = HUE_SLIDER_MAX;

const int SAT_SLIDER_MAX = 255;
int sat_min_slider = 52;
int sat_max_slider = SAT_SLIDER_MAX;

const int VAL_SLIDER_MAX = 255;
int val_min_slider = 169;
int val_max_slider = VAL_SLIDER_MAX;

int main(int, char**){
    VideoCapture cap(0);
    if(!cap.isOpened()) return -1;
 
    namedWindow("capture", WINDOW_NORMAL);
    namedWindow("HSV boundaries", WINDOW_NORMAL);
    namedWindow("mask", WINDOW_NORMAL);


    createTrackbar("Hue min", "HSV boundaries", &hue_min_slider, HUE_SLIDER_MAX, nullptr);
    createTrackbar("Hue max", "HSV boundaries", &hue_max_slider, HUE_SLIDER_MAX, nullptr);

    createTrackbar("Satur. min", "HSV boundaries", &sat_min_slider, SAT_SLIDER_MAX, nullptr);
    createTrackbar("Satur. max", "HSV boundaries", &sat_max_slider, SAT_SLIDER_MAX, nullptr);

    createTrackbar("Value min", "HSV boundaries", &val_min_slider, VAL_SLIDER_MAX, nullptr);
    createTrackbar("Value max", "HSV boundaries", &val_max_slider, VAL_SLIDER_MAX, nullptr);

    Mat frame, hsv_frame, mask;

    while(true)
    {
        cap >> frame;

        // flip the image to better control our drawing.
        // 1 is a flip flag. Positive value means "flip around Y-axis".
        flip(frame, frame, 1);

        // convert RGB image to HSV
        cvtColor(frame, hsv_frame, COLOR_BGR2HSV);

        Scalar lower_bound(hue_min_slider, sat_min_slider, val_min_slider);
        Scalar upper_bound(hue_max_slider, sat_max_slider, val_max_slider);
        
        // create a mask of white pixels fitting in boundaries
        inRange(hsv_frame, lower_bound, upper_bound, mask);


        // kernel for erode and dilate iterations
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));

        // erode - remove "weak" parts of mask
        erode(mask, mask, kernel, Point(-1,-1), 1);

        // dilate - make everything that is left stronger and more consistent
        dilate(mask, mask, kernel, Point(-1,-1), 3);  // more iterations to make pointer stronger

        imshow("mask", mask);

        imshow("capture", frame);
        
        if(waitKey(30) >= 0) break;
    }
    return 0;
}