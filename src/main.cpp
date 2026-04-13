#include <iostream>
#include <vector>
#include <memory>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

const int HUE_SLIDER_MAX = 179;
int hue_min_slider = 112;
int hue_max_slider = HUE_SLIDER_MAX;

const int SAT_SLIDER_MAX = 255;
int sat_min_slider = 113;
int sat_max_slider = SAT_SLIDER_MAX;

const int VAL_SLIDER_MAX = 255;
int val_min_slider = 90;
int val_max_slider = VAL_SLIDER_MAX;

const int CONTOUR_AREA_TRESHOLD = 200;

int main(int, char**){
    VideoCapture cap(0, CAP_V4L2);
    if(!cap.isOpened()) return -1;

    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));

    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
 
    namedWindow("HSV boundaries", WINDOW_NORMAL);
    namedWindow("mask", WINDOW_NORMAL);
    namedWindow("capture", WINDOW_NORMAL);


    createTrackbar("Hue min", "HSV boundaries", &hue_min_slider, HUE_SLIDER_MAX, nullptr);
    createTrackbar("Hue max", "HSV boundaries", &hue_max_slider, HUE_SLIDER_MAX, nullptr);

    createTrackbar("Satur. min", "HSV boundaries", &sat_min_slider, SAT_SLIDER_MAX, nullptr);
    createTrackbar("Satur. max", "HSV boundaries", &sat_max_slider, SAT_SLIDER_MAX, nullptr);

    createTrackbar("Value min", "HSV boundaries", &val_min_slider, VAL_SLIDER_MAX, nullptr);
    createTrackbar("Value max", "HSV boundaries", &val_max_slider, VAL_SLIDER_MAX, nullptr);

    Mat frame, display_frame, hsv_frame, mask;

    // smoothing logic for flickering "cursor"
    int prev_x = -1;
    int prev_y = -1;
    int frames_lost = 0;
    const int MAX_FRAMES_LOST = 10;  // how long do we remember the position
    const float SMOOTHING = 0.35f;   // smoothing factor (from 0.1 to 1.0)

    while(true)
    {
        cap >> frame;
        if (frame.empty()) {
           std::cout << "WARNING: skipped frame from the camera" << std::endl;
           continue;
        }

        // flip the image to better control our drawing.
        // 1 is a flip flag. Positive value means "flip around Y-axis".
        flip(frame, frame, 1);

        display_frame = frame.clone();

        medianBlur(frame, frame, 5);    // soft blur to reduce noise and keep edges

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

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;

        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        bool cursor_found = false;

        if (!contours.empty()){

            double max_area = 0;
            int max_contour_id = -1;

            // find a contour with maximum area
            for (size_t i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > max_area) {
                    max_area = area;
                    max_contour_id = i;
                }
            }

            if (max_contour_id != -1 && max_area > CONTOUR_AREA_TRESHOLD) {
                Moments m = moments(contours[max_contour_id]);

                // calculate it's central coordinates
                if (m.m00 > 0) {
                    cursor_found = true;
                    frames_lost = 0;

                    int target_x = m.m10 / m.m00;
                    int target_y = m.m01 / m.m00;

                    if (prev_x == -1) {
                        // It's first time when we see the cursor, just teleport there
                        prev_x = target_x;
                        prev_y = target_y;
                    } else {
                        // smoothing between current and previous position
                        prev_x = prev_x + SMOOTHING * (target_x - prev_x);
                        prev_y = prev_y + SMOOTHING * (target_y - prev_y);
                    }
                }
            }
        }

        if (!cursor_found){
            frames_lost++;
            if (frames_lost > MAX_FRAMES_LOST){
                prev_x = -1;
                prev_y = -1;
            }
        }

        if (prev_x != -1 && prev_y != -1){
            circle(display_frame, Point(prev_x, prev_y), 10, Scalar(255, 0, 0), 2);
        }

        imshow("mask", mask);

        imshow("capture", display_frame);
        
        if(waitKey(30) >= 0) break;
    }
    return 0;
}