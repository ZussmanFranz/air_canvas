#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/viz/types.hpp>

using namespace cv;
using namespace viz;
using namespace std;

// --- Data Structures ---

enum RADIAL_STATE {
    WAIT_SPAWN,
    WAIT_CHOICE
};

RADIAL_STATE radial_state = WAIT_SPAWN;

// --- Global Variables

vector<Color> radial_colors = {Color(255,0,0), Color(0,255,0), Color(0,0,255)};
Color cursor_color = Color(255,0,0);

bool capture_mouse = false;

// --- Constants ---

const int HUE_SLIDER_MAX = 179;
int hue_min_slider = 112;
int hue_max_slider = HUE_SLIDER_MAX;

const int SAT_SLIDER_MAX = 255;
int sat_min_slider = 113;
int sat_max_slider = SAT_SLIDER_MAX;

const int VAL_SLIDER_MAX = 255;
int val_min_slider = 90;
int val_max_slider = VAL_SLIDER_MAX;

const int CONTOUR_AREA_TRESHOLD = 625;

// --- Function Headers ---

void overlay_mats(const Mat& top_layer, const Mat& bottom_layer, Mat& output_mat);
bool check_move(Point prev_pos, Point current_pos, int move_treshold);
void draw_colorwheel(Mat& canvas, const Point& center, int radius, int thickness, const vector<Color>& colors);
Color determine_color(const Point& wheel_center, const Point& cursor_position, const vector<Color>& colors);

void mouse_callback(int  event, int  x, int  y, int  flag, void *param)
{
    if (event == EVENT_MOUSEMOVE && capture_mouse) {
        cout << "(" << x << ", " << y << ")" << endl;
    }
}

int main(int, char**){
    VideoCapture cap(0, CAP_V4L2);
    if(!cap.isOpened()) return -1;

    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));

    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
 
    namedWindow("HSV boundaries", WINDOW_NORMAL);
    namedWindow("mask", WINDOW_NORMAL);
    namedWindow("capture", WINDOW_NORMAL);

    // we are tracking mouse position
    setMouseCallback("capture", mouse_callback);

    createTrackbar("Hue min", "HSV boundaries", &hue_min_slider, HUE_SLIDER_MAX, nullptr);
    createTrackbar("Hue max", "HSV boundaries", &hue_max_slider, HUE_SLIDER_MAX, nullptr);

    createTrackbar("Satur. min", "HSV boundaries", &sat_min_slider, SAT_SLIDER_MAX, nullptr);
    createTrackbar("Satur. max", "HSV boundaries", &sat_max_slider, SAT_SLIDER_MAX, nullptr);

    createTrackbar("Value min", "HSV boundaries", &val_min_slider, VAL_SLIDER_MAX, nullptr);
    createTrackbar("Value max", "HSV boundaries", &val_max_slider, VAL_SLIDER_MAX, nullptr);

    Mat frame, display_frame, hsv_frame, mask;

    Mat canvas, canvas_mask, radial_canvas;

    cap >> frame;
    if (frame.empty()) return -1;

    // we will be drawing on this canvas, but now it is empty (black)
    canvas = Mat::zeros(frame.size(), CV_8UC3);
    radial_canvas = Mat::zeros(frame.size(), CV_8UC3);

    // smoothing logic for flickering cursor
    int prev_x = -1;
    int prev_y = -1;

    int smoothed_x = -1;
    int smoothed_y = -1;

    int frames_lost = 0;
    const int MAX_FRAMES_LOST = 10;  // how long do we remember the position
    const float SMOOTHING = 0.35f;   // smoothing factor (from 0.1 to 1.0)

    int key_pressed = -1;

    bool capture_drawing = false;

    Point radial_center;
    int radial_spawn_time = -1;
    const int RADIAL_LIFETIME_SECONDS = 10;
    const int RADIAL_MOVE_TRESHOLD = 2;
    const int RADIAL_SPAWN_SECONDS = 2;
    double RADIAL_SPAWN_TICKS = RADIAL_SPAWN_SECONDS * getTickFrequency();
    const int RADIAL_SIZE = 60;
    bool is_moving = true;
    double stop_time = -1;



    while(true)
    {
        cap >> frame;
        if (frame.empty()) {
           cout << "WARNING: skipped frame from the camera" << endl;
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
                        smoothed_x = prev_x + SMOOTHING * (target_x - prev_x);
                        smoothed_y = prev_y + SMOOTHING * (target_y - prev_y);

                        if (capture_drawing){
                            // draw a new smoothed line on canvas
                            line(canvas, Point(prev_x, prev_y), Point(smoothed_x, smoothed_y), Scalar(255, 0, 0), 5);
                        }

                        if (!check_move(Point(prev_x, prev_y), Point(smoothed_x, smoothed_y), RADIAL_MOVE_TRESHOLD)){
                            if (is_moving){
                                is_moving = false;
                                stop_time = getTickCount();
                            }

                            if ((radial_state == WAIT_SPAWN) && ((double)(getTickCount() - stop_time) > RADIAL_SPAWN_TICKS)) {
                                // PLACEHOLDER LOGIC
                                stop_time = getTickCount();
                                radial_center = Point(smoothed_x, smoothed_y);

                                draw_colorwheel(radial_canvas, radial_center, RADIAL_SIZE, RADIAL_SIZE / 3, radial_colors);
                                
                                radial_state = WAIT_CHOICE;
                            } 
                        } else {
                            is_moving = true;
                            stop_time = -1;
                        }

                        if (radial_state == WAIT_CHOICE) {
                            // did cursor go out of radial menu?

                            if (check_move(Point(smoothed_x, smoothed_y), radial_center, RADIAL_SIZE)) {
                                // clear radial menu
                                radial_canvas = Scalar(0,0,0);

                                // change color
                                cursor_color = determine_color(radial_center, Point(smoothed_x, smoothed_y), radial_colors);

                                radial_state= WAIT_SPAWN;
                            }
                        }

                        prev_x = smoothed_x;
                        prev_y = smoothed_y;
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
            circle(display_frame, Point(prev_x, prev_y), 10, cursor_color, 2);
        }

        // adding drawing from canvas to display frame

        overlay_mats(canvas, display_frame, display_frame);
        overlay_mats(radial_canvas, display_frame, display_frame);

        imshow("mask", mask);

        imshow("capture", display_frame);
        
        key_pressed = waitKey(30);

        // 27 == Esc, 32 = Space, 109 = m
        if(key_pressed == 27) break;
        if(key_pressed == 32) {
            // reset previous position
            prev_x = -1;
            prev_y = -1;

            // toggle drawing mode
            capture_drawing = !capture_drawing;
        }
        else if (key_pressed == 109) {
            capture_mouse = !capture_mouse;
        }
    }
    return 0;
}


void overlay_mats(const Mat& top_layer, const Mat& bottom_layer, Mat& output_mat) {
    /*
        Overlays mats. Puts non-black pixels of top layer on bottom layer.
    */

    Mat top_mask;

    // copy bottom_layer content to output_mat
    output_mat = bottom_layer.clone();

    // convert top_layer to grayscale
    cvtColor(top_layer, top_mask, COLOR_BGR2GRAY);

    // everything not black becomes white
    threshold(top_mask, top_mask, 0, 255, THRESH_BINARY);

    // invert the mask
    bitwise_not(top_mask, top_mask);

    // crop the overlay area from bottom layer
    bitwise_and(output_mat, output_mat, output_mat, top_mask);

    // add top layer to the bottom layer
    output_mat += top_layer;

    return;
}

bool check_move(Point prev_pos, Point current_pos, int move_treshold){
    return norm(current_pos - prev_pos) > move_treshold;
}

void draw_colorwheel(Mat& canvas, const Point& center, int radius, int thickness, const vector<Color>& colors){
    int ncolors = colors.size();
    int start_angle, end_angle;

    for (int i = 0; i < ncolors; ++i) {
        start_angle = 0 + i * 360 / ncolors;
        end_angle = start_angle + 360 / ncolors;

        ellipse(canvas, center, Size(radius, radius), 0, start_angle, end_angle, colors[i], thickness);
    }

    return;
}

Color determine_color(const Point& wheel_center, const Point& cursor_position, const vector<Color>& colors) {
    double angle_rad = atan2(cursor_position.y - wheel_center.y, cursor_position.x - wheel_center.x);

    // convert to degrees
    int angle = abs((int)(angle_rad * 180 / CV_PI) % 360);

    cout << "Angle: " << angle << endl;

    int color_index = angle / (360 / colors.size());

    cout << "Color index: " << color_index << endl;

    return colors[color_index];
}