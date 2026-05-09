#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <fstream>
#include <algorithm>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/viz/types.hpp>

#include <toml++/toml.hpp>

using namespace cv;
using namespace viz;
using namespace std;

// --- Enums ---
enum RADIAL_STATE {
    WAIT_SPAWN,
    WAIT_CHOICE
};

// --- Global Value Containers ---
struct AppConfig {
    const std::string CONFIG_FILENAME = "config.toml";

    const int HUE_SLIDER_MAX = 179;
    const int SAT_SLIDER_MAX = 255;
    const int VAL_SLIDER_MAX = 255;

    const int CONTOUR_AREA_TRESHOLD = 625;

    int hue_min = HUE_SLIDER_MAX, hue_max = HUE_SLIDER_MAX;
    int sat_min = SAT_SLIDER_MAX, sat_max = SAT_SLIDER_MAX;
    int val_min = VAL_SLIDER_MAX, val_max = VAL_SLIDER_MAX;

    bool fill_background = false;
    Color background_color = Color(255, 255, 255);

    bool capture_mouse = false;
    Point mouse_pos = Point(-1,-1);

    Point calibration_target = Point(-1, -1);
};

struct Cursor {
    const int CURSOR_SIZE = 4;
    const int CURSOR_THICKNESS = 2;

    Point pos = Point(-1, -1);
    Point prev_pos = Point(-1, -1);

    bool is_moving = true;
    double stop_time = -1;

    Color color;
};

struct RadialMenu {
    const int PROGRESS_SIZE = 8;
    const int PROGRESS_THICKNESS = 2;

    const int RADIAL_LIFETIME_SECONDS = 10;
    const int RADIAL_MOVE_TRESHOLD = 2;
    const int RADIAL_SPAWN_SECONDS = 2;
    const int RADIAL_PROGRESS_HEADSTART_SECONDS = 1;
    const int RADIAL_SIZE = 60;

    double spawn_ticks = 0;
    double progress_headstart_ticks = 0;

    RADIAL_STATE state = WAIT_SPAWN;
    Point center;

    vector<Color> colors = {Color(255,0,0), Color(0,255,0), Color(0,0,255), Color(0,255,255), Color(255,0,255)};
};

// --- Global Instances ---
AppConfig app_config;
Cursor cursor;
RadialMenu radial_menu;

// --- Function Headers ---
void overlay_mats(const Mat& top_layer, const Mat& bottom_layer, Mat& output_mat);
bool check_move(Point prev_pos, Point current_pos, int move_treshold);
void draw_colorwheel(Mat& canvas, const Point& center, int radius, int thickness, const vector<Color>& colors);
Color determine_color(const Point& wheel_center, const Point& cursor_position, const vector<Color>& colors);
void mouse_callback(int event, int x, int y, int flag, void *param);
void auto_calibration(const Mat& hsv_frame, int x, int y, int kernel_size, int tolerance);
void save_config(const std::string& filename);
void load_config(const std::string& filename);

// --- Main ---
int main(int, char**){
    VideoCapture cap(0, CAP_V4L2);
    if(!cap.isOpened()) return -1;

    radial_menu.spawn_ticks = radial_menu.RADIAL_SPAWN_SECONDS * getTickFrequency();
    radial_menu.progress_headstart_ticks = radial_menu.RADIAL_PROGRESS_HEADSTART_SECONDS * getTickFrequency();

    if (!radial_menu.colors.empty()) {
        cursor.color = radial_menu.colors[0];
    }

    load_config(app_config.CONFIG_FILENAME);

    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
 
    namedWindow("HSV boundaries", WINDOW_NORMAL);
    namedWindow("mask", WINDOW_NORMAL);
    namedWindow("capture", WINDOW_NORMAL);

    setMouseCallback("capture", mouse_callback);

    createTrackbar("Hue min", "HSV boundaries", &app_config.hue_min, app_config.HUE_SLIDER_MAX, nullptr);
    createTrackbar("Hue max", "HSV boundaries", &app_config.hue_max, app_config.HUE_SLIDER_MAX, nullptr);
    createTrackbar("Satur. min", "HSV boundaries", &app_config.sat_min, app_config.SAT_SLIDER_MAX, nullptr);
    createTrackbar("Satur. max", "HSV boundaries", &app_config.sat_max, app_config.SAT_SLIDER_MAX, nullptr);
    createTrackbar("Value min", "HSV boundaries", &app_config.val_min, app_config.VAL_SLIDER_MAX, nullptr);
    createTrackbar("Value max", "HSV boundaries", &app_config.val_max, app_config.VAL_SLIDER_MAX, nullptr);

    Mat frame, display_frame, hsv_frame, mask, background;
    Mat canvas, canvas_mask, radial_canvas;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    cap >> frame;
    if (frame.empty()) return -1;

    canvas = Mat::zeros(frame.size(), CV_8UC3);
    radial_canvas = Mat::zeros(frame.size(), CV_8UC3);
    background = Mat(frame.size(), CV_8UC3, app_config.background_color);

    int frames_lost = 0;
    const int MAX_FRAMES_LOST = 10;
    const float SMOOTHING = 0.35f;
    int key_pressed = -1;
    bool capture_drawing = false;

    while(true)
    {
        cap >> frame;
        if (frame.empty()) {
           cout << "WARNING: skipped frame from the camera" << endl;
           continue;
        }

        flip(frame, frame, 1);

        if (app_config.fill_background) display_frame = background.clone();
        else display_frame = frame.clone();

        medianBlur(frame, frame, 5);
        cvtColor(frame, hsv_frame, COLOR_BGR2HSV);

        if (app_config.calibration_target.x != -1 && app_config.calibration_target.y != -1) {
            auto_calibration(hsv_frame, app_config.calibration_target.x, app_config.calibration_target.y, 5, 10);
            app_config.calibration_target = Point(-1, -1);
        }

        Scalar lower_bound(app_config.hue_min, app_config.sat_min, app_config.val_min);
        Scalar upper_bound(app_config.hue_max, app_config.sat_max, app_config.val_max);
        
        inRange(hsv_frame, lower_bound, upper_bound, mask);        // erode - remove "weak" parts of mask
        // erode - remove "weak" parts of mask
        erode(mask, mask, kernel, Point(-1,-1), 1);
        // dilate - make everything that is left stronger and more consistent
        dilate(mask, mask, kernel, Point(-1,-1), 3);

        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        bool cursor_found = false;

        if (app_config.capture_mouse || !contours.empty()){
            double max_area = 0;
            int max_contour_id = -1;

            for (size_t i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > max_area) {
                    max_area = area;
                    max_contour_id = i;
                }
            }

            if (app_config.capture_mouse || (max_contour_id != -1 && max_area > app_config.CONTOUR_AREA_TRESHOLD)) {
                Moments m;
                if (!app_config.capture_mouse) {
                    m = moments(contours[max_contour_id]);
                }

                // calculate it's central coordinates
                if (app_config.capture_mouse || m.m00 > 0) {
                    cursor_found = true;
                    frames_lost = 0;

                    int target_x, target_y;
                    
                    if (app_config.capture_mouse){
                        target_x = app_config.mouse_pos.x;
                        target_y = app_config.mouse_pos.y;
                    } else {
                        target_x = m.m10 / m.m00;
                        target_y = m.m01 / m.m00;
                    }

                    if (cursor.prev_pos.x == -1) {
                        // It's first time when we see the cursor, just teleport there
                        cursor.pos.x = target_x;
                        cursor.pos.y = target_y;
                        cursor.prev_pos = cursor.pos;
                    } else {
                        // smoothing between current and previous position
                        cursor.pos.x = cursor.prev_pos.x + SMOOTHING * (target_x - cursor.prev_pos.x);
                        cursor.pos.y = cursor.prev_pos.y + SMOOTHING * (target_y - cursor.prev_pos.y);

                        if (capture_drawing){
                            // draw a new smoothed line on canvas
                            line(canvas, cursor.prev_pos, cursor.pos, cursor.color, 5);
                        }

                        if (!check_move(cursor.prev_pos, cursor.pos, radial_menu.RADIAL_MOVE_TRESHOLD)){
                            if (cursor.is_moving){
                                cursor.is_moving = false;
                                cursor.stop_time = getTickCount();
                            }

                            if ((radial_menu.state == WAIT_SPAWN) && ((double)(getTickCount() - cursor.stop_time) > radial_menu.spawn_ticks)) {
                                cursor.stop_time = getTickCount();
                                radial_menu.center = cursor.pos;

                                draw_colorwheel(radial_canvas, radial_menu.center, radial_menu.RADIAL_SIZE, radial_menu.RADIAL_SIZE / 3, radial_menu.colors);
                                radial_menu.state = WAIT_CHOICE;
                            } 
                        } else {
                            cursor.is_moving = true;
                            cursor.stop_time = -1;
                        }

                        if (radial_menu.state == WAIT_CHOICE) {
                            if (check_move(cursor.pos, radial_menu.center, radial_menu.RADIAL_SIZE)) {
                                radial_canvas = Scalar(0,0,0);
                                cursor.color = determine_color(radial_menu.center, cursor.pos, radial_menu.colors);
                                radial_menu.state = WAIT_SPAWN;
                            }
                        }

                        cursor.prev_pos = cursor.pos;
                    }
                }
            }
        }

        if (!cursor_found){
            frames_lost++;
            if (frames_lost > MAX_FRAMES_LOST){
                cursor.prev_pos = Point(-1,-1);
            }
        }

        if (cursor.prev_pos.x != -1 && cursor.prev_pos.y != -1){
            circle(display_frame, cursor.prev_pos, cursor.CURSOR_SIZE, cursor.color, cursor.CURSOR_THICKNESS);
            
            if (!cursor.is_moving && radial_menu.state == WAIT_SPAWN){
                double progress = max(((double)(getTickCount() - cursor.stop_time - radial_menu.progress_headstart_ticks)) / (double)(radial_menu.spawn_ticks - radial_menu.progress_headstart_ticks), 0.0);

                if (progress >= 0.05) {
                    ellipse(display_frame, cursor.prev_pos, Size(radial_menu.PROGRESS_SIZE, radial_menu.PROGRESS_SIZE), 0, 0, (double)(360 * progress), cursor.color, radial_menu.PROGRESS_THICKNESS);
                }
            }
        }

        overlay_mats(canvas, display_frame, display_frame);
        overlay_mats(radial_canvas, display_frame, display_frame);

        imshow("mask", mask);
        imshow("capture", display_frame);
        
        key_pressed = waitKey(30);

        if(key_pressed == 27) break;
        if(key_pressed == 32) {
            cursor.prev_pos.x = -1;
            cursor.prev_pos.y = -1;
            capture_drawing = !capture_drawing;
        }
        else if (key_pressed == 98) {
            app_config.fill_background = !app_config.fill_background;
        }
        else if (key_pressed == 109) {
            app_config.capture_mouse = !app_config.capture_mouse;
            if (!app_config.capture_mouse) {
                app_config.mouse_pos.x = -1;
                app_config.mouse_pos.y = -1;
            }
        }
    }

    save_config(app_config.CONFIG_FILENAME);
    return 0;
}

// --- Implementations ---

void overlay_mats(const Mat& top_layer, const Mat& bottom_layer, Mat& output_mat) {
    Mat top_mask;
    output_mat = bottom_layer.clone();
    cvtColor(top_layer, top_mask, COLOR_BGR2GRAY);
    threshold(top_mask, top_mask, 0, 255, THRESH_BINARY);
    top_layer.copyTo(output_mat, top_mask);
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
}

Color determine_color(const Point& wheel_center, const Point& cursor_position, const vector<Color>& colors) {
    double angle_rad = atan2(cursor_position.y - wheel_center.y, cursor_position.x - wheel_center.x);
    int angle = ((int)(angle_rad * 180 / CV_PI) + 360) % 360;
    int color_index = angle / (360 / colors.size());
    return colors[color_index];
}

void mouse_callback(int event, int x, int y, int flag, void *param) {
    if (event == EVENT_MOUSEMOVE && app_config.capture_mouse) {
        app_config.mouse_pos.x = x;
        app_config.mouse_pos.y = y;
    } else if (!app_config.capture_mouse && event == EVENT_LBUTTONDOWN) {
        app_config.calibration_target.x = x;
        app_config.calibration_target.y = y;
    }
}

void auto_calibration(const Mat& hsv_frame, int x, int y, int kernel_size, int tolerance) {
    int half_k = kernel_size / 2;
    int rx = std::max(0, x - half_k);
    int ry = std::max(0, y - half_k);
    int rw = std::min(hsv_frame.cols - rx, kernel_size);
    int rh = std::min(hsv_frame.rows - ry, kernel_size);

    Rect roi(rx, ry, rw, rh);
    Mat patch = hsv_frame(roi);
    Scalar mean_hsv = mean(patch);

    app_config.hue_min = std::clamp((int)mean_hsv[0] - tolerance, 0, app_config.HUE_SLIDER_MAX);
    app_config.hue_max = std::clamp((int)mean_hsv[0] + tolerance, 0, app_config.HUE_SLIDER_MAX);
    app_config.sat_min = std::clamp((int)mean_hsv[1] - tolerance * 3, 0, app_config.SAT_SLIDER_MAX);
    app_config.sat_max = app_config.SAT_SLIDER_MAX;
    app_config.val_min = std::clamp((int)mean_hsv[2] - tolerance * 3, 0, app_config.VAL_SLIDER_MAX);
    app_config.val_max = app_config.VAL_SLIDER_MAX;

    setTrackbarPos("Hue min", "HSV boundaries", app_config.hue_min);
    setTrackbarPos("Hue max", "HSV boundaries", app_config.hue_max);
    setTrackbarPos("Satur. min", "HSV boundaries", app_config.sat_min);
    setTrackbarPos("Satur. max", "HSV boundaries", app_config.sat_max);
    setTrackbarPos("Value min", "HSV boundaries", app_config.val_min);
    setTrackbarPos("Value max", "HSV boundaries", app_config.val_max);

    save_config(app_config.CONFIG_FILENAME);
}

void save_config(const std::string& filename) {
    toml::table tbl{
        {"hsv", toml::table{
            {"hue_min", app_config.hue_min},
            {"hue_max", app_config.hue_max},
            {"sat_min", app_config.sat_min},
            {"sat_max", app_config.sat_max},
            {"val_min", app_config.val_min},
            {"val_max", app_config.val_max}
        }}
    };

    std::ofstream file(filename);
    if (file.is_open()) {
        file << tbl;
        cout << "Successfully saved config to the file " << filename << endl;
    } else {
        cerr << "Error: Could not open config file" << endl;
    }
}

void load_config(const std::string& filename) {
    try {
        auto config = toml::parse_file(filename);

        app_config.hue_min = config["hsv"]["hue_min"].value_or(app_config.HUE_SLIDER_MAX);
        app_config.hue_max = config["hsv"]["hue_max"].value_or(app_config.HUE_SLIDER_MAX);
        app_config.sat_min = config["hsv"]["sat_min"].value_or(app_config.SAT_SLIDER_MAX);
        app_config.sat_max = config["hsv"]["sat_max"].value_or(app_config.SAT_SLIDER_MAX);
        app_config.val_min = config["hsv"]["val_min"].value_or(app_config.VAL_SLIDER_MAX);
        app_config.val_max = config["hsv"]["val_max"].value_or(app_config.VAL_SLIDER_MAX);

        cout << "Config is loaded from " << filename << endl;
    } catch (const toml::parse_error& err) {
        cout << "Could not find config file. Using default values." << endl;
    }
}