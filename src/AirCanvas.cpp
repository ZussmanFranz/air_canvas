#include "AirCanvas.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <toml++/toml.hpp>

using namespace cv;
using namespace std;



AirCanvas::AirCanvas() {
    loadConfig();

    cap.open(0, CAP_V4L2);
    if(!cap.isOpened()) {
        cerr << "Error: Camera missing" << endl;
        exit(-1);
    }

    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    // timers initialization
    menu.spawn_ticks = 2.0 * getTickFrequency();
    menu.progress_headstart_ticks = 1.0 * getTickFrequency();
    if (!menu.colors.empty()) cursor.color = menu.colors[0];

    setupWindows();

    // test frame
    cap >> frame;
    if (!frame.empty()) {
        canvas = Mat::zeros(frame.size(), CV_8UC3);
        radial_canvas = Mat::zeros(frame.size(), CV_8UC3);
        background = Mat(frame.size(), CV_8UC3, config.background_color);
    }
    
    kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
}

AirCanvas::~AirCanvas() {
    saveConfig();
    cap.release();
    destroyAllWindows();
}

// main method
void AirCanvas::run() {
    while (is_running) {
        cap >> frame;
        if (frame.empty()) {
            cout << "Warning. Missing frame" << endl;
            continue;
        }

        flip(frame, frame, 1);

        processVision();  // computer vision
        updateLogic();    // tracking
        renderUI();       // drawing
        handleInput();    // input
    }
}

// --- Computer Vision ---
void AirCanvas::processVision() {
    medianBlur(frame, frame, 5);
    cvtColor(frame, hsv_frame, COLOR_BGR2HSV);

    // autocalibration
    if (config.calibration_target.x != -1 && config.calibration_target.y != -1) {
        autoCalibration(config.calibration_target.x, config.calibration_target.y, 5, 10);
        config.calibration_target = Point(-1, -1);
    }

    Scalar lower_bound(config.hue_min, config.sat_min, config.val_min);
    Scalar upper_bound(config.hue_max, config.sat_max, config.val_max);
    
    inRange(hsv_frame, lower_bound, upper_bound, mask);
    erode(mask, mask, kernel, Point(-1,-1), 1);
    dilate(mask, mask, kernel, Point(-1,-1), 3);

    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
}

// --- Tracking ---
void AirCanvas::updateLogic() {
    bool cursor_found = false;

    if (config.capture_mouse || !contours.empty()) {
        double max_area = 0;
        int max_contour_id = -1;

        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                max_contour_id = i;
            }
        }

        if (config.capture_mouse || (max_contour_id != -1 && max_area > config.CONTOUR_AREA_TRESHOLD)) {
            Moments m;
            if (!config.capture_mouse) m = moments(contours[max_contour_id]);

            if (config.capture_mouse || m.m00 > 0) {
                cursor_found = true;
                frames_lost = 0;

                int target_x = config.capture_mouse ? config.mouse_pos.x : m.m10 / m.m00;
                int target_y = config.capture_mouse ? config.mouse_pos.y : m.m01 / m.m00;

                if (cursor.prev_pos.x == -1) {
                    cursor.pos = Point(target_x, target_y);
                    cursor.prev_pos = cursor.pos;
                } else {
                    cursor.pos.x = cursor.prev_pos.x + SMOOTHING * (target_x - cursor.prev_pos.x);
                    cursor.pos.y = cursor.prev_pos.y + SMOOTHING * (target_y - cursor.prev_pos.y);

                    if (capture_drawing) {
                        line(canvas, cursor.prev_pos, cursor.pos, cursor.color, 5);
                    }

                    updateRadialMenu();
                    
                    cursor.prev_pos = cursor.pos;
                }
            }
        }
    }

    if (!cursor_found) {
        frames_lost++;
        if (frames_lost > MAX_FRAMES_LOST) cursor.prev_pos = Point(-1, -1);
    }
}

void AirCanvas::updateRadialMenu() {
    if (!checkMove(cursor.prev_pos, cursor.pos, menu.RADIAL_MOVE_TRESHOLD)) {
        if (cursor.is_moving) {
            cursor.is_moving = false;
            cursor.stop_time = getTickCount();
        }

        if (menu.state == WAIT_SPAWN && ((double)(getTickCount() - cursor.stop_time) > menu.spawn_ticks)) {
            cursor.stop_time = getTickCount();
            menu.center = cursor.pos;
            drawColorWheel(radial_canvas, menu.center, menu.RADIAL_SIZE, menu.RADIAL_SIZE / 3, menu.colors);
            menu.state = WAIT_CHOICE;
        } 
    } else {
        cursor.is_moving = true;
        cursor.stop_time = -1;
    }

    if (menu.state == WAIT_CHOICE && checkMove(cursor.pos, menu.center, menu.RADIAL_SIZE)) {
        radial_canvas = Scalar(0,0,0);
        cursor.color = determineColor(menu.center, cursor.pos, menu.colors);
        menu.state = WAIT_SPAWN;
    }
}

// --- Drawing ---
void AirCanvas::renderUI() {
    if (config.fill_background) background.copyTo(display_frame);
    else frame.copyTo(display_frame);

    if (cursor.prev_pos.x != -1 && cursor.prev_pos.y != -1) {
        circle(display_frame, cursor.prev_pos, cursor.SIZE, cursor.color, cursor.THICKNESS);
        
        if (!cursor.is_moving && menu.state == WAIT_SPAWN) {
            double progress = max(((double)(getTickCount() - cursor.stop_time - menu.progress_headstart_ticks)) / (double)(menu.spawn_ticks - menu.progress_headstart_ticks), 0.0);
            if (progress >= 0.05) {
                ellipse(display_frame, cursor.prev_pos, Size(menu.PROGRESS_SIZE, menu.PROGRESS_SIZE), 0, 0, (double)(360 * progress), cursor.color, menu.PROGRESS_THICKNESS);
            }
        }
    }

    overlayMats(canvas, display_frame, display_frame);
    overlayMats(radial_canvas, display_frame, display_frame);

    imshow("mask", mask);
    imshow("capture", display_frame);
}

// --- Input handling ---
void AirCanvas::handleInput() {
    int key = waitKey(30);
    if (key == 27) is_running = false; // Esc
    if (key == 32) { // Space
        cursor.prev_pos = Point(-1, -1);
        capture_drawing = !capture_drawing;
    }
    if (key == 98) config.fill_background = !config.fill_background; // b
    if (key == 109) { // m
        config.capture_mouse = !config.capture_mouse;
        if (!config.capture_mouse) config.mouse_pos = Point(-1, -1);
    }
}

// --- Utilities ---

// static wrapper function for mouse callback
void AirCanvas::staticMouseCallback(int event, int x, int y, int flag, void* param) {
    AirCanvas* app = static_cast<AirCanvas*>(param);
    app->handleMouse(event, x, y);
}

void AirCanvas::handleMouse(int event, int x, int y) {
    if (event == EVENT_MOUSEMOVE && config.capture_mouse) {
        config.mouse_pos = Point(x, y);
    } else if (!config.capture_mouse && event == EVENT_LBUTTONDOWN) {
        config.calibration_target = Point(x, y);
    }
}

void AirCanvas::setupWindows() {
    namedWindow("HSV boundaries", WINDOW_NORMAL);
    namedWindow("mask", WINDOW_NORMAL);
    namedWindow("capture", WINDOW_NORMAL);

    // we are passing "this" so that setMouseCallback can use class method
    setMouseCallback("capture", AirCanvas::staticMouseCallback, this);

    createTrackbar("Hue min", "HSV boundaries", &config.hue_min, config.HUE_SLIDER_MAX, nullptr);
    createTrackbar("Hue max", "HSV boundaries", &config.hue_max, config.HUE_SLIDER_MAX, nullptr);
    createTrackbar("Satur. min", "HSV boundaries", &config.sat_min, config.SAT_SLIDER_MAX, nullptr);
    createTrackbar("Satur. max", "HSV boundaries", &config.sat_max, config.SAT_SLIDER_MAX, nullptr);
    createTrackbar("Value min", "HSV boundaries", &config.val_min, config.VAL_SLIDER_MAX, nullptr);
    createTrackbar("Value max", "HSV boundaries", &config.val_max, config.VAL_SLIDER_MAX, nullptr);
}

void AirCanvas::autoCalibration(int x, int y, int kernel_size, int tolerance) {
    int half_k = kernel_size / 2;
    int rx = std::max(0, x - half_k);
    int ry = std::max(0, y - half_k);
    int rw = std::min(hsv_frame.cols - rx, kernel_size);
    int rh = std::min(hsv_frame.rows - ry, kernel_size);

    Mat patch = hsv_frame(Rect(rx, ry, rw, rh));
    Scalar mean_hsv = mean(patch);

    config.hue_min = std::clamp((int)mean_hsv[0] - tolerance, 0, config.HUE_SLIDER_MAX);
    config.hue_max = std::clamp((int)mean_hsv[0] + tolerance, 0, config.HUE_SLIDER_MAX);
    config.sat_min = std::clamp((int)mean_hsv[1] - tolerance * 3, 0, config.SAT_SLIDER_MAX);
    config.sat_max = config.SAT_SLIDER_MAX;
    config.val_min = std::clamp((int)mean_hsv[2] - tolerance * 3, 0, config.VAL_SLIDER_MAX);
    config.val_max = config.VAL_SLIDER_MAX;

    setTrackbarPos("Hue min", "HSV boundaries", config.hue_min);
    setTrackbarPos("Hue max", "HSV boundaries", config.hue_max);
    setTrackbarPos("Satur. min", "HSV boundaries", config.sat_min);
    setTrackbarPos("Satur. max", "HSV boundaries", config.sat_max);
    setTrackbarPos("Value min", "HSV boundaries", config.val_min);
    setTrackbarPos("Value max", "HSV boundaries", config.val_max);

    saveConfig();
}

void AirCanvas::overlayMats(const Mat& top_layer, const Mat& bottom_layer, Mat& output_mat) {
    Mat top_mask;
    output_mat = bottom_layer.clone();
    cvtColor(top_layer, top_mask, COLOR_BGR2GRAY);
    threshold(top_mask, top_mask, 0, 255, THRESH_BINARY);
    top_layer.copyTo(output_mat, top_mask);
}

bool AirCanvas::checkMove(Point prev_pos, Point current_pos, int threshold_val) {
    return norm(current_pos - prev_pos) > threshold_val;
}

void AirCanvas::drawColorWheel(Mat& tgt_canvas, const Point& center, int radius, int thickness, const vector<Scalar>& colors) {
    int ncolors = colors.size();
    for (int i = 0; i < ncolors; ++i) {
        int start_angle = 0 + i * 360 / ncolors;
        int end_angle = start_angle + 360 / ncolors;
        ellipse(tgt_canvas, center, Size(radius, radius), 0, start_angle, end_angle, colors[i], thickness);
    }
}

Scalar AirCanvas::determineColor(const Point& wheel_center, const Point& cursor_position, const vector<Scalar>& colors) {
    double angle_rad = atan2(cursor_position.y - wheel_center.y, cursor_position.x - wheel_center.x);
    int angle = ((int)(angle_rad * 180 / CV_PI) + 360) % 360;
    int color_index = angle / (360 / colors.size());
    return colors[color_index];
}

void AirCanvas::saveConfig() {
    toml::table tbl{
        {"hsv", toml::table{
            {"hue_min", config.hue_min},
            {"hue_max", config.hue_max},
            {"sat_min", config.sat_min},
            {"sat_max", config.sat_max},
            {"val_min", config.val_min},
            {"val_max", config.val_max}
        }}
    };
    std::ofstream file(config.CONFIG_FILENAME);
    if (file.is_open()) file << tbl;
}

void AirCanvas::loadConfig() {
    try {
        auto toml_config = toml::parse_file(config.CONFIG_FILENAME);
        config.hue_min = toml_config["hsv"]["hue_min"].value_or(config.HUE_SLIDER_MAX);
        config.hue_max = toml_config["hsv"]["hue_max"].value_or(config.HUE_SLIDER_MAX);
        config.sat_min = toml_config["hsv"]["sat_min"].value_or(config.SAT_SLIDER_MAX);
        config.sat_max = toml_config["hsv"]["sat_max"].value_or(config.SAT_SLIDER_MAX);
        config.val_min = toml_config["hsv"]["val_min"].value_or(config.VAL_SLIDER_MAX);
        config.val_max = toml_config["hsv"]["val_max"].value_or(config.VAL_SLIDER_MAX);
    } catch (const toml::parse_error& err) {}
}