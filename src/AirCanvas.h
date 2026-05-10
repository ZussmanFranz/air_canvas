#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// --- Enums ---
enum RADIAL_STATE {
    WAIT_SPAWN,
    WAIT_CHOICE
};

// --- Data Structures ---
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
    cv::Scalar background_color = cv::Scalar(255, 255, 255);

    bool capture_mouse = false;
    cv::Point mouse_pos = cv::Point(-1,-1);
    cv::Point calibration_target = cv::Point(-1, -1);
};

struct CursorData {
    const int SIZE = 4;
    const int THICKNESS = 2;
    cv::Point pos = cv::Point(-1, -1);
    cv::Point prev_pos = cv::Point(-1, -1);
    bool is_moving = true;
    double stop_time = -1;
    cv::Scalar color;
};

struct RadialMenu {
    const int PROGRESS_SIZE = 8;
    const int PROGRESS_THICKNESS = 2;
    const int RADIAL_MOVE_TRESHOLD = 2;
    const int RADIAL_SIZE = 60;
    
    double spawn_ticks = 0;
    double progress_headstart_ticks = 0;

    RADIAL_STATE state = WAIT_SPAWN;
    cv::Point center;
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255,0,0), 
        cv::Scalar(0,255,0), 
        cv::Scalar(0,0,255), 
        cv::Scalar(0,255,255), 
        cv::Scalar(255,0,255)
    };
};

// --- Class ---

class AirCanvas {
private:
    AppConfig config;
    CursorData cursor;
    RadialMenu menu;

    cv::VideoCapture cap;
    cv::Mat frame, display_frame, hsv_frame, mask, background;
    cv::Mat canvas, radial_canvas;
    cv::Mat kernel;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    bool is_running = true;
    bool capture_drawing = false;
    int frames_lost = 0;
    const int MAX_FRAMES_LOST = 10;
    const float SMOOTHING = 0.35f;

    bool show_mask = true;
    bool show_hsv_boundaries = true;

public:
    AirCanvas();
    ~AirCanvas();

    void run();

    void set_mask_visibility(bool visible);
    void set_hsv_boundaries_visibility(bool visible);

private:
    void processVision();
    void updateLogic();
    void updateRadialMenu();
    void renderUI();
    void handleInput();

    // utilities
    
    static void staticMouseCallback(int event, int x, int y, int flag, void* param);
    static void staticTrackbarCallback(int pos, void* userdata);

    void handleMouse(int event, int x, int y);
    void setupWindows();
    void setupTrackbarWindow();
    void autoCalibration(int x, int y, int kernel_size, int tolerance);
    void overlayMats(const cv::Mat& top_layer, const cv::Mat& bottom_layer, cv::Mat& output_mat);
    bool checkMove(cv::Point prev_pos, cv::Point current_pos, int threshold_val);
    void drawColorWheel(cv::Mat& tgt_canvas, const cv::Point& center, int radius, int thickness, const std::vector<cv::Scalar>& colors);
    cv::Scalar determineColor(const cv::Point& wheel_center, const cv::Point& cursor_position, const std::vector<cv::Scalar>& colors);
    void saveConfig();
    void loadConfig();
};