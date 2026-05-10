#include "AirCanvas.h"
#include <iostream>

int main(int argc, char** argv) {
    try {
        AirCanvas app;
        app.set_mask_visibility(false);
        app.set_hsv_boundaries_visibility(false);
        app.run();
    } 
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}