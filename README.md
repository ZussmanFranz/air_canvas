# Air Canvas CV

This project's goal is to get familiar with OpenCV C++ library in a fun and simple way.

## Project structure

```text
air_canvas_cv/
├── CMakeLists.txt          # Project build file
├── README.md
├── .gitignore
├── src/
│   ├── main.cpp            # Main loop
│   ├── AirCanvas.cpp       # Main features class
│   └── AirCanvas.h
├── build.sh                # Shell script for building project
└── build/                  # Build output
    └── AirCanvas           # Executable file (after build)
```

## Setup

1) Install **OpenCV library** (For example `sudo dnf install opencv opencv-devel` on Fedora Linux).
1) `./build.sh` to **build** the app.
1) `./build/AirCanvas` to **run** the app.
