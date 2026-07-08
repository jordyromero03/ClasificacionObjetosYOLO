/**
 * Benchmark CPU-only para comparación — Parte 1C
 * Mismas operaciones que main.cpp pero sin CUDA
 */

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>

using namespace std;
using namespace cv;
namespace ch = chrono;

int main(int argc, char** argv) {
    string source = (argc > 1) ? argv[1] : "0";
    VideoCapture cap(source == "0" ? 0 : stoi(source));

    Mat frame;
    cap >> frame;
    cap.release();

    if (frame.empty()) {
        cerr << "No se pudo capturar frame\n";
        return 1;
    }
    resize(frame, frame, Size(640, 480));

    vector<double> times;
    int n_runs = 100;

    for (int i = 0; i < n_runs; i++) {
        auto t0 = ch::high_resolution_clock::now();

        Mat gray, blurred, morph, edges, eq;
        GaussianBlur(frame, blurred, Size(5,5), 1.5);
        cvtColor(blurred, gray, COLOR_BGR2GRAY);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));
        erode(gray, morph, kernel);
        dilate(morph, morph, kernel);
        Canny(morph, edges, 50, 150);
        equalizeHist(gray, eq);

        auto t1 = ch::high_resolution_clock::now();
        times.push_back(ch::duration<double, milli>(t1-t0).count());
    }

    double avg = accumulate(times.begin(), times.end(), 0.0) / times.size();
    cout << "CPU avg: " << avg << " ms | FPS: " << 1000.0/avg << "\n";
    return 0;
}
