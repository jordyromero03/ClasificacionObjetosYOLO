/**
 * Práctica 4-1C: Preprocesamiento de Imágenes CPU vs GPU
 * Universidad Politécnica Salesiana — Visión por Computador
 * Estudiante: Jordy Romero
 *
 * Pipeline GPU-only con OpenCV CUDA:
 *   1. Filtro Gaussiano
 *   2. Operaciones morfológicas (Erosión / Dilatación)
 *   3. Detección de bordes (Canny)
 *   4. Ecualización de histograma
 */

#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>

using namespace std;
using namespace cv;
namespace ch = chrono;

// ── Utilidades de tiempo ──────────────────────────────────────────────────────
struct Timer {
    ch::high_resolution_clock::time_point t0;
    void start() { t0 = ch::high_resolution_clock::now(); }
    double ms() {
        auto t1 = ch::high_resolution_clock::now();
        return ch::duration<double, milli>(t1 - t0).count();
    }
};

// ── Pipeline CPU ──────────────────────────────────────────────────────────────
double pipeline_cpu(const Mat& frame, Mat& result, int n_runs = 100) {
    vector<double> times;
    Timer t;

    for (int i = 0; i < n_runs; i++) {
        t.start();

        Mat gray, blurred, morph_result, edges, equalized;

        // 1. Filtro Gaussiano
        GaussianBlur(frame, blurred, Size(5, 5), 1.5);

        // 2. Convertir a grises para morfología y Canny
        cvtColor(blurred, gray, COLOR_BGR2GRAY);

        // 3. Operaciones morfológicas (Erosión + Dilatación = Opening)
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        erode(gray, morph_result, kernel);
        dilate(morph_result, morph_result, kernel);

        // 4. Detección de bordes Canny
        Canny(morph_result, edges, 50, 150);

        // 5. Ecualización del histograma
        equalizeHist(gray, equalized);

        times.push_back(t.ms());

        if (i == n_runs - 1) {
            // Combinar resultados para visualización
            Mat edges_color, equalized_color;
            cvtColor(edges, edges_color, COLOR_GRAY2BGR);
            cvtColor(equalized, equalized_color, COLOR_GRAY2BGR);

            Mat top, bot;
            hconcat(frame, edges_color, top);
            hconcat(equalized_color, Mat::zeros(frame.size(), CV_8UC3), bot);
            vconcat(top, bot, result);
        }
    }

    return accumulate(times.begin(), times.end(), 0.0) / times.size();
}

// ── Pipeline GPU-only ─────────────────────────────────────────────────────────
double pipeline_gpu(const Mat& frame, Mat& result, int n_runs = 100) {
    vector<double> times;
    Timer t;

    // Crear filtros una sola vez fuera del loop (eficiente)
    auto gaussian = cuda::createGaussianFilter(
        CV_8UC3, CV_8UC3, Size(5,5), 1.5);
    auto gaussian_gray = cuda::createGaussianFilter(
        CV_8UC1, CV_8UC1, Size(5,5), 1.5);
    auto canny = cuda::createCannyEdgeDetector(50, 150);
    auto erode_filter = cuda::createMorphologyFilter(
        MORPH_ERODE, CV_8UC1, getStructuringElement(MORPH_RECT, Size(3,3)));
    auto dilate_filter = cuda::createMorphologyFilter(
        MORPH_DILATE, CV_8UC1, getStructuringElement(MORPH_RECT, Size(3,3)));

    cuda::GpuMat d_frame, d_blurred, d_gray, d_morph, d_edges, d_equalized;

    for (int i = 0; i < n_runs; i++) {
        t.start();

        // CPU → GPU (solo una transferencia al inicio)
        d_frame.upload(frame);

        // 1. Filtro Gaussiano (GPU)
        gaussian->apply(d_frame, d_blurred);

        // 2. Convertir a grises (GPU)
        cuda::cvtColor(d_blurred, d_gray, COLOR_BGR2GRAY);

        // 3. Operaciones morfológicas (GPU)
        erode_filter->apply(d_gray, d_morph);
        dilate_filter->apply(d_morph, d_morph);

        // 4. Detección de bordes Canny (GPU)
        canny->detect(d_morph, d_edges);

        // 5. Ecualización del histograma (GPU)
        cuda::equalizeHist(d_gray, d_equalized);

        // Sincronizar GPU antes de medir tiempo
        cuda::Stream::Null().waitForCompletion();

        times.push_back(t.ms());

        if (i == n_runs - 1) {
            // GPU → CPU solo al final (eficiente)
            Mat edges_cpu, equalized_cpu;
            d_edges.download(edges_cpu);
            d_equalized.download(equalized_cpu);

            Mat edges_color, equalized_color;
            cvtColor(edges_cpu, edges_color, COLOR_GRAY2BGR);
            cvtColor(equalized_cpu, equalized_color, COLOR_GRAY2BGR);

            Mat top, bot;
            hconcat(frame, edges_color, top);
            hconcat(equalized_color, Mat::zeros(frame.size(), CV_8UC3), bot);
            vconcat(top, bot, result);
        }
    }

    return accumulate(times.begin(), times.end(), 0.0) / times.size();
}

// ── Mostrar resultados en consola ─────────────────────────────────────────────
void print_results(double ms_cpu, double ms_gpu, Size frame_size) {
    cout << "\n" << string(54, '=') << "\n";
    cout << "  Preprocesamiento CPU vs GPU — OpenCV CUDA\n";
    cout << "  Frame: " << frame_size.width << "x" << frame_size.height << "\n";
    cout << string(54, '=') << "\n";
    cout << left << setw(28) << "Métrica"
         << right << setw(12) << "CPU"
         << right << setw(12) << "GPU" << "\n";
    cout << string(54, '-') << "\n";
    cout << left << setw(28) << "Tiempo/frame (ms)"
         << right << setw(12) << fixed << setprecision(2) << ms_cpu
         << right << setw(12) << ms_gpu << "\n";
    cout << left << setw(28) << "FPS equivalente"
         << right << setw(12) << setprecision(1) << 1000.0/ms_cpu
         << right << setw(12) << 1000.0/ms_gpu << "\n";
    cout << string(54, '-') << "\n";
    cout << left << setw(28) << "Aceleración GPU/CPU"
         << right << setw(12) << ""
         << right << setw(11) << setprecision(1) << ms_cpu/ms_gpu << "x\n";
    cout << string(54, '=') << "\n\n";
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // Verificar CUDA
    int cuda_devices = cuda::getCudaEnabledDeviceCount();
    cout << "CUDA devices disponibles: " << cuda_devices << "\n";
    if (cuda_devices == 0) {
        cerr << "ERROR: No hay GPU CUDA disponible\n";
        return 1;
    }
    cuda::printShortCudaDeviceInfo(cuda::getDevice());

    // Fuente de video/imagen
    string source = (argc > 1) ? argv[1] : "0";  // 0 = webcam por defecto
    VideoCapture cap;

    if (source == "0")
        cap.open(0);
    else
        cap.open(source);

    if (!cap.isOpened()) {
        cerr << "No se pudo abrir la fuente: " << source << "\n";
        return 1;
    }

    // Warm-up: descartar primeros frames (GStreamer necesita estabilizarse)
    Mat frame;
    for (int i = 0; i < 10; i++) {
        cap >> frame;
        if (!frame.empty()) break;
    }
    cap.release();

    if (frame.empty()) {
        cerr << "Frame vacío — no se pudo capturar de la fuente\n";
        return 1;
    }

    // Redimensionar para consistencia
    resize(frame, frame, Size(640, 480));

    cout << "\nEjecutando pipeline CPU (100 runs)...\n";
    Mat result_cpu;
    double ms_cpu = pipeline_cpu(frame, result_cpu);

    cout << "Ejecutando pipeline GPU-only (100 runs)...\n";
    Mat result_gpu;
    double ms_gpu = pipeline_gpu(frame, result_gpu);

    // Mostrar tabla
    print_results(ms_cpu, ms_gpu, frame.size());

    // Guardar resultados en carpeta resultados/
    system("mkdir -p resultados");
    imwrite("resultados/resultado_cpu.png", result_cpu);
    imwrite("resultados/resultado_gpu.png", result_gpu);
    cout << "Imágenes guardadas: resultados/resultado_cpu.png / resultados/resultado_gpu.png\n";

    return 0;
}
