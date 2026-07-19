/**
 * Práctica 4-1C: Preprocesamiento de Imágenes CPU vs GPU — OpenCV CUDA
 * Universidad Politécnica Salesiana — Visión por Computador
 * Estudiante: Jordy Romero
 *
 * Pipeline GPU-only con cv::cuda::GpuMat:
 *   1. Suavizado Gaussiano
 *   2. Conversión a escala de grises
 *   3. Ecualización del histograma
 *   4. Operaciones morfológicas (Erosión → Dilatación)
 *   5. Detección de bordes (Canny)
 *
 * Uso:
 *   ./practica4_1c imagen.jpg      — benchmark en imagen fija (100 runs x 2 resoluciones)
 *   ./practica4_1c 0               — modo webcam en vivo (teclas: g=GPU, c=CPU, q=salir)
 */

#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock  = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── Pipeline CPU ──────────────────────────────────────────────────────────────
// Orden: Gaussian → gray → equalizeHist → erode/dilate → Canny
cv::Mat pipeline_cpu(const cv::Mat& frame, double& ms_out) {
    auto t0 = Clock::now();

    cv::Mat gray, blurred, eq, morph, edges;
    cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 1.5);
    cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, eq);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::erode(eq, morph, kernel);
    cv::dilate(morph, morph, kernel);
    cv::Canny(morph, edges, 50, 150);

    ms_out = elapsed_ms(t0);
    cv::Mat out;
    cv::cvtColor(edges, out, cv::COLOR_GRAY2BGR);
    return out;
}

// ── Pipeline GPU-only ─────────────────────────────────────────────────────────
// Un solo upload al inicio, todas las operaciones en GpuMat, un solo download al final.
// Los filtros CUDA se crean UNA vez fuera del bucle de benchmark: crearlos dentro
// del bucle paga el costo de planificación/allocación en cada iteración y destruye
// la ventaja de la GPU (error clásico de medición).
struct GpuPipeline {
    cv::Ptr<cv::cuda::Filter>          gauss;
    cv::Ptr<cv::cuda::Filter>          erode_f;
    cv::Ptr<cv::cuda::Filter>          dilate_f;
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny;

    explicit GpuPipeline() {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
        gauss   = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 1.5);
        erode_f = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE,  CV_8UC1, kernel);
        dilate_f= cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, kernel);
        canny   = cv::cuda::createCannyEdgeDetector(50, 150);
    }

    cv::Mat process(const cv::Mat& frame, double& ms_out) {
        auto t0 = Clock::now();

        cv::cuda::GpuMat d_frame, d_gray, d_blur, d_eq, d_morph, d_edges;

        // CPU → GPU (único upload)
        d_frame.upload(frame);

        // 1. Convertir a grises (GPU)
        cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY);

        // 2. Suavizado Gaussiano (GPU)
        gauss->apply(d_gray, d_blur);

        // 3. Ecualización del histograma (GPU)
        cv::cuda::equalizeHist(d_blur, d_eq);

        // 4. Operaciones morfológicas (GPU)
        erode_f->apply(d_eq, d_morph);
        dilate_f->apply(d_morph, d_morph);

        // 5. Detección de bordes Canny (GPU)
        canny->detect(d_morph, d_edges);

        // Sincronizar antes de medir tiempo
        cv::cuda::Stream::Null().waitForCompletion();
        ms_out = elapsed_ms(t0);

        // GPU → CPU (único download)
        cv::Mat result;
        d_edges.download(result);
        cv::Mat out;
        cv::cvtColor(result, out, cv::COLOR_GRAY2BGR);
        return out;
    }
};

// ── Tabla de resultados ───────────────────────────────────────────────────────
static void print_table(const std::string& label,
                         double ms_cpu, double ms_gpu) {
    std::cout << "\n" << std::string(54, '=') << "\n";
    std::cout << "  " << label << "\n";
    std::cout << std::string(54, '=') << "\n";
    std::cout << std::left  << std::setw(26) << "Metrica"
              << std::right << std::setw(12) << "CPU"
              << std::right << std::setw(12) << "GPU-only" << "\n";
    std::cout << std::string(54, '-') << "\n";
    std::cout << std::left  << std::setw(26) << "Tiempo/frame (ms)"
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << ms_cpu
              << std::right << std::setw(12) << ms_gpu << "\n";
    std::cout << std::left  << std::setw(26) << "FPS equivalente"
              << std::right << std::setw(12) << std::setprecision(1) << 1000.0 / ms_cpu
              << std::right << std::setw(12) << 1000.0 / ms_gpu << "\n";
    std::cout << std::string(54, '-') << "\n";
    std::cout << std::left  << std::setw(26) << "Aceleracion GPU/CPU"
              << std::right << std::setw(24) << std::setprecision(2) << ms_cpu / ms_gpu << "x\n";
    std::cout << std::string(54, '=') << "\n";
}

// ── Reflexión final ───────────────────────────────────────────────────────────
static void print_reflexion(double ms_cpu_sd, double ms_gpu_sd,
                             double ms_cpu_hd, double ms_gpu_hd) {
    std::cout << "\n" << std::string(54, '=') << "\n";
    std::cout << "  Reflexion: CPU vs GPU-only\n";
    std::cout << std::string(54, '=') << "\n";
    std::cout <<
        "Pipeline CPU<->GPU (alternado): cada operacion paga\n"
        "el costo de transferencia PCIe (upload + download).\n"
        "Con 5 operaciones, se paga 10 transferencias por frame.\n\n"
        "Pipeline GPU-only: un solo upload al inicio y un solo\n"
        "download al final, sin importar cuantas operaciones se\n"
        "encadenen. Esto elimina el cuello de botella de PCIe.\n\n"
        "Cuándo vale la GPU:\n"
        "- Cuando las operaciones son suficientemente pesadas\n"
        "  para amortizar el costo fijo de lanzar cada kernel.\n"
        "- Cuando se encadenan muchas operaciones (GPU-only\n"
        "  gana mas cuanto mas larga sea la cadena).\n"
        "- A mayor resolucion, mayor ventaja de la GPU:\n";
    std::cout << "  SD 640x480:  aceleracion = "
              << std::fixed << std::setprecision(2) << ms_cpu_sd / ms_gpu_sd << "x\n";
    std::cout << "  HD 1280x720: aceleracion = "
              << ms_cpu_hd / ms_gpu_hd << "x\n";
    std::cout << std::string(54, '=') << "\n\n";
}

// ── Benchmark sobre imagen fija ───────────────────────────────────────────────
static void run_benchmark(const cv::Mat& src) {
    const int N = 100;
    fs::create_directories("resultados");

    GpuPipeline gpu;

    // Warmup GPU: la primera llamada compila/inicializa recursos internos del driver
    double dummy;
    gpu.process(src, dummy);

    for (const auto& [label, size] : std::vector<std::pair<std::string, cv::Size>>{
             {"SD 640x480",  cv::Size(640,  480)},
             {"HD 1280x720", cv::Size(1280, 720)}}) {

        cv::Mat frame;
        cv::resize(src, frame, size);

        // Benchmark CPU
        std::vector<double> t_cpu, t_gpu;
        cv::Mat out_cpu, out_gpu;
        double ms;

        std::cout << "\nBenchmark CPU  — " << label << " (" << N << " runs)...\n";
        for (int i = 0; i < N; ++i) {
            out_cpu = pipeline_cpu(frame, ms);
            t_cpu.push_back(ms);
        }

        // Benchmark GPU
        std::cout << "Benchmark GPU  — " << label << " (" << N << " runs)...\n";
        for (int i = 0; i < N; ++i) {
            out_gpu = gpu.process(frame, ms);
            t_gpu.push_back(ms);
        }

        double avg_cpu = std::accumulate(t_cpu.begin(), t_cpu.end(), 0.0) / N;
        double avg_gpu = std::accumulate(t_gpu.begin(), t_gpu.end(), 0.0) / N;

        print_table("Preprocesamiento " + label, avg_cpu, avg_gpu);

        // Guardar mosaico: original | CPU | GPU
        cv::Mat orig_bgr = frame.clone();
        cv::Mat mosaic;
        cv::hconcat(std::vector<cv::Mat>{orig_bgr, out_cpu, out_gpu}, mosaic);

        std::string tag = (size.width == 640) ? "sd" : "hd";
        std::string path_cpu = "resultados/resultado_cpu_" + tag + ".png";
        std::string path_gpu = "resultados/resultado_gpu_" + tag + ".png";
        std::string path_mosaic = "resultados/mosaic_" + tag + ".png";
        cv::imwrite(path_cpu,    out_cpu);
        cv::imwrite(path_gpu,    out_gpu);
        cv::imwrite(path_mosaic, mosaic);
        std::cout << "Imagenes guardadas: " << path_mosaic << "\n";
    }

    // Reflexión comparativa entre las dos resoluciones
    // (re-corremos para tener ambos valores disponibles en esta función — simplificado)
    cv::Mat sd, hd;
    double ms_cpu_sd, ms_gpu_sd, ms_cpu_hd, ms_gpu_hd;
    cv::resize(src, sd, cv::Size(640,  480));
    cv::resize(src, hd, cv::Size(1280, 720));
    pipeline_cpu(sd, ms_cpu_sd);  gpu.process(sd, ms_gpu_sd);
    pipeline_cpu(hd, ms_cpu_hd);  gpu.process(hd, ms_gpu_hd);
    print_reflexion(ms_cpu_sd, ms_gpu_sd, ms_cpu_hd, ms_gpu_hd);
}

// ── Modo webcam en vivo ───────────────────────────────────────────────────────
static void run_webcam(int device) {
    cv::VideoCapture cap(device);
    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir la camara " << device << "\n";
        return;
    }

    std::cout << "Modo webcam — teclas: [g] GPU-only  [c] CPU  [q] salir\n";

    cv::Mat first;
    cap >> first;
    if (first.empty()) {
        std::cerr << "No se pudo leer el primer frame\n";
        return;
    }
    GpuPipeline gpu;
    // Warmup
    double dummy;
    gpu.process(first, dummy);

    bool use_gpu = true;
    cv::Mat frame, out;
    double ms;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        out = use_gpu ? gpu.process(frame, ms) : pipeline_cpu(frame, ms);

        std::string mode  = use_gpu ? "GPU-only" : "CPU";
        cv::Scalar  color = use_gpu ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        std::string label = mode + " | " + std::to_string(static_cast<int>(1000.0 / ms)) + " FPS"
                            + " | " + std::to_string(static_cast<int>(ms)) + " ms";
        cv::putText(out, label, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv::LINE_AA);

        cv::imshow("Practica4-1C | [g]=GPU [c]=CPU [q]=salir", out);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') break;
        if (key == 'g') use_gpu = true;
        if (key == 'c') use_gpu = false;
    }
    cv::destroyAllWindows();
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
        std::cerr << "ERROR: No hay GPU CUDA disponible\n";
        return 1;
    }
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <imagen.jpg | indice_camara>\n";
        return 1;
    }

    std::string arg = argv[1];
    bool is_number  = !arg.empty() &&
                      std::all_of(arg.begin(), arg.end(), ::isdigit);

    if (is_number) {
        run_webcam(std::stoi(arg));
    } else {
        cv::Mat src = cv::imread(arg);
        if (src.empty()) {
            std::cerr << "No se pudo leer la imagen: " << arg << "\n";
            return 1;
        }
        run_benchmark(src);
    }

    return 0;
}
