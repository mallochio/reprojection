#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// constexpr const char* const _stream = "rtsp://omnidai.dtic.ua.es:554/live1.sdp";
// constexpr const char* const _stream = "rtsp://172.19.33.33:554/live1.sdp";
constexpr const char* const _stream = "rtsp://192.168.0.176:554/live1.sdp";

void write_image(const std::string& filename, const cv::Mat& image) {
    cv::imwrite(filename, image);
}

int main(int argc, char** argv) {
    std::cout << "Omnidirectional camera capture tool" << std::endl;

    double fps_param = 30;
    if (argc == 2) {
        fps_param = std::stod(argv[1]);
    } else {
        std::cerr << "Provide fps parameter" << std::endl;
        return -1;
    }

    cv::VideoCapture cap = cv::VideoCapture(_stream);

    auto epoch = std::chrono::high_resolution_clock::from_time_t(0);
    auto before = std::chrono::high_resolution_clock::now();

    double desired_hz = std::min(30., fps_param);
    int desired_diff = (int)(1000./desired_hz);

    int frame_no = 0;
    if (cap.isOpened()) {
        while (true) {
            auto now = std::chrono::high_resolution_clock::now();
            auto mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - before).count();
            auto mseconds_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(now - epoch).count();

            int waited = 0;
            if (mseconds < desired_diff) {
                waited = desired_diff - mseconds;
                std::this_thread::sleep_for(std::chrono::milliseconds(waited));
            }
            if (frame_no > 60) {
                std::cout << (mseconds+waited) << " ms/frame\t" << 1000./(mseconds+waited) << " FPS" << std::endl;
                frame_no = 0;
            }
            frame_no ++;

            cv::Mat image;
            bool ret = cap.read(image);

            if(ret) {
                // cv::imshow("camera", image);
                std::stringstream ss_rgb;
                ss_rgb << "capture0/omni/" << mseconds_epoch << ".jpg";
                std::thread(write_image, ss_rgb.str(), image).detach();
            } else {
                std::cerr << "Error: could not capture frame" << std::endl;
                exit(-1);
            }

            char key = (char)cv::waitKey(5);
            if(key == 27)
                break;
            
            before = now;
        }
    }


    return 0;
}
