#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "mini-yaml/Yaml.hpp"

using namespace std::chrono;

std::vector<int> write_parameters = {cv::IMWRITE_JPEG_QUALITY, 85};

void write_image(const std::string& filename, const cv::Mat& image) {
    cv::Mat dest;
    if(image.rows >= 1080) {
        cv::resize(image, dest, cv::Size(), 2/3., 2/3.);
    } else {
        dest = image;
    }
    cv::imwrite(filename, dest, write_parameters);
}

int main(int argc, char** argv) {
    int device_no = 0;
    double fps_param = 15;
    std::string work_path;

    if (argc != 4) {
        std::cerr << "Error: Incorrect argument number. Avoid calling this program directly, use launch script instead.";
        return -1;
    }
    device_no = std::stoi(argv[1]);
    work_path = argv[2];
    fps_param = std::stod(argv[3]);
    
    std::cout << "Kinect capture tool (device #" << device_no << ")" << std::endl;

    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = 0;
    libfreenect2::PacketPipeline *pipeline = 0;

    pipeline = new libfreenect2::OpenGLPacketPipeline();

    int num_devices = freenect2.enumerateDevices();
    if(num_devices == 0) {
        std::cerr << "Error: no device connected!" << std::endl;
        return -1;
    }
    
    std::string serial = freenect2.getDeviceSerialNumber(0);
    std::cout << "Detected Kinect with serial number: " << serial << std::endl;

    Yaml::Node root;
    Yaml::Parse(root, "kinects.yaml");

    Yaml::Node & kinects = root["kinects"];
    int device_index = 0;
    for (int k = 0; k < kinects.Size(); ++k) {
        if (kinects[k].As<std::string>() == serial) {
            std::cout << "Found kinect in config file: index = " << device_index << std::endl;
            break;
        }
        device_index ++;
    }
    if (device_index != device_no) {
        std::cerr << "Error: Sanity check failed. Expected kinect " << device_no << ", but found kinect " << device_index << ". Check the cabling and retry!" << std::endl;
        return -1;
    }
    if (device_index == kinects.Size()) {
        std::cerr << "Error: The connected Kinect device could not be found in the config file." << std::endl;
        return -1;
    }

    if(pipeline) {
        std::cout << "OpenGL pipeline created correctly" << std::endl;
        dev = freenect2.openDevice(serial, pipeline);
    } else {
        std::cout << "No pipeline" << std::endl;
        dev = freenect2.openDevice(serial);
    }

    /// [listeners]
    int types = 0;
    types |= libfreenect2::Frame::Color;
    types |= libfreenect2::Frame::Ir | libfreenect2::Frame::Depth;
    libfreenect2::SyncMultiFrameListener listener(types);
    libfreenect2::FrameMap frames;

    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);

    std::cout << "Starting ..." << std::endl;
    if (!dev->start())
        return -1;
  
    std::cout << "Passed" << std::endl;
    
    auto epoch = high_resolution_clock::from_time_t(0);
    auto before = high_resolution_clock::now();

    double desired_hz = std::min(30., fps_param);
    int desired_diff = (int)(1000./desired_hz);

    int frame_no = 0;
    while (true) {
        auto now = high_resolution_clock::now();
        auto mseconds = duration_cast<milliseconds>(now - before).count();
        auto mseconds_epoch = duration_cast<milliseconds>(now - epoch).count();

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

        if (!listener.waitForNewFrame(frames, 10*1000)) {  // 10[s]
            std::cout << "timeout!" << std::endl;
            return -1;
        }
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

        cv::Mat rgb_mat = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data).clone();
        cv::Mat ir_mat = cv::Mat(ir->height, ir->width, CV_32F, ir->data).clone();
        cv::Mat depth_mat = cv::Mat(depth->height, depth->width, CV_32F, depth->data).clone();
        cv::Mat ir_vis, depth_16u;
        ir_mat.convertTo(ir_vis, CV_8U, 255./65535.);
        depth_mat.convertTo(depth_16u, CV_16U);
        
        std::stringstream ss_rgb;
        ss_rgb << work_path << "/rgb/" << mseconds_epoch << ".jpg";
        std::thread (write_image, ss_rgb.str(), rgb_mat).detach();

        std::stringstream ss_ir;
        ss_ir << work_path << "/ir/" << mseconds_epoch << ".jpg";
        std::thread (write_image, ss_ir.str(), ir_vis).detach();

        std::stringstream ss_depth;
        ss_depth << work_path << "/depth/" << mseconds_epoch << ".png";
        std::thread (write_image, ss_depth.str(), depth_16u).detach();

        char key = (char)cv::waitKey(5);
        if(key == 27)
            break;

        listener.release(frames);
        before = now;
    }

    dev->stop();
    dev->close();
    
    return 0;
}
