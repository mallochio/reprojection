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
    double fps_param = 30;
    if (argc >= 2)
        device_no = std::stoi(argv[1]);
    else
    {
        std::cerr << "Insufficient parameters: provide device_no and optionally fps";
        return -1;
    }
    if (argc == 3) {
        fps_param = std::stod(argv[2]);
    } else {
        std::cerr << "Excess parameters: provide only device_no and optionally fps";
    }
    

    std::cout << "Kinect capture tool (device #" << device_no << ")" << std::endl;

    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = 0;
    libfreenect2::PacketPipeline *pipeline = 0;

    pipeline = new libfreenect2::OpenGLPacketPipeline();

    int num_devices = freenect2.enumerateDevices();
    if(num_devices == 0) {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }
    
    if(device_no > num_devices-1) {
        std::cout << "No device #" << device_no << " found. Only " << num_devices << " connected.";
        return -1;
    }

    std::string serial = freenect2.getDeviceSerialNumber(device_no);
    // freenect2.getDefaultDeviceSerialNumber();

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
    /// [registration setup]
    libfreenect2::Registration* registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
    libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);
    /// [registration setup]
    
    /*
    cv::namedWindow("RGB");
    cv::namedWindow("IR");
    cv::namedWindow("Depth");
    */

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

        registration->apply(rgb, depth, &undistorted, &registered);

        cv::Mat rgb_mat = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data).clone();
        cv::Mat ir_mat = cv::Mat(ir->height, ir->width, CV_32F, ir->data).clone();
        cv::Mat depth_mat = cv::Mat(depth->height, depth->width, CV_32F, depth->data).clone();
        cv::Mat ir_vis, depth_vis, depth_16u;
        ir_mat.convertTo(ir_vis, CV_8U, 255./65535.);
        // depth_mat.convertTo(depth_vis, CV_8U, 255./4500.);
        depth_mat.convertTo(depth_16u, CV_16U);
        /*
        cv::imshow("RGB", rgb_mat);
        cv::imshow("IR", ir_vis);
        cv::imshow("Depth", depth_vis);
        */

        std::stringstream ss_rgb;
        ss_rgb << "capture" << device_no << "/rgb/" << mseconds_epoch << ".jpg";
        // cv::imwrite(ss_rgb.str(), rgb_mat);
        std::thread (write_image, ss_rgb.str(), rgb_mat).detach();

        std::stringstream ss_ir;
        ss_ir << "capture" << device_no << "/ir/" << mseconds_epoch << ".jpg";
        // cv::imwrite(ss_ir.str(), ir_vis);
        std::thread (write_image, ss_ir.str(), ir_vis).detach();

        std::stringstream ss_depth;
        ss_depth << "capture" << device_no << "/depth/" << mseconds_epoch << ".png";
        // cv::imwrite(ss_depth.str(), depth_16u);
        std::thread (write_image, ss_depth.str(), depth_16u).detach();

        char key = (char)cv::waitKey(5);
        if(key == 27)
            break;

        listener.release(frames);
        before = now;
    }

    dev->stop();
    dev->close();

    delete registration;
    
    return 0;
}
