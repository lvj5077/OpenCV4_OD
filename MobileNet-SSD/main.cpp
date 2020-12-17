#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "util.h"
using namespace cv;
using namespace dnn;
using namespace std;
int main( int argc, char* argv[] )
{
    // Open Video Capture
    // cv::VideoCapture capture = cv::VideoCapture( 0 );
    // if( !capture.isOpened() ){
    //     return -1;
    // }

    // Read Class Name List and Color Table
    const std::string list = "../voc.names";
    const std::vector<std::string> classes = readClassNameList( list );
    const std::vector<cv::Scalar> colors = getClassColors( classes.size() );

    // Read MobileNet-SSD
    const std::string model  = "../MobileNetSSD_deploy.caffemodel";
    const std::string config = "../MobileNetSSD_deploy.prototxt";
    cv::dnn::Net net = cv::dnn::readNet( model, config );
    if( net.empty() ){
        return -1;
    }

    /*
        Please see list of supported combinations backend/target.
        https://docs.opencv.org/4.2.0/db/d30/classcv_1_1dnn_1_1Net.html#a9dddbefbc7f3defbe3eeb5dc3d3483f4
    */

    // Set Preferable Backend
    net.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );

    // Set Preferable Target
    net.setPreferableTarget( cv::dnn::DNN_TARGET_OPENCL );

    // while( true )
    {
        // Read Frame
        cv::Mat frame;
        frame = cv::imread(argv[1]);
        // capture >> frame;
        // if( frame.empty() ){
        //     cv::waitKey( 0 );
        //     break;
        // }
        // if( frame.channels() == 4 ){
        //     cv::cvtColor( frame, frame, cv::COLOR_BGRA2BGR );
        // }

        // Create Blob from Input Image
        // MobileNet-SSD ( Scale : 2 / 255, Size : 300 x 300, Mean Subtraction : ( 127.5, 127.5, 127.5 ), Channels Order : BGR )
        cv::Mat blob = cv::dnn::blobFromImage( frame, 2 / 255.f, cv::Size( 300, 300 ), cv::Scalar( 127.5, 127.5, 127.5 ), true, false );

        // Set Input Blob
        net.setInput( blob );

        // Run Forward Network
        std::vector<cv::Mat> detections;
        net.forward( detections, getOutputsNames( net ) );

        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %.2f ms", t);
        

        std::vector<int32_t> class_ids; std::vector<float> confidences; std::vector<cv::Rect> rectangles;
        // Draw Detection Output
        for( cv::Mat& detection : detections ){
            // Transform Matrix to Nx7 from 1x1xNx7
            const std::vector<int32_t> size = { detection.size[2], detection.size[3] };
            cv::Mat mat( static_cast<int32_t>( size.size() ), &size[0], CV_32F, detection.ptr<float>() );
            for( int32_t i = 0; i < mat.rows; i++ ){
                // Retrieve Object
                //   object0 [batch_id][class_id][confidence][left][top][right][bottom]
                //   object1 [batch_id][class_id][confidence][left][top][right][bottom]
                //   object2 [batch_id][class_id][confidence][left][top][right][bottom]
                //   ...
                const cv::Mat object = mat.row( i );

                // Retrieve Class Index
                const int32_t class_id = static_cast<int32_t>( object.at<float>( 1 ) );

                // Retrieve and Check Confidence
                const float confidence = object.at<float>( 2 );
                constexpr float threshold = 0.2;
                if( threshold > confidence ){
                    continue;
                }

                // Retrieve Object Position
                const int32_t x      = static_cast<int32_t>( object.at<float>( 3 ) * frame.cols );
                const int32_t y      = static_cast<int32_t>( object.at<float>( 4 ) * frame.rows );
                const int32_t width  = static_cast<int32_t>( object.at<float>( 5 ) * frame.cols ) - x;
                const int32_t height = static_cast<int32_t>( object.at<float>( 6 ) * frame.rows ) - y;
                const cv::Rect rectangle = cv::Rect( x, y, width, height );

                // // Draw Rectangle
                // const cv::Scalar color = colors[class_id];
                // constexpr int32_t thickness = 3;
                // cv::rectangle( frame, rectangle, color, thickness );


                // Add Class ID, Confidence, Rectangle
                class_ids.push_back( class_id );
                confidences.push_back( confidence );
                rectangles.push_back( rectangle );


            }
        }



        // Remove Overlap Rectangles using Non-Maximum Suppression
        constexpr float confidence_threshold = 0.5; // Confidence
        constexpr float nms_threshold = 0.5; // IoU (Intersection over Union)
        std::vector<int32_t> indices;
        cv::dnn::NMSBoxes( rectangles, confidences, confidence_threshold, nms_threshold, indices );
        // Draw Rectangle
        for( const int32_t& index : indices ){
            const cv::Rect rectangle = rectangles[index];
            const cv::Scalar color = colors[class_ids[index]];
            constexpr int32_t thickness = 3;
            cv::rectangle( frame, rectangle, color, thickness );

            cout << "class: "<<classes[class_ids[index]]  <<" | rectangle " << rectangle << endl;
            
            
            cout << rectangle.x <<", "<< rectangle.y<<", " <<rectangle.width<<", "<< rectangle.height
                <<", " << class_ids[index] << endl<< endl;
        }

        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

        cv::imwrite("/Users/jin/Desktop/test01ssd.png",frame);
        // Show Image
        cv::imshow( "Object Detection SSD", frame );
        const int32_t key = cv::waitKey( 0 );
        // if( key == 'q' ){
        //     break;
        // }
    }

    cv::destroyAllWindows();

    return 0;
}
