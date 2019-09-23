//
// Created by jlurobot on 19-9-22.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    if(argc != 2){
        cout << "usage: useLK path_to_dataset" << endl;
        return 1;
    }

    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";
    ifstream fin(associate_file);
    string rgb_file, depth_file, time_rgb, time_depth;
    list<Point2f> keypoints;
    Mat color, depth, last_color;
    for(int i = 0; i < 100; i++){
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        color = imread(path_to_dataset + "/" + rgb_file);
        depth = imread(path_to_dataset + "/" + depth_file);
        if( i == 0){
            vector<KeyPoint> kps;
            Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
            detector->detect(color, kps);
            for(auto kp: kps){
                keypoints.push_back(kp.pt);
            }
            last_color = color;
            continue;
        }

        if(color.data == nullptr || depth.data == nullptr) continue;

        vector<Point2f> next_keypoints, prev_keypoints;
        for(auto kp: keypoints)
            prev_keypoints.push_back(kp);
        vector<unsigned char> status;
        vector<float> error;

        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        calcOpticalFlowPyrLK(last_color, color, prev_keypoints, next_keypoints, status, error);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double> > (t2-t1);
        cout << "LK Flow used " << time_used.count() << " s." << endl;

        int j = 0;
        for(auto iter = keypoints.begin(); iter != keypoints.end(); j++){
            if(status[i] == 0){
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = next_keypoints[j];
            iter++;
        }

        cout << "tracked keypoints: " << keypoints.size() << endl;

        if(keypoints.empty()){
            cout << "all keypoints are lost." << endl;
            break;
        }

        Mat img_show = color.clone();
        for(auto kp : keypoints)
            circle(img_show, kp, 10, Scalar(0, 240, 0), 1);
        imshow("corners", img_show);
        waitKey(0);
        last_color = color;
    }

    return 0;
}