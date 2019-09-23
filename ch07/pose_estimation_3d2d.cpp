//
// Created by jlurobot on 19-9-22.
//
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat& img_1, const Mat& img_2, std::vector<KeyPoint>& keypoint_1,
                          std::vector<KeyPoint>& keypoint_2, std::vector<DMatch>& matches){

    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img_1, keypoint_1);
    detector->detect(img_2, keypoint_2);

    Mat descriptors_1, descriptors_2;

    descriptor->compute(img_1, keypoint_1, descriptors_1);
    descriptor->compute(img_2, keypoint_2, descriptors_2);

    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    double min_dist = 10000, max_dist = 0;

    for(int i = 0; i < descriptors_1.rows; i++){
        double dist = match[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    //printf("--Max Dist--: %f", max_dist);
    //printf("--Min Dist--: %f", min_dist);
    cout << max_dist << " " << min_dist << endl;

    for(int i = 0; i < descriptors_1.rows; i++){
        if(match[i].distance <= max(2*min_dist, 30.0)){
            matches.push_back(match[i]);
        }
    }
}

Point2f pixel2cam(const Point2f& p, Mat& K){
    return Point2f(
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

int main(int argc, char ** argv){

    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<DMatch> matches;
    vector<KeyPoint> keypoint_1, keypoint_2;
    find_feature_matches(img_1, img_1, keypoint_1, keypoint_2, matches);

    Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;

    for(DMatch m: matches){
        ushort d = d1.ptr<unsigned short> ( int(keypoint_1[m.queryIdx].pt.y))[int(keypoint_1[m.queryIdx].pt.x)];
        if(d == 0) continue;
        float dd = d / 1000.0;
        Point2f p1 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
        pts_3d.emplace_back(Point3f(p1.x*dd, p1.y*dd, dd));
        pts_2d.push_back(keypoint_2[m.trainIdx].pt);
    }

    cout << "3d-2d pairs :" << pts_3d.size() << endl;

    Mat r, t;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false, cv::SOLVEPNP_EPNP);

    cout << "r = " << r << endl;

    Mat R;
    Rodrigues(r, R);

    cout << "R = " << R << endl;
    cout << "t = " << t << endl;

    return 0;
}