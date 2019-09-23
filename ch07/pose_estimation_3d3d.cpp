//
// Created by jlurobot on 19-9-22.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat& img_1, const Mat& img_2,
        vector<KeyPoint>& keypoint_1,
        vector<KeyPoint>& keypoint_2,
        vector<DMatch>& matches
        ){

    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);

    cout << img_1.size << endl;
    detector->detect(img_1, keypoint_1);
    detector->detect(img_2, keypoint_2);

    Mat description_1, description_2;
    descriptor->compute(img_1, keypoint_1, description_1);
    descriptor->compute(img_2, keypoint_2, description_2);

    vector<DMatch> match;
    matcher->match(description_1, description_2, match);

    double min_dist = 10000, max_dist = 0;
    for(int i = 0; i < match.size(); i++){
        double dist = match[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    cout << "Min_dist = " << min_dist << endl;

    for(int i = 0; i < match.size(); i++){
        if(match[i].distance <= min(2*min_dist, 30.0)){
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

void pose_estimation_3d3d(
        const vector<Point3f>& pts1,
        const vector<Point3f>& pts2,
        Mat& R, Mat& t
        ){
    Point3f p1, p2;
    int N = pts1.size();
    for(int i = 0; i < N; i++){
        p1 += pts1[i];
        p2 += pts2[i];
    }

    p1 /= N, p2 /= N;
    vector<Point3f> q1(N), q2(N);
    for(int i = 0; i < N; i++){
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(int i = 0; i < N; i++){
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << "W = " << W << endl;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout << "U = " << U << endl;
    cout << "V = " << V << endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    R = (Mat_<double>(3, 3) <<
            R_(0,0), R_(0,1), R_(0,2),
            R_(1,0), R_(1,1), R_(1,2),
            R_(2,0), R_(2,1), R_(2,2)
            );
    t = (Mat_<double>(3, 1) << t_(0,0), t_(1,0), t_(2,0));
}

int main(int argc, char** argv){

    if(argc != 5){
        cout << "Error "<< endl;
        return 1;
    }

    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoint_1, keypoint_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoint_1, keypoint_2, matches);
    cout << "the number of keypoint_matches: " << matches.size() << endl;

    Mat depth_1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    Mat depth_2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

    vector<Point3f> pts_1, pts_2;
    for(DMatch m: matches){
        ushort d1 = depth_1.ptr<unsigned short>(int(keypoint_1[m.queryIdx].pt.y))[int(keypoint_1[m.queryIdx].pt.x)];
        ushort d2 = depth_2.ptr<unsigned short>(int(keypoint_2[m.trainIdx].pt.y))[int(keypoint_2[m.trainIdx].pt.x)];
        if(d1 == 0 || d2 == 0) continue;

        Point2f p1 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
        Point2f p2 = pixel2cam(keypoint_2[m.trainIdx].pt, K);
        float dd1 = float(d1) / 5000.0;
        float dd2 = float(d2) / 5000.0;
        pts_1.push_back(Point3f(p1.x*dd1, p1.y*dd1, dd1));
        pts_2.push_back(Point3f(p2.x*dd2, p2.y*dd2, dd2));
    }


    cout << "The number of 3D-3D pair: " << pts_1.size() << endl;

    Mat R, t;
    pose_estimation_3d3d(pts_1, pts_2, R, t);

    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "R_inv = " << R.t() << endl;
    cout << "t_inv = " << -R.t() << endl;

    return 0;
}