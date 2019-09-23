//
// Created by jlurobot on 19-9-21.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void pose_estimation_2d2d(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2,
        std::vector<DMatch> matches, Mat& R, Mat& t){

    Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<Point2f> points1;
    vector<Point2f> points2;

    for(int i = 0; i < (int) matches.size(); i++){
        points1.push_back(keypoint_1[matches[i].queryIdx].pt);
        points2.push_back(keypoint_2[matches[i].trainIdx].pt);
    }

    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "fundamental matrix is : " << endl << fundamental_matrix << endl;

    Point2d principal_point(325.1, 249.7);
    int focal_length = 521;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point, RANSAC);
    cout << "essential_matrix is " << endl << essential_matrix << endl;


    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3, noArray(), 2000, 0.99);
    cout << "homography matrix is" << endl << homography_matrix << endl;

    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is" << endl << t << endl;
}

void find_feature_matches(const Mat& img_1, const Mat& img_2, std::vector<KeyPoint>& keypoint_1,
        std::vector<KeyPoint>& keypoint_2, std::vector<DMatch>& matches){

    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img_1, keypoint_1);
    detector->detect(img_2, keypoint_2);

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

Point2d pixel2cam(const Point2d& p, const Mat& K){
    return Point2d(
            (p.x - K.at<double> (0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double> (1, 2)) / K.at<double>(1, 1)
            );
}

int main(int argc, char** argv){
    if(argc != 3){
        cout << "Error ! You should try: ./pose_estimation_2d2d 1.jpg 2.jpg" << endl;
        return 1;
    }

    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoint_1, keypoint_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoint_1, keypoint_2, matches);
    cout << "the number of matches " << matches.size() << endl;

    Mat R, t;
    pose_estimation_2d2d(keypoint_1, keypoint_2, matches, R, t);

    Mat t_x = (Mat_<double>(3, 3) <<
            0, -t.at<double>(2, 0), t.at<double>(1, 0),
            t.at<double>(2, 0), 0, -t.at<double>(0, 0),
            -t.at<double>(1, 0), t.at<double>(0, 0), 0
            );

    cout << "t^R = " << endl << t_x * R << endl;

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for(DMatch m: matches){
        Point2d pt1 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoint_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);

        //configuration
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolor constrainnt = " << d << endl;
    }

    return 0;
}

//R is
//[0.9985534106102478, -0.05339308467584829, 0.006345444621109364;
//0.05321959721496342, 0.9982715997131746, 0.02492965459802013;
//-0.007665548311698023, -0.02455588961730218, 0.9996690690694515]
//t is
//[-0.8829934995085557;
//-0.05539655431450295;
//0.4661048182498381]
