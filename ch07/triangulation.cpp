//
// Created by jlurobot on 19-9-22.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Point2f pixel2cam(const Point2f& p, const Mat& K){
    return Point2f(
            (p.x - K.at<double> (0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double> (1, 2)) / K.at<double>(1, 1)
        );
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2,
                          std::vector<DMatch> matches, Mat& R, Mat& t){

    Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<Point2f> points1;
    vector<Point2f> points2;

    for(int i = 0; i < (int) matches.size(); i++){
        points1.push_back(keypoint_1[matches[i].queryIdx].pt);
        points2.push_back(keypoint_2[matches[i].trainIdx].pt);
    }

//    Mat fundamental_matrix;
//    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
//    cout << "fundamental matrix is : " << endl << fundamental_matrix << endl;

    Point2d principal_point(325.1, 249.7);
    int focal_length = 521;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point, RANSAC);
    cout << "essential_matrix is " << endl << essential_matrix << endl;


//    Mat homography_matrix;
//    homography_matrix = findHomography(points1, points2, RANSAC, 3, noArray(), 2000, 0.99);
//    cout << "homography matrix is" << endl << homography_matrix << endl;

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

void triangulation(
        const vector<KeyPoint>& keypoint_1,
        const vector<KeyPoint>& keypoint_2,
        const vector<DMatch>& matches,
        const Mat& R,
        const Mat& t,
        vector<Point3d>& points){

    Mat T1 = (Mat_<double>(3, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0);
    Mat T2 = (Mat_<double>(3, 4) <<
            R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), R.at<double>(1,0),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), R.at<double>(2,0));

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point2f> pts_1, pts_2;
    for(DMatch m : matches){
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }

    Mat pts_4d;
    triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for(int i = 0; i < pts_4d.cols; i++){
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0);
        Point3d p(
                x.at<float>(0, 0),
                x.at<float>(1, 0),
                x.at<float>(2, 0)
                );
        points.push_back(p);
    }
}



int main(int argc, char** argv){

    if(argc != 3){
        cout << "Error" << endl;
        return 1;
    }

    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);

    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);

    //configuration
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for(int i = 0; i < matches.size(); i++){
        Point2f pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        Point2f pt1_cam_3d (
                points[i].x / points[i].z,
                points[i].y / points[i].z
                );

        cout << "point in the first camera frame: " << pt1_cam << endl;
        cout << "point project from 3d " << pt1_cam_3d << ", d = " << points[i].z << endl;

        Point2f pt2_cam = pixel2cam(keypoints_2[matches[i].trainIdx].pt, K);
        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        pt2_trans /= pt2_trans.at<double>(2, 0);
        cout << "point in the second camera frame: " << pt2_cam << endl;
        cout << "point reproject from second frame: " << pt2_trans.t() << endl;
        cout << endl;
    }
    return 0;
}