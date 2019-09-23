//
// Created by jlurobot on 19-9-21.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){

    if(argc != 3) {
        cout << "error! you should try: ./feature_extraction 1.jpg 2.jpg" << endl;
        return 1;
    }

    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoint_1, keypoint_2;
    Mat descriptors_1, descriptors_2;
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    orb->detect(img_1, keypoint_1);
    orb->detect(img_2, keypoint_2);

    orb->compute(img_1, keypoint_1, descriptors_1);
    orb->compute(img_2, keypoint_2, descriptors_2);

    Mat outimg1;
    drawKeypoints(img_1, keypoint_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("orb features", outimg1);

    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, matches);

    double min_dist = 10000, max_dist = 0;

    for(int i = 0; i < descriptors_1.rows; i++){
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    printf("--Max Dist--: %f", max_dist);
    printf("--Min Dist--: %f", min_dist);

    vector<DMatch> good_matches;
    for(int i = 0; i < descriptors_1.rows; i++){
        if(matches[i].distance <= max(2*min_dist, 30.0)){
            good_matches.push_back(matches[i]);
        }
    }


    Mat img_match;
    Mat img_goodmatch;

    drawMatches(img_1, keypoint_1, img_2, keypoint_2, matches, img_match);
    drawMatches(img_1, keypoint_1, img_2, keypoint_2, good_matches, img_goodmatch);

    imshow("all matched point", img_match);
    imshow("good matched point", img_goodmatch);
    waitKey(0);

    return 0;
}