//
// Created by jlurobot on 19-9-20.
//

#include <iostream>
#include <fstream>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp> //string format
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char ** argv){

    vector<cv::Mat> colorImage, depthImage;
    vector<Eigen::Isometry3d> poses;

    ifstream fin("./pose.txt");
    if(!fin){
        cerr << "error ! " << endl;
        return 1;
    }

    for(int i = 0; i < 5; i++){
        boost::format fmt("./%s/%d.%s");
        colorImage.push_back(cv::imread( (fmt%"color"%(i+1)%"png").str()));
        depthImage.push_back(cv::imread((fmt%"depth"%(i+1)%"pgm").str(), -1));

        double data[7] = {0};
        for(auto& d: data)
            fin >> d;
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);

        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;

    cout << "Transforming to PointMap" << endl;

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloud::Ptr pointCloud(new PointCloud);
    for(int i = 0; i < 5; i++){
        cout << "Transforming Image" << endl;
        cv::Mat color = colorImage[i];
        cv::Mat depth = depthImage[i];
        Eigen::Isometry3d T = poses[i];
        for(int v = 0; v < color.rows; v++){
            for(int u = 0; u < color.cols; u++){
                unsigned int d = depth.ptr<unsigned short> (v)[u];
                if(d == 0) continue;
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;

                PointT p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[v * color.step + u * color.channels()];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + u * color.channels() + 2];

                pointCloud->points.push_back(p);
            }
        }
    }

    pointCloud->is_dense = false;
    cout << "The number of point: " << pointCloud->size() << endl;
    pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
    return 0;

};