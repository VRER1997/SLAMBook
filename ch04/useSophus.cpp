//
// Created by c1over on 2019-07-16.
//
#include <iostream>
#include <cmath>

using namespace std;

#include <Eigen/Geometry>
#include <Eigen/Core>

#include "sophus/so3.h"
#include "sophus/se3.h"

int main(int argc, char** argv){

    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
    Sophus::SO3 SO3_R(R);
//    Sophus::SO3 SO3_v(0, 0, M_PI/2);
//    Eigen::Quaterniond q(R);
//    Sophus::SO3 SO3_q(q);

//    cout << "SO(3) from matrix:" << SO3_R <<    endl;
//    cout << "SO(3) from vector:" << SO3_v << endl;
//    cout << "SO(3) from quaternion : " << SO3_q << endl;
    return 0;
}