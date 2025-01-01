#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float64.h>  // 引入 Float64 消息类型
#include <geometry_msgs/PoseWithCovarianceStamped.h> 
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <visualization_msgs/Marker.h>
#include <tf/tf.h>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

// 全局参数配置（可调节）
const double a = std::sqrt(4);      // 椭圆长轴参数
const double b = std::sqrt(4);      // 椭圆短轴参数
const double c = 1.0;               // 椭圆常数
const double k = 0.05;               // 向量场增益
const double alpha = 1.0;           // 非线性增益
const double k_theta = 2.0;         // 偏航角控制增益
const double alpha_u = 1;        // 角速度非线性增益
const int PUBLISH_RATE = 20;        // 控制频率（Hz）
const double v_max = 2.0;           // 最大线速度
const double omega_max = 3.0;       // 最大角速度

// 机器人状态
double x = 0.0, y = 0.0, theta = 0.0;
// 保存小车的轨迹点
std::vector<geometry_msgs::Point> robot_trajectory;
ros::Publisher vx_odom_pub, vy_odom_pub, error_pub, norm_pub, vx_pub, vy_pub;  // 发布者声明
// 工具函数：计算符号函数 sign(x)
inline double sign(double x) {
    return (x > 0) - (x < 0);
}

// 限幅函数
double saturate(double value, double limit) {
    if (value > limit) {
        ROS_WARN("Value %.2f exceeds positive limit %.2f, saturating to limit.", value, limit);
        return limit;
    }
    if (value < -limit) {
        ROS_WARN("Value %.2f exceeds negative limit %.2f, saturating to limit.", value, -limit);
        return -limit;
    }
    return value;
}
//里程计回调函数，发布机器人速度信息
void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    if (std::isnan(msg->pose.pose.position.x) || std::isnan(msg->pose.pose.position.y)) {
        ROS_WARN("Received invalid AMCL pose data: NaN detected!");
        return;  // 如果位置数据无效，则返回，不进行更新
    }
    // 更新位置
    x = msg->pose.pose.position.x;
    y = msg->pose.pose.position.y;

    // 获取四元数，计算偏航角（theta）
    tf::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch;
    m.getRPY(roll, pitch, theta);  // 获取roll, pitch, yaw
    // 更新速度
    double vx_odom = msg->twist.twist.linear.x;
    double vy_odom = msg->twist.twist.linear.y;
    std_msgs::Float64 vx_msg;
    vx_msg.data = vx_odom;
    std_msgs::Float64 vy_msg;
    vy_msg.data = vy_odom;
    vx_odom_pub.publish(vx_msg);
    vy_odom_pub.publish(vy_msg);
}
// amcl_pose回调函数，更新机器人位置和姿态
// void amclPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
//     if (std::isnan(msg->pose.pose.position.x) || std::isnan(msg->pose.pose.position.y)) {
//         ROS_WARN("Received invalid AMCL pose data: NaN detected!");
//         return;  // 如果位置数据无效，则返回，不进行更新
//     }
//     // 更新位置
//     x = msg->pose.pose.position.x;
//     y = msg->pose.pose.position.y;
//     // 获取四元数，计算偏航角（theta）
//     tf::Quaternion q(
//         msg->pose.pose.orientation.x,
//         msg->pose.pose.orientation.y,
//         msg->pose.pose.orientation.z,
//         msg->pose.pose.orientation.w);
//     tf::Matrix3x3 m(q);
//     double roll, pitch;
//     m.getRPY(roll, pitch, theta);  // 获取roll, pitch, yaw
// }

// 计算向量场
void computeVectorField(double x, double y, double& vx, double& vy, double& e_norm) {
    // 路径误差计算
    double e = (x * x) / (a * a) + (y * y) / (b * b) - c;
    ROS_DEBUG("Computed error (e): %.5f", e);  // 输出误差e
    if (std::isnan(e)) {
        ROS_WARN("Invalid error value: NaN detected!");
        return;  // 如果误差计算无效，则返回
    }
    //发布误差e的绝对值
    std_msgs::Float64 error_msg;
    error_msg.data = std::abs(e);  // 发布误差的绝对值
    error_pub.publish(error_msg);
    // 使用 Eigen 库简化矩阵和向量的操作
    Eigen::Vector2d n;  // 梯度向量 n
    n << (2 * x) / (a * a), (2 * y) / (b * b);  // 填充梯度向量
    // ROS_DEBUG("Gradient vector n: [%.4f, %.4f]", n(0), n(1));  // 输出梯度向量
    // double phi_e = sign(e) * std::pow(std::abs(e), alpha);
    double phi_e = e;

    // 计算 nke
    Eigen::Vector2d nke = k * phi_e * n;
    // 旋转矩阵 E
    Eigen::Matrix2d E;
    E << 0, -1, 1, 0;
    // 计算 v = E * n - nke
    Eigen::Vector2d v_vec = E*n- nke;
    // ROS_DEBUG("Velocity vector v: [%.4f, %.4f]", v_vec(0), v_vec(1));  // 输出v_vec
    //发布二范数
    std_msgs::Float64 norm_msg;
    norm_msg.data = nke.norm();
    norm_pub.publish(norm_msg);
    // Eigen::Vector2d v_vec =  -nke;
    vx = v_vec(0);  // 获取 x 分量
    vy = v_vec(1);  // 获取 y 分量
    //发布vx,vy
    std_msgs::Float64 vx_msg;
    vx_msg.data = vx;
    vx_pub.publish(vx_msg);
    std_msgs::Float64 vy_msg;
    vy_msg.data = vy;
    vy_pub.publish(vy_msg);
    // 误差范数
    e_norm = std::sqrt(e * e);
}


void publishMarkers(ros::Publisher& marker_pub) {
    visualization_msgs::Marker path_marker, traj_marker;

    // 椭圆轨迹
    path_marker.header.frame_id = traj_marker.header.frame_id = "map";  
    path_marker.header.stamp = traj_marker.header.stamp = ros::Time::now();
    path_marker.ns = "ellipse_path";
    traj_marker.ns = "robot_trajectory";
    path_marker.id = 0;
    traj_marker.id = 1;
    path_marker.type = traj_marker.type = visualization_msgs::Marker::LINE_STRIP;
    path_marker.action = traj_marker.action = visualization_msgs::Marker::ADD;
    path_marker.scale.x = traj_marker.scale.x = 0.05;  // 线宽
    path_marker.color.r = 1.0;  // 红色椭圆轨迹
    path_marker.color.a = 1.0;
    traj_marker.color.g = 1.0;  // 绿色机器人轨迹
    traj_marker.color.a = 1.0;

    // 添加椭圆轨迹点
    for (int i = 0; i <= 360; i++) {
        geometry_msgs::Point p;
        p.x = a * std::cos(i * M_PI / 180.0);
        p.y = b * std::sin(i * M_PI / 180.0);
        p.z = 0.0;
        path_marker.points.push_back(p);
    }

    // 添加机器人轨迹点
    traj_marker.points = robot_trajectory;

    // 发布标记
    marker_pub.publish(path_marker);
    marker_pub.publish(traj_marker);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "vector_field_path_tracking");
    ros::NodeHandle nh;

    ros::Publisher cmd_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 5);
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 5);
    // 添加一个发布器，用于发布误差e
     error_pub = nh.advertise<std_msgs::Float64>("/error_e", 5);
    //发布速度vx,vy,vx_odom,vy_odom
     vx_pub = nh.advertise<std_msgs::Float64>("/vx", 5);
     vy_pub = nh.advertise<std_msgs::Float64>("/vy", 5);
     vx_odom_pub = nh.advertise<std_msgs::Float64>("/vx_odom", 5);
     vy_odom_pub = nh.advertise<std_msgs::Float64>("/vy_odom", 5);
    //发布二范数
     norm_pub = nh.advertise<std_msgs::Float64>("/norm", 5);
    // 订阅话题
    ros::Subscriber odom_sub = nh.subscribe("/odom", 5, odomCallback); // 订阅里程计话题
    // ros::Subscriber amcl_sub = nh.subscribe("/amcl_pose", 5, amclPoseCallback);
    // ros::Subscriber laser_pose_sub = nh.subscribe("robot_pose", 1, laserPoseCallback);


    ros::Rate rate(PUBLISH_RATE);  // 控制循环频率


    // 开始路径跟踪
    double vx = 0.0, vy = 0.0, e_norm = 0.0;
    double v_u = 0.0, omega_u = 0.0;
    while (ros::ok()) {
        ros::spinOnce();

        // 计算向量场
        computeVectorField(x, y, vx, vy, e_norm); 
        // 确保位置和速度信息没有NaN值
        if (std::isnan(x) || std::isnan(y) || std::isnan(v_u) || std::isnan(omega_u)) {
        ROS_ERROR("Invalid initial odom data: NaN detected!");
        return -1;
        }
        // 发布控制命令
        geometry_msgs::Twist cmd_vel;
        cmd_vel.linear.x = vx;
        cmd_vel.linear.y = vy;
        cmd_pub.publish(cmd_vel);
        // 输出调试信息
        // ROS_INFO("Position: (x: %.5f, y: %.5f), Theta: %.5f", x, y, theta);
        // ROS_INFO("Control: Linear Velocity: %.2f, Angular Velocity: %.2f", vx, vy);
        // ROS_INFO("Control: vx: %.5f, vy: %.5f", vx, vy);


        // 记录机器人当前位置到轨迹
        geometry_msgs::Point robot_position;
        robot_position.x = x;
        robot_position.y = y;
        robot_position.z = 0.0;
        robot_trajectory.push_back(robot_position);

        // 发布可视化标记
        publishMarkers(marker_pub);

        rate.sleep();
    }

    return 0;
}