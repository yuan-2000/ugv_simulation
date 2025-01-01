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
const double a = std::sqrt(6);      // 椭圆长轴参数
const double b = std::sqrt(4);      // 椭圆短轴参数
const double c = 1.0;               // 椭圆常数
const double k = 0.05;               // 向量场增益
const double alpha = 1.0;           // 非线性增益
const double k_theta = 2.0;         // 偏航角控制增益
const double alpha_u = 1;        // 角速度非线性增益
const int PUBLISH_RATE = 80;        // 控制频率（Hz）
const double v_max = 3.0;           // 最大线速度
const double omega_max = 3.0;       // 最大角速度
const double t_p = 3.0;            // 预设时间
const double error = 0.001;          // 预设时间误差

// 机器人状态
double x = 0.0, y = 0.0, theta = 0.0;
// 保存小车的轨迹点
std::vector<geometry_msgs::Point> robot_trajectory;

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

// amcl_pose回调函数，更新机器人位置和姿态
void amclPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
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
}
// 用于计算预设时间向量场的时变增益
void calculateKGain(double simulation_t, double& k_gain, double& dot_k_gain,ros::Publisher& k_gain_pub) {
    if (simulation_t <= t_p) {
        double eta = (10 / pow(t_p, 6)) * pow(simulation_t, 6)
                     - (24 / pow(t_p, 5)) * pow(simulation_t, 5)
                     + (15 / pow(t_p, 4)) * pow(simulation_t, 4);
        double dot_eta = (60 / pow(t_p, 6)) * pow(simulation_t, 5)
                         - (120 / pow(t_p, 5)) * pow(simulation_t, 4)
                         + (60 / pow(t_p, 4)) * pow(simulation_t, 3);
        double second_dot_eta = (300 / pow(t_p, 6)) * pow(simulation_t, 4)
                                - (480 / pow(t_p, 5)) * pow(simulation_t, 3)
                                + (180 / pow(t_p, 4)) * pow(simulation_t, 2);
        k_gain = dot_eta / (1 - eta + error);
        ROS_INFO("The calculate k_Gain: %.5f", k_gain);
        //发布k_gain
        std_msgs::Float64 k_gain_msg;
        k_gain_msg.data = k_gain;  // 发布k_gain
        k_gain_pub.publish(k_gain_msg);
        dot_k_gain = (second_dot_eta * (1 - eta + error) + pow(dot_eta, 2)) / pow((1 - eta + error),2);
    } else {
        k_gain = 0;
        dot_k_gain = 0;
        //发布k_gain
        std_msgs::Float64 k_gain_msg;
        k_gain_msg.data = k_gain;  // 发布k_gain
        k_gain_pub.publish(k_gain_msg);
    }
}
// 计算向量场
void computeVectorField(double x, double y, double& vx, double& vy, double& e_norm,double k_gain,ros::Publisher& error_pub,ros::Publisher& vy_pub) {
    // 路径误差计算
    double e = (x * x) / (a * a) + (y * y) / (b * b) - c;
    //发布误差e
    std_msgs::Float64 error_msg;
    error_msg.data = std::abs(e);  // 发布误差的绝对值
    error_pub.publish(error_msg);

    if (std::isnan(e)) {
        ROS_WARN("Invalid error value: NaN detected!");
        return;  // 如果误差计算无效，则返回
    }
    // 使用 Eigen 库简化矩阵和向量的操作
    Eigen::Vector2d n;  // 梯度向量 n
    n << (2 * x) / (a * a), (2 * y) / (b * b);  // 填充梯度向量
    //n的2范数
    double n_norm = n.norm();
    n_norm=n_norm*n_norm;
    // 计算 nke
    ROS_INFO("The VectorField k_Gain: %.5f", k_gain);
    // Eigen::Vector2d nke = k *(k_gain+1)*e* n;
    Eigen::Vector2d nke = (k_gain/n_norm+k)*e* n;

    // 旋转矩阵 E
    Eigen::Matrix2d E;
    E << 0, -1, 1, 0;
    // 计算 v = E * n - nke
    // Eigen::Vector2d v_vec = E * n - nke;
    Eigen::Vector2d v_vec =  - nke;
    vx = v_vec(0);  // 获取 x 分量
    vy = v_vec(1);  // 获取 y 分量
    //发布速度vy
    std_msgs::Float64 vy_msg;
    vy_msg.data = vy;  // 发布速度vy
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
    // 初始化算法开始时间
    ros::Time start_time = ros::Time::now();
    double start_t = start_time.toSec(); // 记录开始时间
    // 订阅话题
    ros::Subscriber amcl_sub = nh.subscribe("/amcl_pose", 5, amclPoseCallback);

    ros::Publisher cmd_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 5);
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 5);
    // 添加一个发布器，用于发布误差e
    ros::Publisher error_pub = nh.advertise<std_msgs::Float64>("/error_e", 5);
    //发布速度vy
    ros::Publisher vy_pub = nh.advertise<std_msgs::Float64>("/vy", 5);
    //发布k_gain
    ros::Publisher k_gain_pub = nh.advertise<std_msgs::Float64>("/k_gain", 5);
    ros::Rate rate(PUBLISH_RATE);  // 控制循环频率


    // 开始路径跟踪
    double vx = 0.0, vy = 0.0, e_norm = 0.0;
    double v_u = 0.0, omega_u = 0.0;
    double k_gain = 0.0, dot_k_gain = 0.0;
    double simulation_t = 0.0;
    while (ros::ok()) {
        // 获取当前算法运行时间
        // simulation_t = (ros::Time::now() - start_time).toSec();
        simulation_t += 1.0 / PUBLISH_RATE; // 假设时间步长为1/PUBLISH_RATE
        // 计算预设时间向量场的时变增益
        // ROS_INFO("The Start Time: %.5f", start_t);
        // ROS_INFO("The Current Time: %.5f", ros::Time::now().toSec());
        ROS_INFO("The Current Simulation Time: %.5f", simulation_t);
        calculateKGain(simulation_t, k_gain, dot_k_gain,k_gain_pub);
        // 计算向量场
        computeVectorField(x, y, vx, vy, e_norm,k_gain,error_pub,vy_pub);
        // 计算控制输入
        // computeControlLaw(vx, vy, x, y, theta, v_u, omega_u,error_pub,k_gain,dot_k_gain);
        
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
        ROS_INFO("Position: (x: %.2f, y: %.2f), Theta: %.2f", x, y, theta);
        ROS_INFO("Control: vx: %.2f, vy: %.2f", vx, vy);

        // 记录机器人当前位置到轨迹
        geometry_msgs::Point robot_position;
        robot_position.x = x;
        robot_position.y = y;
        robot_position.z = 0.0;
        robot_trajectory.push_back(robot_position);

        // 发布可视化标记
        publishMarkers(marker_pub);
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}