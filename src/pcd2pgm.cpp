// Copyright 2025 Lihan Chen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pcd2pgm/pcd2pgm.hpp"

#include "pcl/common/transforms.h"
#include "pcl/filters/radius_outlier_removal.h"
#include "pcl/io/pcd_io.h"
#include "pcl_conversions/pcl_conversions.h"

namespace pcd2pgm
{
Pcd2PgmNode::Pcd2PgmNode(const rclcpp::NodeOptions & options) : Node("pcd2pgm", options)
{
  declareParameters();
  getParameters();

  rclcpp::QoS map_qos(10);
  map_qos.transient_local();
  map_qos.reliable();
  map_qos.keep_last(1);

  pcd_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  map_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(map_topic_name_, map_qos);
  pcd_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pcd_cloud", 10);

  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_, *pcd_cloud_) == -1) {
    RCLCPP_ERROR(get_logger(), "Couldn't read file: %s", pcd_file_.c_str());
    return;
  }

  RCLCPP_INFO(get_logger(), "Initial point cloud size: %lu", pcd_cloud_->points.size());

  applyTransform();

  passThroughFilter(thre_z_min_, thre_z_max_, flag_pass_through_);
  radiusOutlierFilter(cloud_after_pass_through_, thre_radius_, thres_point_count_);
  setMapTopicMsg(cloud_after_radius_, map_topic_msg_);

  timer_ =
    create_wall_timer(std::chrono::seconds(1), std::bind(&Pcd2PgmNode::publishCallback, this));
}

void Pcd2PgmNode::publishCallback()
{
  if (!cloud_after_radius_) {
    RCLCPP_WARN(get_logger(), "cloud_after_radius_ is nullptr.");
    return;
  }

  sensor_msgs::msg::PointCloud2 output;
  pcl::toROSMsg(*cloud_after_radius_, output);
  output.header.frame_id = "map";
  pcd_publisher_->publish(output);
  map_publisher_->publish(map_topic_msg_);
}

void Pcd2PgmNode::declareParameters()
{
  declare_parameter("pcd_file", "");
  declare_parameter("thre_z_min", 0.5);
  declare_parameter("thre_z_max", 2.0);
  declare_parameter("flag_pass_through", false);
  declare_parameter("thre_radius", 0.5);
  declare_parameter("map_resolution", 0.05);
  declare_parameter("thres_point_count", 10);
  declare_parameter("map_topic_name", "map");
  declare_parameter(
    "odom_to_lidar_odom", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});  // 新增的参数
}

void Pcd2PgmNode::getParameters()
{
  get_parameter("pcd_file", pcd_file_);
  get_parameter("thre_z_min", thre_z_min_);
  get_parameter("thre_z_max", thre_z_max_);
  get_parameter("flag_pass_through", flag_pass_through_);
  get_parameter("thre_radius", thre_radius_);
  get_parameter("map_resolution", map_resolution_);
  get_parameter("thres_point_count", thres_point_count_);
  get_parameter("map_topic_name", map_topic_name_);
  get_parameter("odom_to_lidar_odom", odom_to_lidar_odom_);  // 获取新的参数
}

void Pcd2PgmNode::passThroughFilter(double thre_low, double thre_high, bool flag_in)
{
  auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::PassThrough<pcl::PointXYZ> passthrough;
  passthrough.setInputCloud(pcd_cloud_);
  passthrough.setFilterFieldName("z");
  passthrough.setFilterLimits(thre_low, thre_high);
  passthrough.setNegative(false);
  passthrough.filter(*filtered_cloud);

  cloud_after_pass_through_ = filtered_cloud;
  RCLCPP_INFO(
    get_logger(), "After PassThrough filtering: %lu points",
    cloud_after_pass_through_->points.size());
}

void Pcd2PgmNode::radiusOutlierFilter(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & input_cloud, double radius, int thre_count)
{
  auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> radius_outlier;
  radius_outlier.setInputCloud(input_cloud);
  radius_outlier.setRadiusSearch(radius);
  radius_outlier.setMinNeighborsInRadius(thre_count);
  radius_outlier.filter(*filtered_cloud);

  cloud_after_radius_ = filtered_cloud;
  RCLCPP_INFO(
    get_logger(), "After RadiusOutlier filtering: %lu points", cloud_after_radius_->points.size());
}

void Pcd2PgmNode::setMapTopicMsg(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, nav_msgs::msg::OccupancyGrid & msg)
{
  msg.header.stamp = now();
  msg.header.frame_id = "map";

  msg.info.map_load_time = now();
  msg.info.resolution = map_resolution_;

  double x_min = std::numeric_limits<double>::max();
  double x_max = std::numeric_limits<double>::lowest();
  double y_min = std::numeric_limits<double>::max();
  double y_max = std::numeric_limits<double>::lowest();

  if (cloud->points.empty()) {
    RCLCPP_WARN(get_logger(), "Point cloud is empty!");
    return;
  }

  for (const auto & point : cloud->points) {
    x_min = std::min(x_min, static_cast<double>(point.x));
    x_max = std::max(x_max, static_cast<double>(point.x));
    y_min = std::min(y_min, static_cast<double>(point.y));
    y_max = std::max(y_max, static_cast<double>(point.y));
  }

  // 計算結果が負にならないよう、最小1に制限しつつ size_t に変換
  size_t width = static_cast<size_t>(std::max(1, static_cast<int>(std::ceil((x_max - x_min) / map_resolution_))));
  size_t height = static_cast<size_t>(std::max(1, static_cast<int>(std::ceil((y_max - y_min) / map_resolution_))));

  msg.info.width = static_cast<uint32_t>(width);
  msg.info.height = static_cast<uint32_t>(height);

  msg.info.origin.position.x = x_min;
  msg.info.origin.position.y = y_min;
  msg.info.origin.position.z = 0.0;
  msg.info.origin.orientation.x = 0.0;
  msg.info.origin.orientation.y = 0.0;
  msg.info.origin.orientation.z = 0.0;
  msg.info.origin.orientation.w = 1.0;

  msg.data.assign(width * height, -1);

  for (const auto & point : cloud->points) {
    double dx = point.x - x_min;
    double dy = point.y - y_min;

    if (dx < 0 || dy < 0) continue;

    size_t i = static_cast<size_t>(dx / map_resolution_);
    size_t j = static_cast<size_t>(dy / map_resolution_);

    if (i < width && j < height) {
      size_t index = j * width + i;
      if (index < msg.data.size() && point.z < 0.0) {
        // RCLCPP_INFO(get_logger(), "point z: %lf", point.z);
        if(msg.data[index]!=100){
          msg.data[index] = 0; 
        }
      } 
      else if(point.z >= 0.0){
      //   // RCLCPP_INFO(get_logger(), "point z: %lf", point.z);
         msg.data[index] = 100; 
      }
      else {
        RCLCPP_WARN(
          get_logger(),
          "Index out of bounds after safe casting: i=%zu, j=%zu, index=%zu, data.size=%zu",
          i, j, index, msg.data.size());
      }
    } else {
      RCLCPP_WARN(get_logger(), "Invalid index range: i=%zu, j=%zu (width=%zu, height=%zu)", i, j, width, height);
    }
  }

  RCLCPP_INFO(get_logger(), "Map width: %u, height: %u", msg.info.width, msg.info.height);
  RCLCPP_INFO(get_logger(), "Origin: x=%.2f, y=%.2f", x_min, y_min);
  RCLCPP_INFO(get_logger(), "Map data size: %zu", msg.data.size());
}

void Pcd2PgmNode::applyTransform()
{
  if (odom_to_lidar_odom_.size() != 6) {
    RCLCPP_ERROR(get_logger(), "Transform parameter 'odom_to_lidar_odom' must have 6 elements.");
    return;
  }

  Eigen::Affine3f transform = Eigen::Affine3f::Identity();

  transform.translation() << odom_to_lidar_odom_[0], odom_to_lidar_odom_[1], odom_to_lidar_odom_[2];
  transform.rotate(Eigen::AngleAxisf(odom_to_lidar_odom_[3], Eigen::Vector3f::UnitX()));
  transform.rotate(Eigen::AngleAxisf(odom_to_lidar_odom_[4], Eigen::Vector3f::UnitY()));
  transform.rotate(Eigen::AngleAxisf(odom_to_lidar_odom_[5], Eigen::Vector3f::UnitZ()));

  pcl::transformPointCloud(*pcd_cloud_, *pcd_cloud_, transform.inverse());
}

}  // namespace pcd2pgm

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pcd2pgm::Pcd2PgmNode)
