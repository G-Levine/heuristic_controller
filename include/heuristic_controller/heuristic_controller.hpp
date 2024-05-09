#ifndef HEURISTIC_CONTROLLER__HEURISTIC_CONTROLLER_HPP_
#define HEURISTIC_CONTROLLER__HEURISTIC_CONTROLLER_HPP_

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/subscription.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "realtime_tools/realtime_buffer.h"
#include "realtime_tools/realtime_publisher.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
// auto-generated by generate_parameter_library
#include "heuristic_controller_parameters.hpp"

namespace heuristic_controller {
using CmdType = geometry_msgs::msg::Twist;

enum locomotion_state {INIT, STAND, WALK, STOP};

struct control_step_state {
    locomotion_state state;
    double global_time;
    Eigen::Vector4d gait_phases;
    Eigen::Vector3d contact_centroid_pos_in_world;
    Eigen::Vector3d body_pos_in_world;
    Eigen::Vector3d body_vel_in_world;
    Eigen::Quaterniond body_rot_in_world;
    Eigen::Vector3d body_angvel_in_world;
    Eigen::Vector3d body_accel_in_world;
    Eigen::Vector4i contact_states;
    Eigen::Vector4d swing_t0s;
    Eigen::Vector4d swing_tfs;
    Eigen::Matrix<double, 3, 4> swing_x0s;
    Eigen::Matrix<double, 3, 4> swing_xfs;
    Eigen::Matrix<double, 3, 4> swing_v0s;
    Eigen::Matrix<double, 3, 4> swing_vfs;
    Eigen::Matrix<double, 3, 4> foot_pos_in_world;
    Eigen::Matrix<double, 3, 4> foot_vel_in_world;
    Eigen::Matrix<double, 3, 4> foot_pos_in_body_rotated;
    Eigen::Matrix<double, 3, 4> foot_vel_in_body_rotated;
    Eigen::Matrix<double, 3, 4> foot_pos_in_body_rotated_desired;
    Eigen::Matrix<double, 3, 4> balancing_forces_in_world;

    // Joint states
    Eigen::Matrix<double, 3, 4> joint_pos;
    Eigen::Matrix<double, 3, 4> joint_vel;

    // Joint control outputs
    Eigen::Matrix<double, 3, 4> joint_pos_desired;
    Eigen::Matrix<double, 3, 4> joint_vel_desired;
    Eigen::Matrix<double, 3, 4> joint_torque_desired;
    Eigen::Matrix<double, 3, 4> joint_kp_desired;
    Eigen::Matrix<double, 3, 4> joint_kd_desired;
};

struct control_step_inputs {
    // Desired body state
    Eigen::Vector3d body_pos_in_world_desired;
    Eigen::Vector3d body_vel_in_world_desired;
    Eigen::Quaterniond body_rot_in_world_desired;
    Eigen::Vector3d body_angvel_in_world_desired;

    // // State estimator parameters
    // double body_pos_estimator_smoothing;
    // double body_vel_estimator_smoothing;

    // Balance controller parameters
    // double friction_coeff;
    // double min_z_force;
    // double max_z_force;
    // double max_torque;
    // double body_mass;
    // Eigen::Vector3d pos_gains;
    // Eigen::Vector3d vel_gains;
    // Eigen::Vector3d rot_gains;
    // Eigen::Vector3d angvel_gains;
};


class HeuristicController : public controller_interface::ControllerInterface {
 public:
  HeuristicController();

  ~HeuristicController() = default;

  controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;

  controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;

  controller_interface::CallbackReturn on_init() override;

  controller_interface::CallbackReturn on_configure(
      const rclcpp_lifecycle::State &previous_state) override;

  controller_interface::CallbackReturn on_activate(
      const rclcpp_lifecycle::State &previous_state) override;

  controller_interface::CallbackReturn on_deactivate(
      const rclcpp_lifecycle::State &previous_state) override;

  controller_interface::return_type update(
      const rclcpp::Time &time, const rclcpp::Duration &period) override;

 protected:
  std::shared_ptr<ParamListener> param_listener_;
  Params params_;

  // Map from joint names to command types to command interfaces
  std::map<
      std::string,
      std::map<std::string, std::reference_wrapper<
                                hardware_interface::LoanedCommandInterface>>>
      command_interfaces_map_;

  // Map from joint/sensor names to state types to state interfaces
  std::map<std::string,
           std::map<std::string, std::reference_wrapper<
                                     hardware_interface::LoanedStateInterface>>>
      state_interfaces_map_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<CmdType>> rt_command_ptr_;
  rclcpp::Subscription<CmdType>::SharedPtr cmd_subscriber_;

  control_step_state prev_control_step_state_;
  control_step_inputs control_step_inputs_;

  rclcpp::Time init_time_;
  rclcpp::Time gait_start_time_;
};

}  // namespace heuristic_controller

#endif  // HEURISTIC_CONTROLLER__HEURISTIC_CONTROLLER_HPP_
