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
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"

// auto-generated by generate_parameter_library
#include "heuristic_controller_parameters.hpp"

namespace heuristic_controller {
using CmdType = geometry_msgs::msg::Twist;

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

  float cmd_x_vel_, cmd_y_vel_, cmd_yaw_vel_;

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

  rclcpp::Time init_time_;

  double init_joint_pos_[ACTION_SIZE];

  bool estop_active_;
};

}  // namespace heuristic_controller

#endif  // HEURISTIC_CONTROLLER__HEURISTIC_CONTROLLER_HPP_