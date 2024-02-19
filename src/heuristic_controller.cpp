#include "heuristic_controller/heuristic_controller.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "controller_interface/helpers.hpp"
#include "hardware_interface/loaned_command_interface.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/qos.hpp"

namespace heuristic_controller {
HeuristicController::HeuristicController()
    : controller_interface::ControllerInterface(),
      rt_command_ptr_(nullptr),
      cmd_subscriber_(nullptr) {}

controller_interface::CallbackReturn HeuristicController::on_init() {
  try {
    param_listener_ = std::make_shared<ParamListener>(get_node());
    params_ = param_listener_->get_params();
  } catch (const std::exception &e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n",
            e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn HeuristicController::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  RCLCPP_INFO(get_node()->get_logger(), "configure successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
HeuristicController::command_interface_configuration() const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::ALL};
}

controller_interface::InterfaceConfiguration
HeuristicController::state_interface_configuration() const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::ALL};
}

controller_interface::CallbackReturn HeuristicController::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  rt_command_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<CmdType>>(nullptr);

  // Populate the command interfaces map
  for (auto &command_interface : command_interfaces_) {
    command_interfaces_map_[command_interface.get_prefix_name()].emplace(
        command_interface.get_interface_name(), std::ref(command_interface));
  }

  // Populate the state interfaces map
  for (auto &state_interface : state_interfaces_) {
    state_interfaces_map_[state_interface.get_prefix_name()].emplace(
        state_interface.get_interface_name(), std::ref(state_interface));
  }

  // Store the initial joint positions
  for (int i = 0; i < ACTION_SIZE; i++) {
    init_joint_pos_[i] = state_interfaces_map_.at(params_.joint_names[i])
                             .at("position")
                             .get()
                             .get_value();
  }

  init_time_ = get_node()->now();

  // Initialize the command subscriber
  cmd_subscriber_ = get_node()->create_subscription<CmdType>(
      "/cmd_vel", rclcpp::SystemDefaultsQoS(),
      [this](const CmdType::SharedPtr msg) {
        rt_command_ptr_.writeFromNonRT(msg);
      });

  RCLCPP_INFO(get_node()->get_logger(), "activate successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn HeuristicController::on_deactivate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  rt_command_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<CmdType>>(nullptr);
  for (auto &command_interface : command_interfaces_) {
    command_interface.set_value(0.0);
  }
  RCLCPP_INFO(get_node()->get_logger(), "deactivate successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type HeuristicController::update(
    const rclcpp::Time &time, const rclcpp::Duration &period) {
  // When started, return to the default joint positions
  double time_since_init = (time - init_time_).seconds();
  if (time_since_init < params_.init_duration) {
    for (int i = 0; i < ACTION_SIZE; i++) {
      // Interpolate between the initial joint positions and the default joint
      // positions
      double interpolated_joint_pos =
          init_joint_pos_[i] * (1 - time_since_init / params_.init_duration) +
          params_.default_joint_pos[i] *
              (time_since_init / params_.init_duration);
      command_interfaces_map_.at(params_.joint_names[i])
          .at("position")
          .get()
          .set_value(interpolated_joint_pos);
      command_interfaces_map_.at(params_.joint_names[i])
          .at("kp")
          .get()
          .set_value(params_.init_kps[i]);
      command_interfaces_map_.at(params_.joint_names[i])
          .at("kd")
          .get()
          .set_value(params_.init_kds[i]);
    }
    return controller_interface::return_type::OK;
  }

  // After the init_duration has passed, fade in the policy actions
  double time_since_fade_in =
      (time - init_time_).seconds() - params_.init_duration;
  float fade_in_multiplier =
      std::min(time_since_fade_in / params_.fade_in_duration, 1.0);

  // If an emergency stop has been triggered, set all commands to 0 and return
  if (estop_active_) {
    for (auto &command_interface : command_interfaces_) {
      command_interface.set_value(0.0);
    }
    return controller_interface::return_type::OK;
  }

  // Get the latest commanded velocities
  auto command = rt_command_ptr_.readFromRT();
  if (command && command->get()) {
    cmd_x_vel_ = command->get()->linear.x;
    cmd_y_vel_ = command->get()->linear.y;
    cmd_yaw_vel_ = command->get()->angular.z;
  }

  // Get the latest observation
  double pitch_vel, roll_vel, yaw_vel, orientation_w, orientation_x,
      orientation_y, orientation_z;
  try {
    // read IMU states from hardware interface
    pitch_vel = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.y")
                    .get()
                    .get_value();
    roll_vel = state_interfaces_map_.at(params_.imu_sensor_name)
                   .at("angular_velocity.x")
                   .get()
                   .get_value();
    yaw_vel = state_interfaces_map_.at(params_.imu_sensor_name)
                  .at("angular_velocity.z")
                  .get()
                  .get_value();
    orientation_w = state_interfaces_map_.at(params_.imu_sensor_name)
                        .at("orientation.w")
                        .get()
                        .get_value();
    orientation_x = state_interfaces_map_.at(params_.imu_sensor_name)
                        .at("orientation.x")
                        .get()
                        .get_value();
    orientation_y = state_interfaces_map_.at(params_.imu_sensor_name)
                        .at("orientation.y")
                        .get()
                        .get_value();
    orientation_z = state_interfaces_map_.at(params_.imu_sensor_name)
                        .at("orientation.z")
                        .get()
                        .get_value();

    // Calculate the projected gravity vector
    tf2::Quaternion q(orientation_x, orientation_y, orientation_z,
                      orientation_w);
    tf2::Matrix3x3 m(q);
    tf2::Vector3 world_gravity_vector(0, 0, -1);
    tf2::Vector3 projected_gravity_vector = m.inverse() * world_gravity_vector;

    // If the maximum body angle is exceeded, trigger an emergency stop
    if (-projected_gravity_vector[2] < cos(params_.max_body_angle)) {
      estop_active_ = true;
      RCLCPP_INFO(get_node()->get_logger(), "Emergency stop triggered");
      return controller_interface::return_type::OK;
    }

    control_step();

    // Send the action to the hardware interface
    command_interfaces_map_.at(params_.joint_names[i])
        .at("position")
        .get()
        .set_value((double)action_[i]);
    // command_interfaces_map_.at(params_.joint_names[i])
    //     .at("velocity")
    //     .get()
    //     .set_value((double)action_[i]);
    command_interfaces_map_.at(params_.joint_names[i])
        .at("kp")
        .get()
        .set_value(params_.kps[i]);
    command_interfaces_map_.at(params_.joint_names[i])
        .at("kd")
        .get()
        .set_value(params_.kds[i]);
  }

  return controller_interface::return_type::OK;
}

void control_step() {
  // Update the target x, y, and yaw positions and velocities
  target_yaw_ += target_yaw_vel_ * params_.control_period;

  // Calculate the balancing torque
  double pitch_vel, roll_vel, yaw_vel, orientation_w, orientation_x,
      orientation_y, orientation_z;

  tf2::Quaternion q(orientation_x, orientation_y, orientation_z,
                    orientation_w);
  tf2::Matrix3x3 orientation(q);

  // Construct target orientation matrix from roll, pitch, and yaw
  tf2::Quaternion q_target;
  q_target.setRPY(target_roll_, target_pitch_, target_yaw_);
  tf2::Matrix3x3 orientation_target(q_target);

  orientation_error = orientation.inverse() * orientation_target;
  // Convert to angle-axis representation

  // orientation_error_torque = orientation_gains * orientation_error
  // angular_velocity_error_torque = angular_velocity_gains * angular_velocity_error

  // body_torque = orientation_error_torque + angular_velocity_error_torque
  // body_force = position_gains * position_error

  // Find the least-squares ground reaction forces to produce the desired body
  // force and torque, given the current positions of the feet and the contact states

  // ground_reaction_forces = ...

  // Loop through the 4 legs
  for (int i = 0; i < 4; i++) {
    if in_stance[i] {

    }
    else {

    }

  }
}

}  // namespace heuristic_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(heuristic_controller::HeuristicController,
                       controller_interface::ControllerInterface)