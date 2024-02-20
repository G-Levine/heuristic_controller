#include "heuristic_controller/heuristic_controller.hpp"
#include "heuristic_controller/kinematics.hpp"
#include "heuristic_controller/planning.hpp"

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

  // Initialize the command subscriber
  cmd_subscriber_ = get_node()->create_subscription<CmdType>(
      "/cmd_vel", rclcpp::SystemDefaultsQoS(),
      [this](const CmdType::SharedPtr msg) {
        rt_command_ptr_.writeFromNonRT(msg);
      });
    
  // Populate the control step inputs
  control_step_inputs_.body_pos_in_world_desired(2) = params_.body_height;

  // Populate the previous control step state
  prev_control_step_state_.state = locomotion_state::INIT;
  prev_control_step_state_.global_time = (get_node()->now()).seconds();
  prev_control_step_state_.contact_states.setOnes();
  // Reset the gait phases
  for (int i = 0; i < 4; i++) {
    prev_control_step_state_.gait_phases(i) = -params_.phase_offsets(i);
  }

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

  control_step_state curr_control_step_state;
  curr_control_step_state.global_time = time.seconds();

  // Get the latest observation
  try {
    // read IMU states from hardware interface
    curr_control_step_state.body_angvel_in_world.x() = state_interfaces_map_.at(params_.imu_sensor_name)
                   .at("angular_velocity.x")
                   .get()
                   .get_value();
    curr_control_step_state.body_angvel_in_world.y() = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.y")
                    .get()
                    .get_value();
    curr_control_step_state.body_angvel_in_world.z() = state_interfaces_map_.at(params_.imu_sensor_name)
                  .at("angular_velocity.z")
                  .get()
                  .get_value();
    curr_control_step_state.body_rot_in_world.w() = state_interfaces_map_.at(params_.imu_sensor_name)
                        .at("orientation.w")
                        .get()
                        .get_value();
    curr_control_step_state.body_rot_in_world.x() = state_interfaces_map_.at(params_.imu_sensor_name)
                        .at("orientation.x")
                        .get()
                        .get_value();
    curr_control_step_state.body_rot_in_world.y() = state_interfaces_map_.at(params_.imu_sensor_name)
                        .at("orientation.y")
                        .get()
                        .get_value();
    curr_control_step_state.body_rot_in_world.z() = state_interfaces_map_.at(params_.imu_sensor_name)
                        .at("orientation.z")
                        .get()
                        .get_value();
  }

  // Get the latest commanded velocities
  auto command = rt_command_ptr_.readFromRT();
  if (command && command->get()) {
    Eigen::Vector3d body_vel_desired;
    Eigen::Vector3d body_angvel_desired;
    body_vel_desired.x() = command->get()->linear.x;
    body_vel_desired.y() = command->get()->linear.y;
    body_angvel_desired.z() = command->get()->angular.z;

    // Make a rotation matrix from the current body orientation but only the yaw axis
    auto yaw_rot = extractYawQuaternion(curr_control_step_state.body_rot_in_world);
    control_step_inputs_.body_vel_in_world_desired = yaw_rot * body_vel_desired;
    control_step_inputs_.body_angvel_in_world_desired = body_angvel_desired;
    control_step_inputs_.body_pos_in_world_desired += control_step_inputs_.body_vel_in_world_desired * period.seconds();

    // Update the desired body orientation
    Eigen::AngleAxisd rot_delta(control_step_inputs_.body_angvel_in_world_desired.norm() * period.seconds(), control_step_inputs_.body_angvel_in_world_desired.normalized());
    control_step_inputs_.body_rot_in_world_desired = curr_control_step_state.body_rot_in_world * rot_delta;
    control_step_inputs_.body_rot_in_world_desired = control_step_inputs_.body_rot_in_world_desired.normalized();
  }

  // Do state estimation using the current IMU and joint states but the previous step's contact states
  auto[contact_centroid_pos_in_world, body_pos_in_world, body_vel_in_world, foot_pos_in_world, foot_vel_in_world] = stateEstimation(curr_control_step_state, prev_control_step_state_);
  curr_control_step_state.contact_centroid_pos_in_world = contact_centroid_pos_in_world;
  curr_control_step_state.body_pos_in_world = body_pos_in_world;
  curr_control_step_state.body_vel_in_world = body_vel_in_world;
  curr_control_step_state.foot_pos_in_world = foot_pos_in_world;
  curr_control_step_state.foot_vel_in_world = foot_vel_in_world;

  // Locomotion state machine
  if (prev_control_step_state_.state == locomotion_state::INIT) {
    control_step_inputs.body_rot_in_world_desired = curr_control_step_state.body_rot_in_world;
    curr_control_step_state.state = locomotion_state::STAND;
    return controller_interface::return_type::OK;
  } else if (prev_control_step_state_.state == locomotion_state::STAND) {
    // Reset the gait phases
    for (int i = 0; i < 4; i++) {
      curr_control_step_state.gait_phases(i) = -params_.phase_offsets(i);
    }
  } else if (prev_control_step_state_.state == locomotion_state::WALK) {
    // Update the gait phases
    curr_control_step_state.gait_phases = prev_control_step_state_.gait_phases + params_.gait_frequency * period.seconds();
  } else if (prev_control_step_state_.state == locomotion_state::STOP) {
    for (auto &command_interface : command_interfaces_) {
      command_interface.set_value(0.0);
    }
    return controller_interface::return_type::OK;
  }

  // Handle swing and stance phase transitions
  if (curr_control_step_state.state == locomotion_state::STAND || curr_control_step_state.state == locomotion_state::WALK) {
    for (int i = 0; i < 4; i++) {
      bool is_new_phase = (curr_control_step_state.gait_phases(i) > 0.0) && (fmod(curr_control_step_state.gait_phases(i), 1.0) > params_.phase_offsets(i));

      if (prev_control_step_state_.contact_states(i) == 0) {
        // Leg is in swing phase
        // Persist the parameters of the swing trajectory from the previous control step
        curr_control_step_state.swing_t0s(i) = prev_control_step_state_.swing_t0s(i);
        curr_control_step_state.swing_tfs(i) = prev_control_step_state_.swing_tfs(i);
        curr_control_step_state.swing_x0s.col(i) = prev_control_step_state_.swing_x0s.col(i);
        curr_control_step_state.swing_v0s.col(i) = prev_control_step_state_.swing_v0s.col(i);

        if (curr_control_step_state.global_time > control_step_state.swing_tfs(i)) {
          // Switch to stance phase
          curr_control_step_state.contact_states(i) = 1;
        }

      } else {
        // Leg is in stance phase
        if (is_new_phase) {
          // Switch to swing phase
          // Begin the swing trajectory from the current foot position
          curr_control_step_state.contact_states(i) = 0;
          curr_control_step_state.swing_t0s(i) = curr_control_step_state.global_time;
          curr_control_step_state.swing_tfs(i) = curr_control_step_state.global_time + params_.swing_durations(i);
          curr_control_step_state.swing_x0s.col(i) = curr_control_step_state.foot_pos_in_world.col(i);
          curr_control_step_state.swing_v0s.col(i) = curr_control_step_state.foot_vel_in_world.col(i);
        }
      }
    }
  }

  // If the contact states changed, update the contact centroid position
  if (curr_control_step_state.contact_states != prev_control_step_state_.contact_states) {
    auto contact_centroid_pos_delta = contactCentroid(curr_control_step_state.foot_pos_in_world, curr_control_step_state.contact_states, prev_control_step_state_.contact_centroid_pos_in_world) - contactCentroid(curr_control_step_state.foot_pos_in_world, prev_control_step_state_.contact_states, prev_control_step_state_.contact_centroid_pos_in_world);
    curr_control_step_state.contact_centroid_pos_in_world = prev_control_step_state_.contact_centroid_pos_in_world + contact_centroid_pos_delta
  }

  // Do control
  TODO

  for (int i = 0; i < 12; i++) {
    command_interfaces_map_.at(params_.joint_names[i])
        .at("position")
        .get()
        .set_value(joint_pos_desired[i]);
    command_interfaces_map_.at(params_.joint_names[i])
        .at("velocity")
        .get()
        .set_value(joint_vel_desired[i]);
    command_interfaces_map_.at(params_.joint_names[i])
        .at("effort")
        .get()
        .set_value(joint_torque_desired[i]);
    command_interfaces_map_.at(params_.joint_names[i])
        .at("kp")
        .get()
        .set_value(joint_kp_desired[i]);
    command_interfaces_map_.at(params_.joint_names[i])
        .at("kd")
        .get()
        .set_value(joint_kd_desired[i]);
  }

  // Update the previous control step state
  prev_control_step_state_ = curr_control_step_state;

  return controller_interface::return_type::OK;
}

}  // namespace heuristic_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(heuristic_controller::HeuristicController,
                       controller_interface::ControllerInterface)