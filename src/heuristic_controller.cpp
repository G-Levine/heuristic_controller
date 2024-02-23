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
  // for (int i = 0; i < ACTION_SIZE; i++) {
  //   init_joint_pos_[i] = state_interfaces_map_.at(params_.joint_names[i])
  //                            .at("position")
  //                            .get()
  //                            .get_value();
  // }

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
    prev_control_step_state_.gait_phases(i) = -params_.phase_offsets[i];
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
    // read joint states from hardware interface
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        int joint_index = 3 * i + j;
        curr_control_step_state.joint_pos(j, i) = state_interfaces_map_.at(params_.joint_names[joint_index])
            .at("position")
            .get()
            .get_value();
        curr_control_step_state.joint_vel(j, i) = state_interfaces_map_.at(params_.joint_names[joint_index])
            .at("velocity")
            .get()
            .get_value();
      }
    }
  }
  catch (const std::exception &e) {
    fprintf(stderr, "Exception thrown during update stage with message: %s \n",
            e.what());
    return controller_interface::return_type::ERROR;
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

  // Handle swing and stance phase transitions
  if (curr_control_step_state.state == locomotion_state::STAND || curr_control_step_state.state == locomotion_state::WALK) {
    for (int i = 0; i < 4; i++) {
      if (prev_control_step_state_.contact_states(i) == 0) {
        // Leg is in swing phase
        if (curr_control_step_state.global_time > curr_control_step_state.swing_tfs(i)) {
          // Switch to stance phase
          curr_control_step_state.contact_states(i) = 1;
          RCLCPP_INFO(get_node()->get_logger(), "Switching leg %d from swing to stance", i);
        }
      } else {
        // Leg is in stance phase
        bool is_new_phase = (curr_control_step_state.gait_phases(i) > 0.0) && (fmod(curr_control_step_state.gait_phases(i), 1.0) > params_.phase_offsets[i]);
        if (is_new_phase) {
          // Switch to swing phase
          curr_control_step_state.contact_states(i) = 0;
          RCLCPP_INFO(get_node()->get_logger(), "Switching leg %d from stance to swing", i);
        }
      }
    }
  }

  // If the contact states changed, update the contact centroid position
  Eigen::Vector3d contact_centroid_pos_delta = calculateAverageContactValue(prev_control_step_state_.foot_pos_in_world, curr_control_step_state.contact_states) - calculateAverageContactValue(prev_control_step_state_.foot_pos_in_world, prev_control_step_state_.contact_states);
  curr_control_step_state.contact_centroid_pos_in_world = prev_control_step_state_.contact_centroid_pos_in_world + contact_centroid_pos_delta;

  // Do state estimation
  Eigen::Matrix<double, 3, 4> foot_pos_in_body = four_legs_fk(curr_control_step_state.joint_pos);
  Eigen::Matrix<double, 3, 4> foot_pos_in_body_rotated;
  Eigen::Matrix<double, 3, 4> foot_vel_in_body;
  Eigen::Matrix<double, 3, 4> foot_vel_in_body_rotated;
  std::array<Eigen::Matrix3d, 4> jacobians = four_legs_jacobian(curr_control_step_state.joint_pos);
  for (int i = 0; i < 4; i++) {
    foot_pos_in_body_rotated.col(i) = curr_control_step_state.body_rot_in_world * foot_pos_in_body.col(i);
    foot_vel_in_body.col(i) = jacobians[i] * curr_control_step_state.joint_vel.col(i);
    foot_vel_in_body_rotated.col(i) = curr_control_step_state.body_rot_in_world * foot_vel_in_body.col(i) + curr_control_step_state.body_angvel_in_world.cross(foot_pos_in_body_rotated.col(i));
  }
  Eigen::Vector3d contact_centroid_pos_in_body_rotated = calculateAverageContactValue(foot_pos_in_body_rotated, curr_control_step_state.contact_states);
  Eigen::Vector3d contact_centroid_vel_in_body_rotated = calculateAverageContactValue(foot_vel_in_body_rotated, curr_control_step_state.contact_states);
  curr_control_step_state.body_pos_in_world = curr_control_step_state.contact_centroid_pos_in_world - contact_centroid_pos_in_body_rotated;
  curr_control_step_state.body_vel_in_world = -contact_centroid_vel_in_body_rotated;
  for (int i = 0; i < 4; i++) {
    curr_control_step_state.foot_pos_in_world.col(i) = curr_control_step_state.body_pos_in_world + foot_pos_in_body_rotated.col(i);
    curr_control_step_state.foot_vel_in_world.col(i) = curr_control_step_state.body_vel_in_world + foot_vel_in_body_rotated.col(i);
  }

  bool is_curr_state_capturable_by_stand_controller = (curr_control_step_state.body_vel_in_world.norm() < params_.stand_controller_max_body_vel) && (curr_control_step_state.body_angvel_in_world.norm() < params_.stand_controller_max_body_angvel);

  // Locomotion state machine
  if (prev_control_step_state_.state == locomotion_state::INIT) {
    // Set the initial desired body orientation to be the current body orientation
    control_step_inputs_.body_rot_in_world_desired = curr_control_step_state.body_rot_in_world;
    curr_control_step_state.state = locomotion_state::STAND;
    curr_control_step_state.contact_states.setOnes();
    prev_control_step_state_ = curr_control_step_state;
    RCLCPP_INFO(get_node()->get_logger(), "Switching from init to stand");
    return controller_interface::return_type::OK;
  } else if (prev_control_step_state_.state == locomotion_state::STAND) {
    // Reset the gait phases
    for (int i = 0; i < 4; i++) {
      curr_control_step_state.gait_phases(i) = -params_.phase_offsets[i];
    }
    if (prev_control_step_state_.contact_states.sum() == 4) {
      // If all legs are in stance, set the desired body position to be centered above the contact centroid
      control_step_inputs_.body_pos_in_world_desired(0) = curr_control_step_state.contact_centroid_pos_in_world(0);
      control_step_inputs_.body_pos_in_world_desired(1) = curr_control_step_state.contact_centroid_pos_in_world(1);
    } else {
      // If any leg is in swing, nullify the x and y effects of the body position controller
      control_step_inputs_.body_pos_in_world_desired(0) = curr_control_step_state.body_pos_in_world(0);
      control_step_inputs_.body_pos_in_world_desired(1) = curr_control_step_state.body_pos_in_world(1);
    }

    // if (!is_curr_state_capturable_by_stand_controller) {
    //   curr_control_step_state.state = locomotion_state::WALK;
    //   RCLCPP_INFO(get_node()->get_logger(), "Switching from stand to walk");
    // }
  } else if (prev_control_step_state_.state == locomotion_state::WALK) {
    // Update the gait phases
    curr_control_step_state.gait_phases = prev_control_step_state_.gait_phases + Eigen::Vector4d::Ones() * params_.gait_frequency * period.seconds();

    // Nullify the x and y effects of the body position controller
    control_step_inputs_.body_pos_in_world_desired(0) = curr_control_step_state.body_pos_in_world(0);
    control_step_inputs_.body_pos_in_world_desired(1) = curr_control_step_state.body_pos_in_world(1);

    if (is_curr_state_capturable_by_stand_controller) {
      curr_control_step_state.state = locomotion_state::STAND;
      RCLCPP_INFO(get_node()->get_logger(), "Switching from walk to stand");
    }
  } else if (prev_control_step_state_.state == locomotion_state::STOP) {
    for (auto &command_interface : command_interfaces_) {
      command_interface.set_value(0.0);
    }
    return controller_interface::return_type::OK;
  }

  // Get balancing forces from the balancing QP
  Eigen::Vector3d body_pos_error = control_step_inputs_.body_pos_in_world_desired - curr_control_step_state.body_pos_in_world;
  Eigen::Vector3d body_vel_error = control_step_inputs_.body_vel_in_world_desired - curr_control_step_state.body_vel_in_world;
  Eigen::Vector3d body_rot_error = (control_step_inputs_.body_rot_in_world_desired * curr_control_step_state.body_rot_in_world.inverse()).vec();
  Eigen::Vector3d body_angvel_error = control_step_inputs_.body_angvel_in_world_desired - curr_control_step_state.body_angvel_in_world;
  Eigen::Vector3d balancing_force_desired = body_pos_error * params_.balancing_force_kp + body_vel_error * params_.balancing_force_kd;
  Eigen::Vector3d balancing_torque_desired = body_rot_error * params_.balancing_torque_kp + body_angvel_error * params_.balancing_torque_kd;
  balancing_force_desired.setZero();
  balancing_force_desired(2) = -1.0;
  balancing_torque_desired.setZero();
  Eigen::Matrix<double, 3, 4> balancing_forces_in_world = balancingQP(foot_pos_in_body_rotated, curr_control_step_state.contact_states, balancing_force_desired, balancing_torque_desired, params_.min_normal_force, params_.max_normal_force, params_.friction_coefficient);

  std::cout << "Pos error: \n" << body_pos_error << std::endl;
  std::cout << "Vel error: \n" << body_vel_error << std::endl;
  std::cout << "Rot error: \n" << body_rot_error << std::endl;
  std::cout << "Angvel error: \n" << body_angvel_error << std::endl;
  std::cout << "Foot pos in body rotated: \n" << foot_pos_in_body_rotated << std::endl;
  std::cout << "Balancing forces: \n" << balancing_forces_in_world << std::endl;
  balancing_forces_in_world.setZero();

  // Obtain the desired actuator commands for each leg depending on whether it's in swing or stance
  if (curr_control_step_state.state == locomotion_state::STAND || curr_control_step_state.state == locomotion_state::WALK) {
    // Calculate the jacobian for the leg

    Eigen::Matrix<double, 3, 4> foot_pos_in_body_desired;
    Eigen::Matrix<double, 3, 4> foot_vel_in_body_desired;

    for (int i = 0; i < 4; i++) {
      if (curr_control_step_state.contact_states(i) == 0) {
        // Leg is in swing phase
        if (prev_control_step_state_.contact_states(i) == 1) {
          // Switching to swing phase
          // Begin the swing trajectory from the current foot position
          curr_control_step_state.contact_states(i) = 0;
          curr_control_step_state.swing_t0s(i) = curr_control_step_state.global_time;
          curr_control_step_state.swing_tfs(i) = curr_control_step_state.global_time + params_.swing_durations[i];
          curr_control_step_state.swing_x0s.col(i) = curr_control_step_state.foot_pos_in_world.col(i);
          curr_control_step_state.swing_v0s.col(i) = curr_control_step_state.foot_vel_in_world.col(i);
        } else {
          // Continuing in swing phase
          // Persist the parameters of the swing trajectory from the previous control step
          curr_control_step_state.swing_t0s(i) = prev_control_step_state_.swing_t0s(i);
          curr_control_step_state.swing_tfs(i) = prev_control_step_state_.swing_tfs(i);
          curr_control_step_state.swing_x0s.col(i) = prev_control_step_state_.swing_x0s.col(i);
          curr_control_step_state.swing_v0s.col(i) = prev_control_step_state_.swing_v0s.col(i);
        }

        // Evaluate the swing trajectory
        double stance_time = (1 - params_.swing_durations[i]) / params_.gait_frequency;
        double body_yaw = extractYawFromQuaternion(curr_control_step_state.body_rot_in_world);
        auto[foot_pos_in_world_desired, foot_vel_in_world_desired] = calculateSwingFootPosVel(curr_control_step_state.global_time, curr_control_step_state.swing_t0s(i), curr_control_step_state.swing_tfs(i), stance_time, params_.body_height, params_.swing_height, curr_control_step_state.body_pos_in_world(0), curr_control_step_state.body_pos_in_world(1), body_yaw, curr_control_step_state.body_vel_in_world(0), curr_control_step_state.body_vel_in_world(1), curr_control_step_state.body_angvel_in_world(2), control_step_inputs_.body_vel_in_world_desired(0), control_step_inputs_.body_vel_in_world_desired(1), control_step_inputs_.body_angvel_in_world_desired(2), params_.foot_x_origins[i], params_.foot_y_origins[i], curr_control_step_state.swing_x0s.col(i), curr_control_step_state.swing_v0s.col(i));

        // Calculate the desired foot position in body frame
        foot_pos_in_body_desired.col(i) = curr_control_step_state.body_rot_in_world.inverse() * (foot_pos_in_world_desired - curr_control_step_state.body_pos_in_world);

        // Calculate the desired foot velocity in body frame
        foot_vel_in_body_desired.col(i) = curr_control_step_state.body_rot_in_world.inverse() * (foot_vel_in_world_desired - curr_control_step_state.body_vel_in_world - curr_control_step_state.body_angvel_in_world.cross(foot_pos_in_world_desired - curr_control_step_state.body_pos_in_world));

        // Calculate the desired joint velocities
        curr_control_step_state.joint_vel_desired.col(i) = jacobians[i].inverse() * foot_vel_in_body_desired.col(i);

        // Set joint kps and kds to swing values
        // Set joint feedforward torques to zero
        curr_control_step_state.joint_kp_desired.col(i) = Eigen::Vector3d::Ones() * params_.swing_joint_kp;
        curr_control_step_state.joint_kd_desired.col(i) = Eigen::Vector3d::Ones() * params_.swing_joint_kd;
        curr_control_step_state.joint_torque_desired.col(i) = Eigen::Vector3d::Zero();
      } else {
        // Leg is in stance phase
        // Determine joint feedforward torques from the balancing forces
        curr_control_step_state.joint_torque_desired.col(i) = jacobians[i].transpose() * curr_control_step_state.body_rot_in_world.inverse() * balancing_forces_in_world.col(i);

        // Set joint kps and kds to zero
        curr_control_step_state.joint_kp_desired.col(i) = Eigen::Vector3d::Zero();
        curr_control_step_state.joint_kd_desired.col(i) = Eigen::Vector3d::Zero();
      }
    }

    // Apply IK to obtain the desired joint positions
    curr_control_step_state.joint_pos_desired = four_legs_ik(foot_pos_in_body_desired, curr_control_step_state.joint_pos);
  }


  // Send the actuator commands
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      int joint_index = 3 * i + j;
      command_interfaces_map_.at(params_.joint_names[joint_index])
        .at("position")
        .get()
        .set_value(curr_control_step_state.joint_pos_desired(j, i));
      command_interfaces_map_.at(params_.joint_names[joint_index])
        .at("velocity")
        .get()
        .set_value(curr_control_step_state.joint_vel_desired(j, i));
      command_interfaces_map_.at(params_.joint_names[joint_index])
        .at("effort")
        .get()
        .set_value(curr_control_step_state.joint_torque_desired(j, i));
      command_interfaces_map_.at(params_.joint_names[joint_index])
        .at("kp")
        .get()
        .set_value(curr_control_step_state.joint_kp_desired(j, i));
      command_interfaces_map_.at(params_.joint_names[joint_index])
        .at("kd")
        .get()
        .set_value(curr_control_step_state.joint_kd_desired(j, i));
    }
  }

  // Update the previous control step state
  prev_control_step_state_ = curr_control_step_state;

  return controller_interface::return_type::OK;
}

}  // namespace heuristic_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(heuristic_controller::HeuristicController,
                       controller_interface::ControllerInterface)