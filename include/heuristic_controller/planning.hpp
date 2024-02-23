#include <Eigen/Dense>
#include <cmath>  // For sin(), cos()
#include <tuple>  // For std::tuple

std::tuple<double, double, double> getFutureBodyState(double t, double x0,
                                                      double y0, double theta0,
                                                      double vx, double vy,
                                                      double omega) {
  double x_t, y_t, theta_t, vx_t, vy_t;

  // Calculate theta(t)
  theta_t = theta0 + omega * t;

  // Handle the omega = 0 case separately to avoid division by zero
  if (std::abs(omega) < 1e-9) {  // Use a small threshold to consider as zero
    x_t = x0 + vx * t;           // Linear motion
    y_t = y0 + vy * t;
  } else {
    // General case for non-zero omega
    x_t = x0 + (vx * sin(omega * t + theta0) / omega) -
          (vy * (1 - cos(omega * t + theta0)) / omega);
    y_t = y0 + (vy * sin(omega * t + theta0) / omega) +
          (vx * (1 - cos(omega * t + theta0)) / omega);
  }

  return std::make_tuple(x_t, y_t, theta_t);
}

double getNextFootholdRelativeToBody(double stance_time, double body_height,
                                     double v0, double vf) {
  double omega = sqrt(9.81 / body_height);
  double position = -(vf - v0) /
                    (omega * tanh(omega * stance_time));
  return position;
}

std::tuple<double, double> evaluateSplineSegment(double t, double t0, double tf,
                                                 double x0, double v0,
                                                 double xf, double vf) {
  double t_ = (t - t0) / (tf - t0);
  t_ = std::max(0.0, std::min(1.0, t_));  // Clamp t_ to [0, 1]

  // Position
  double h00 = 2 * pow(t_, 3) - 3 * pow(t_, 2) + 1;
  double h10 = pow(t_, 3) - 2 * pow(t_, 2) + t_;
  double h01 = -2 * pow(t_, 3) + 3 * pow(t_, 2);
  double h11 = pow(t_, 3) - pow(t_, 2);
  double xt = h00 * x0 + h10 * (tf - t0) * v0 + h01 * xf + h11 * (tf - t0) * vf;

  // Velocity
  double h00_v = 6 * pow(t_, 2) - 6 * t_;
  double h10_v = 3 * pow(t_, 2) - 4 * t_ + 1;
  double h01_v = -6 * pow(t_, 2) + 6 * t_;
  double h11_v = 3 * pow(t_, 2) - 2 * t_;
  double vt =
      h00_v * x0 + h10_v * (tf - t0) * v0 + h01_v * xf + h11_v * (tf - t0) * vf;

  return std::make_tuple(xt, vt);
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d> evaluateSwingTrajectory(
    double t, double t0, double tf, double swing_height,
    const Eigen::Vector3d& p0, const Eigen::Vector3d& v0,
    const Eigen::Vector3d& pf, const Eigen::Vector3d& vf) {
  double tmid = (t0 + tf) / 2;
  Eigen::Vector3d pt;
  Eigen::Vector3d vt;

  // x and y: interpolate between x0, y0 and xf, yf
  for (int i = 0; i < 2; ++i) {  // For x and y
    auto [pt_val, vt_val] =
        evaluateSplineSegment(t, t0, tf, p0(i), v0(i), pf(i), vf(i));
    pt(i) = pt_val;
    vt(i) = vt_val;
  }

  // z: z0 to swing_height to zf
  if (t < tmid) {
    auto [pt_val, vt_val] =
        evaluateSplineSegment(t, t0, tmid, p0(2), v0(2), swing_height, 0);
    pt(2) = pt_val;
    vt(2) = vt_val;
  } else {
    auto [pt_val, vt_val] =
        evaluateSplineSegment(t, tmid, tf, swing_height, 0, pf(2), vf(2));
    pt(2) = pt_val;
    vt(2) = vt_val;
  }

  return std::make_tuple(pt, vt);
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d> calculateSwingFootPosVel(
    double t, double t0, double tf, double stance_time, double body_height,
    double swing_height, double x, double y, double yaw, double vx, double vy,
    double yaw_vel, double vx_des, double vy_des, double yaw_vel_des,
    double foot_x_offset, double foot_y_offset, const Eigen::Vector3d& foot_p0,
    const Eigen::Vector3d& foot_v0) {
  // Calculate the body state at the end of the swing phase
  auto [xf, yf, yawf] =
      getFutureBodyState(tf - t, x, y, yaw, vx, vy, yaw_vel);
  // Calculate the linear offset of the next foothold disregarding yaw
  double x_offset =
      getNextFootholdRelativeToBody(stance_time, body_height, vx, vx_des);
  double y_offset =
      getNextFootholdRelativeToBody(stance_time, body_height, vy, vy_des);
  // Apply the effect of yaw to the next foothold
  double foothold_yaw = yawf + 0.5 * stance_time * yaw_vel_des;
  double x_offset_rot =
      foot_x_offset * cos(foothold_yaw) - foot_y_offset * sin(foothold_yaw);
  double y_offset_rot =
      foot_x_offset * sin(foothold_yaw) + foot_y_offset * cos(foothold_yaw);
  // Calculate the next foothold position
  double foot_x_next = xf + x_offset + x_offset_rot;
  double foot_y_next = yf + y_offset + y_offset_rot;
  // Evaluate the swing trajectory at the current time
  Eigen::Vector3d foot_pf(foot_x_next, foot_y_next, foot_p0(2));
  Eigen::Vector3d foot_vf(0.0, 0.0, 0.0);
  auto [foot_pt, foot_vt] = evaluateSwingTrajectory(
      t, t0, tf, swing_height + foot_p0(2), foot_p0, foot_v0, foot_pf, foot_vf);
  return std::make_tuple(foot_pt, foot_vt);
}
