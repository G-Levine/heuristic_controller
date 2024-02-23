#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <cmath>
#include <iostream>
#include <algorithm>

class RobotConfig {
public:
    Eigen::Matrix<double, 2, 4> LEG_ORIGINS; // 2D positions for each of the 4 legs (x, y positions)
    Eigen::Vector4d ABDUCTION_OFFSETS; // Abduction offsets for each leg
    double LEG_L1_X; // Length of the first link in x direction
    double LEG_L1_Z; // Length of the first link in z direction
    double LEG_L2;   // Length of the second link (lower leg)
    Eigen::Matrix<double, 3, 4> MOTOR_DIRECTIONS; // Direction multipliers for motors

    RobotConfig() {
        LEG_ORIGINS << 0.075, 0.075, -0.075, -0.075,
                       -0.0835, 0.0835, -0.0725, 0.0725;
        
        ABDUCTION_OFFSETS << -0.018, 0.018, -0.018, 0.018;

        LEG_L1_X = 0.07;
        LEG_L1_Z = 0.05;
        LEG_L2 = 0.088;

        MOTOR_DIRECTIONS << -1, 1, -1, 1,
                             -1, -1, -1, -1,
                             -1, 1, -1, 1;
    }
};

class LegConfig {
public:
    double motor_x;  // x-axis measure of motor to motor distance
    double motor_y;  // y-axis measure of motor to motor distance
    double abduction_offset;  // abduction offset (in +Ny>)
    double link_2_x;  // component of Bo_r_Co vector in -Bx> direction
    double link_2_z;  // component of Bo_r_Co vector in -Bz> direction
    double link_3;  // length (+) of lower leg

    LegConfig(double mx, double my, double ao, double l2x, double l2z, double l3)
        : motor_x(mx), motor_y(my), abduction_offset(ao), link_2_x(l2x), link_2_z(l2z), link_3(l3) {}
};

RobotConfig defaultRobotConfig() {
    return RobotConfig();
}

Eigen::Vector3d leg_fk(double qA, double qB, double qC, const LegConfig& config) {
    double MX = config.motor_x;
    double MY = config.motor_y;
    double LA = config.abduction_offset;
    double LX = config.link_2_x;
    double LZ = config.link_2_z;
    double L3 = config.link_3;

    double sA = sin(qA);
    double cA = cos(qA);
    double sB = sin(qB);
    double cB = cos(qB);
    double sC = sin(qC);
    double cC = cos(qC);

    double Fx = MX - LX * cA - LZ * sA * cB - L3 * (sC * cA + sA * cB * cC);
    double Fy = LZ * sB + L3 * sB * cC + LA + MY;
    double Fz = LX * sA + L3 * (sA * sC - cA * cB * cC) - LZ * cA * cB;
    return Eigen::Vector3d(Fx, Fy, Fz);
}

Eigen::Matrix3d leg_jacobian(double qA, double qB, double qC, const LegConfig& config) {
    Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
    double LX = config.link_2_x;
    double LZ = config.link_2_z;
    double L3 = config.link_3;
    double sA = sin(qA);
    double cA = cos(qA);
    double sB = sin(qB);
    double cB = cos(qB);
    double sC = sin(qC);
    double cC = cos(qC);

    result(0, 0) = LX * sA + L3 * (sA * sC - cA * cB * cC) - LZ * cA * cB;
    result(0, 1) = sA * sB * (LZ + L3 * cC);
    result(0, 2) = -L3 * (cA * cC - sA * sC * cB);
    result(1, 0) = 0;
    result(1, 1) = cB * (LZ + L3 * cC);
    result(1, 2) = -L3 * sB * sC;
    result(2, 0) = LX * cA + LZ * sA * cB + L3 * (sC * cA + sA * cB * cC);
    result(2, 1) = sB * cA * (LZ + L3 * cC);
    result(2, 2) = L3 * (sA * cC + sC * cA * cB);

    return result;
}

Eigen::Vector3d leg_ik(const Eigen::Vector3d& target_pos, const LegConfig& config, const Eigen::Vector3d& initial_guess = Eigen::Vector3d::Zero(), double alpha = 1.0) {
    Eigen::Vector3d guess = initial_guess;
    for (int i = 0; i < 20; ++i) {
        Eigen::Matrix3d jacobian = leg_jacobian(guess(0), guess(1), guess(2), config);
        Eigen::Vector3d current_pos = leg_fk(guess(0), guess(1), guess(2), config);
        Eigen::Vector3d error = current_pos - target_pos;

        Eigen::Matrix3d pseudo_inverse = jacobian.completeOrthogonalDecomposition().pseudoInverse();

        Eigen::Vector3d step = -alpha * pseudo_inverse * error;

        // Prevent big solver steps
        for (int j = 0; j < 3; ++j) {
            step(j) = std::max(-1.0, std::min(1.0, step(j)));
        }

        guess += step;

        if (error.norm() < 1e-6) {
            break;
        }
    }

    // Constrain to -pi to pi
    for (int i = 0; i < 3; ++i) {
        guess(i) = std::fmod(guess(i) + M_PI, 2 * M_PI) - M_PI;
    }

    return guess;
}

Eigen::Matrix<double, 3, 4> four_legs_fk(const Eigen::Matrix<double, 3, 4>& alpha) {
    Eigen::Matrix<double, 3, 4> r_body_foot = Eigen::Matrix<double, 3, 4>::Zero();
    RobotConfig config = defaultRobotConfig();

    for (int i = 0; i < 4; ++i) {
        LegConfig leg_config(
            config.LEG_ORIGINS(0, i),
            config.LEG_ORIGINS(1, i),
            config.ABDUCTION_OFFSETS[i],
            config.LEG_L1_X,
            config.LEG_L1_Z,
            config.LEG_L2
        );

        r_body_foot.col(i) = leg_fk(alpha(0, i) * config.MOTOR_DIRECTIONS(0, i), alpha(1, i) * config.MOTOR_DIRECTIONS(1, i), alpha(2, i) * config.MOTOR_DIRECTIONS(2, i) - 45.0 * M_PI / 180.0, leg_config);
    }

    return r_body_foot;
}

Eigen::Matrix<double, 3, 4> four_legs_ik(const Eigen::Matrix<double, 3, 4>& r_body_foot, const Eigen::Matrix<double, 3, 4>& initial_guess = Eigen::Matrix<double, 3, 4>::Zero()) {
    Eigen::Matrix<double, 3, 4> alpha = Eigen::Matrix<double, 3, 4>::Zero();
    RobotConfig config = defaultRobotConfig();

    for (int i = 0; i < 4; ++i) {
        LegConfig leg_config(
            config.LEG_ORIGINS(0, i),
            config.LEG_ORIGINS(1, i),
            config.ABDUCTION_OFFSETS[i],
            config.LEG_L1_X,
            config.LEG_L1_Z,
            config.LEG_L2
        );

        Eigen::Vector3d target_pos = r_body_foot.col(i);
        Eigen::Vector3d guess = initial_guess.col(i).cwiseProduct(config.MOTOR_DIRECTIONS.col(i)); // Apply motor directions to initial guess

        Eigen::Vector3d leg_alpha = leg_ik(target_pos, leg_config, guess);

        // Adjust for the mechanical offset, if necessary
        leg_alpha(2) += 45.0 * M_PI / 180.0;

        alpha.col(i) = leg_alpha.cwiseProduct(config.MOTOR_DIRECTIONS.col(i)); // Apply motor directions to results
    }

    return alpha;
}

std::array<Eigen::Matrix3d, 4> four_legs_jacobian(const Eigen::Matrix<double, 3, 4>& alpha) {
    std::array<Eigen::Matrix3d, 4> jacobians;
    RobotConfig config = defaultRobotConfig();

    for (int i = 0; i < 4; ++i) {
        LegConfig leg_config(
            config.LEG_ORIGINS(0, i),
            config.LEG_ORIGINS(1, i),
            config.ABDUCTION_OFFSETS[i],
            config.LEG_L1_X,
            config.LEG_L1_Z,
            config.LEG_L2
        );

        Eigen::Matrix3d J_leg = leg_jacobian(alpha(0, i) * config.MOTOR_DIRECTIONS(0, i), alpha(1, i) * config.MOTOR_DIRECTIONS(1, i), alpha(2, i) * config.MOTOR_DIRECTIONS(2, i) - 45.0 * M_PI / 180.0, leg_config);
        for (int j = 0; j < 3; ++j) {
            jacobians[i].col(j) = J_leg.col(j) * config.MOTOR_DIRECTIONS(j, i); // Apply motor directions to results
        }
    }

    return jacobians;
}

Eigen::Matrix<double, 3, 4> balancingQP(const Eigen::Matrix<double, 3, 4>& r,
                                                   const Eigen::Vector4i& s,
                                                   const Eigen::Vector3d& force_desired,
                                                   const Eigen::Vector3d& torque_desired,
                                                   double minZ, double maxZ, double frictionCoefficient) {
    // Check if no legs are in contact
    if (s.sum() == 0) {
        return Eigen::Matrix<double, 3, 4>::Zero();
    }

    // System matrix A and vector b
    Eigen::Matrix<double, 6, 12> A = Eigen::Matrix<double, 6, 12>::Zero(); // Adjusted for 4 legs, 3 dimensions each
    Eigen::Matrix<double, 6, 1> b;
    b << force_desired, torque_desired;

    // Populate A based on contact status
    for (int i = 0; i < 4; ++i) {
        if (s(i) == 1) {
            // Force part
            A.block<3, 3>(0, 3 * i) = Eigen::Matrix3d::Identity();
            
            // Torque part
            Eigen::Matrix3d r_skew;
            r_skew << 0, -r.col(i).z(), r.col(i).y(),
                      r.col(i).z(), 0, -r.col(i).x(),
                      -r.col(i).y(), r.col(i).x(), 0;
            A.block<3, 3>(3, 3 * i) = r_skew;
        }
    }

    // Solve the least squares problem using QR decomposition, suitable for fixed-size matrices
    // Eigen::Matrix<double, 12, 1> f = A.householderQr().solve(b);

    // Solve the least squares problem using SVD decomposition, suitable for potentially ill-conditioned matrices
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix<double, 12, 1> f = svd.solve(b);

    // Apply Z Constraints
    for (int i = 0; i < 4; ++i) {
        f(3*i + 2) = std::min(std::max(f(3*i + 2), minZ), maxZ); // Clamp z-component
    }

    // Apply Friction Cone Constraints
    for (int i = 0; i < 4; ++i) {
        double fz = f(3*i + 2); // Z component of the force
        double maxLateralForce = fz * frictionCoefficient;

        // Calculate the magnitude of the lateral force
        double lateralForceMagnitude = sqrt(pow(f(3*i), 2) + pow(f(3*i + 1), 2));
        
        // If the lateral force exceeds the maximum allowed by the friction cone, scale it down
        if (lateralForceMagnitude > maxLateralForce) {
            double scalingFactor = maxLateralForce / lateralForceMagnitude;
            f(3*i) *= scalingFactor; // Scale x-component
            f(3*i + 1) *= scalingFactor; // Scale y-component
        }
    }

    // Reshape the result to a 3x4 matrix
    Eigen::Matrix<double, 3, 4> result;
    for (int i = 0; i < 4; ++i) {
        result.col(i) = f.block<3, 1>(3 * i, 0);
    }

    // Flip the signs to output the forces applied by the feet
    return -result;
}

Eigen::Vector3d calculateAverageContactValue(const Eigen::Matrix<double, 3, 4>& positions, const Eigen::Vector4i& contact_states) {
    Eigen::Vector3d averagePosition = Eigen::Vector3d::Zero();
    int contactCount = 0;
    for (int i = 0; i < 4; ++i) {
        if (contact_states(i)) {
            averagePosition += positions.col(i);
            contactCount++;
        }
    }
    if (contactCount > 0) {
        averagePosition /= contactCount;
    } else {
        averagePosition /= 4.0;
    }
    return averagePosition;
}

float extractYawFromQuaternion(const Eigen::Quaterniond& quat) {
    // Convert the quaternion to Euler angles (yaw, pitch, roll)
    Eigen::Vector3d euler = quat.toRotationMatrix().eulerAngles(2, 1, 0); // ZYX convention
    return euler[0]; // First component is yaw in ZYX convention
}

Eigen::Quaterniond extractYawQuaternion(const Eigen::Quaterniond& originalQuat) {
    // Extract the yaw component
    double yaw = extractYawFromQuaternion(originalQuat);

    // Create a new quaternion with only the yaw component
    Eigen::AngleAxisd yawRotation(yaw, Eigen::Vector3d::UnitZ()); // Rotation around Z-axis
    Eigen::Quaterniond yawQuat(yawRotation);

    return yawQuat;
}