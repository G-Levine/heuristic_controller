#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "eiquadprog/eiquadprog-rt.hpp"

Eigen::Matrix<double, 3, 4> balancingLinearSolve(const Eigen::Matrix<double, 3, 4>& r,
                                                   const Eigen::Vector4i& s,
                                                   const Eigen::Vector3d& force_desired,
                                                   const Eigen::Vector3d& torque_desired,
                                                   const Eigen::Matrix<double, 3, 4>& f_previous,
                                                   double minZ, double maxZ, double frictionCoefficient,
                                                   double weight_f = 1.0, double weight_tau = 1.0, double weight_diff = 0.1) {
    // Check if no legs are in contact
    if (s.sum() == 0) {
        return Eigen::Matrix<double, 3, 4>::Zero();
    }

    // System matrix A and vector b
    Eigen::Matrix<double, 18, 12> A = Eigen::Matrix<double, 18, 12>::Zero(); // Adjusted for 4 legs, 3 dimensions each
    Eigen::Matrix<double, 18, 1> b;
    Eigen::Matrix<double, 12, 1> f_previous_flat;
    for (int i = 0; i < 4; ++i) {
        f_previous_flat.block<3, 1>(3 * i, 0) = f_previous.col(i);
    }
    b << weight_f * force_desired, weight_tau * torque_desired, weight_diff * f_previous_flat;

    // Populate A based on contact status
    for (int i = 0; i < 4; ++i) {

        if (s(i) == 1) {
            // Force part
            A.block<3, 3>(0, 3 * i) = weight_f * Eigen::Matrix3d::Identity();
            
            // Torque part
            Eigen::Matrix3d r_skew;
            r_skew << 0, -r.col(i).z(), r.col(i).y(),
                      r.col(i).z(), 0, -r.col(i).x(),
                      -r.col(i).y(), r.col(i).x(), 0;
            A.block<3, 3>(3, 3 * i) = weight_tau * r_skew;
        }
    }
    A.block<12, 12>(6, 0) = weight_diff * Eigen::Matrix<double, 12, 12>::Identity();

    // Solve the least squares problem using QR decomposition, suitable for fixed-size matrices
    // Eigen::Matrix<double, 12, 1> f = A.householderQr().solve(b);

    // Solve the least squares problem using SVD decomposition, suitable for potentially ill-conditioned matrices
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix<double, 12, 1> f = svd.solve(b);
    // Eigen::Matrix<double, 12, 1> f = A.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(b)

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

Eigen::Matrix3d crossProductMatrix(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return m;
}

Eigen::Matrix<double, 3, 4> balancingQP(const Eigen::Matrix<double, 3, 4>& r,
                                                   const Eigen::Vector4i& s,
                                                   const Eigen::Vector3d& f_desired,
                                                   const Eigen::Vector3d& tau_desired,
                                                   const Eigen::Matrix<double, 3, 4>& f_previous,
                                                   double f_z_min, double f_z_max, double mu,
                                                   double weight_f = 1.0, double weight_tau = 1.0, double weight_diff = 0.1, double weight_mag = 0.01) {
    // Check if no legs are in contact
    if (s.sum() == 0) {
        return Eigen::Matrix<double, 3, 4>::Zero();
    }

    Eigen::Matrix<double, 12, 12> Q;
    Eigen::Matrix<double, 12, 1> c;
    Eigen::Matrix<double, 24, 12> A;
    Eigen::Matrix<double, 24, 1> b;
    Eigen::Matrix<double, 12, 1> f;

    Q.setZero();
    c.setZero();
    A.setZero();
    b.setZero();
    f.setZero();

    // Definition of the cost function:
    // Force part: (I_block f - f_desired)^T (I_block f - f_desired) -> 0.5 f^T I_block^T I_block f - f_desired^T f
    // -> Q = weight_f * I_block^T I_block, c = -weight_f * I_block^T f_desired
    Eigen::Matrix<double, 3, 12> I_block;
    for (int i = 0; i < 4; ++i) {
        I_block.block<3, 3>(0, 3 * i) = Eigen::Matrix3d::Identity();
    }
    Q += weight_f * I_block.transpose() * I_block;
    c += -weight_f * I_block.transpose() * f_desired;

    // Torque part: (r_x f - tau_desired)^T (r_x f - tau_desired) -> 0.5 f^T (r_x^T r_x) f - (r_x^T tau_desired)^T f
    // -> Q += weight_tau * r_x^T r_x, c += -weight_tau * r_x^T tau_desired
    Eigen::Matrix<double, 3, 12> r_x;
    for (int i = 0; i < 4; ++i) {
        r_x.block<3, 3>(0, 3 * i) = crossProductMatrix(r.col(i));
    }
    Q += weight_tau * r_x.transpose() * r_x;
    c += -weight_tau * r_x.transpose() * tau_desired;

    // Force difference penalty: (f - f_previous)^T (f - f_previous) -> 0.5 f^T I f - f_previous^T f
    // -> Q += weight_diff * I, c += -weight_diff * f_previous
    Eigen::Matrix<double, 12, 1> f_previous_flat;
    for (int i = 0; i < 4; ++i) {
        f_previous_flat.block<3, 1>(3 * i, 0) = f_previous.col(i);
    }
    Q += weight_diff * Eigen::Matrix<double, 12, 12>::Identity();
    c += -weight_diff * f_previous_flat;

    // Force magnitude penalty: f^T f -> 0.5 f^T I f
    // -> Q += weight_mag * I, c += 0
    Q += weight_mag * Eigen::Matrix<double, 12, 12>::Identity();

    Eigen::Vector3d e_x(1, 0, 0);
    Eigen::Vector3d e_y(0, 1, 0);
    Eigen::Vector3d e_z(0, 0, 1);

    for (int i = 0; i < 4; ++i) {
        // Definition of the constraints:
        // Z component positive: f_z <= f_z_max -> A = e_z^T, b = f_z_max
        A.block<1, 3>(6 * i, 3 * i) = e_z.transpose();
        if (s(i) == 1) {
            b(6 * i) = f_z_max;
        } else {
            b(6 * i) = 0.0;
        }
        
        // Z component negative: -f_z <= -f_z_min -> A = -e_z^T, b = -f_z_min
        A.block<1, 3>(6 * i + 1, 3 * i) = -e_z.transpose();
        if (s(i) == 1) {
            b(6 * i + 1) = -f_z_min;
        } else {
            b(6 * i + 1) = 0.0;
        }

        // X component positive: f_x <= mu f_z -> A = (e_x - mu e_z)^T, b = 0
        A.block<1, 3>(6 * i + 2, 3 * i) = (e_x - mu * e_z).transpose();
        b(6 * i + 2) = 0;

        // X component negative: -f_x <= mu f_z -> A = (-e_x - mu e_z)^T, b = 0
        A.block<1, 3>(6 * i + 3, 3 * i) = (-e_x - mu * e_z).transpose();
        b(6 * i + 3) = 0;

        // Y component positive: f_y <= mu f_z -> A = (e_y - mu e_z)^T, b = 0
        A.block<1, 3>(6 * i + 4, 3 * i) = (e_y - mu * e_z).transpose();
        b(6 * i + 4) = 0;

        // Y component negative: -f_y <= mu f_z -> A = (-e_y - mu e_z)^T, b = 0
        A.block<1, 3>(6 * i + 5, 3 * i) = (-e_y - mu * e_z).transpose();
        b(6 * i + 5) = 0;
    }
    
    // Solve the QP
    // EigenQP::quadprog(Q, c, A, b, f);
    eiquadprog::solvers::RtEiquadprog<12, 0, 24> qp;
    RtMatrixX<0, 12>::d Aeq;
    RtVectorX<0>::d beq;
    eiquadprog::solvers::RtEiquadprog_status status =
      qp.solve_quadprog(Q, c, Aeq, beq, -A, b, f);

    // std::cout << Q << std::endl;
    // std::cout << c << std::endl;
    // std::cout << A << std::endl;
    // std::cout << b << std::endl;
    // std::cout << f << std::endl;

    // Print the status of the QP solver
    if (status != eiquadprog::solvers::RT_EIQUADPROG_OPTIMAL) {
        std::cerr << "QP solver failed with status " << status << std::endl;
    }

    // Reshape the result to a 3x4 matrix and flip the signs to output the forces applied by the feet
    Eigen::Matrix<double, 3, 4> result;
    for (int i = 0; i < 4; ++i) {
        result.col(i) = f.block<3, 1>(3 * i, 0);
    }
    return -result;
}
