#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimensions
  n_x_ = 5;

  // Augmented state dimensions
  n_aug_ = 7;

  // Sigma points spreading factor
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 15;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */

  // Set the initialization flag
  is_initialized_ = false;

  // Initialize the weights
  weights_ = VectorXd(2*n_aug_+1);
  weights_.fill(1/(2*(lambda_+n_aug_)));
  weights_(0) = lambda_/(lambda_+n_aug_);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Lidar state to measurement mapping matrix
  Lidar_H_ = MatrixXd(2, n_x_);
  Lidar_H_.fill(0.0);
  Lidar_H_(0, 0) = 1;
  Lidar_H_(1, 1) = 1;

  // Lidar measurement noise covariance matrix
  Lidar_R_ = MatrixXd(2, 2);
  Lidar_R_.fill(0.0);
  Lidar_R_(0, 0) = std_laspx_*std_laspx_;
  Lidar_R_(1, 1) = std_laspy_*std_laspy_;

  // set measurement dimension, radar can measure r, phi, and r_dot
  n_z_ = 3;

  // Radar measurement noise covariance matrix
  Radar_R_ = MatrixXd(n_z_,n_z_);
  Radar_R_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0,std_radrd_*std_radrd_;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if(is_initialized_){
    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
    Prediction(delta_t);

    time_us_ = meas_package.timestamp_;

    if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
      UpdateLidar(meas_package);
    } else if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
      UpdateRadar(meas_package);
    }
  }else{
    is_initialized_ = true;

    time_us_ = meas_package.timestamp_;

    x_.fill(0.0);
    if(meas_package.sensor_type_ == meas_package.LASER){
      x_(0) = meas_package.raw_measurements_(0); //Px
      x_(1) = meas_package.raw_measurements_(1); //Py

      P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
            0, std_laspy_*std_laspy_, 0, 0, 0,
            0, 0, 100, 0, 0,
            0, 0, 0, 40, 0,
            0, 0, 0, 0, 40;
    } else {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      x_(0) = rho*cos(phi); //Px
      x_(1) = rho*sin(phi); //Py

      P_ << (std_radr_+std_radphi_)*(std_radr_+std_radphi_), 0, 0, 0, 0,
            0, (std_radr_+std_radphi_)*(std_radr_+std_radphi_), 0, 0, 0,
            0, 0, 100, 0, 0,
            0, 0, 0, 40, 0,
            0, 0, 0, 0, 40;
    }
  }
}

void UKF::Prediction(double delta_t) {
  if(is_initialized_){
    // create sigma point matrix
    MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);

    GenerateSigmaPoints(Xsig_aug);
    
    SigmaPointsPrediction(Xsig_aug, delta_t);

    PredictMeanAndCovariance();
  }
}

void UKF::GenerateSigmaPoints(MatrixXd &Xsig_aug) {
  // create augmented state mean vector
  VectorXd x_aug(n_aug_);
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented state covariance matrix
  MatrixXd P_aug(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // create square root matrix
  MatrixXd P_aug_root = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++) {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * P_aug_root.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * P_aug_root.col(i);
  }
}

void UKF::SigmaPointsPrediction(MatrixXd &Xsig_aug, double &delta_t){
  // predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++) {
    // extract values for better readability
    double px = Xsig_aug(0,i);
    double py = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // avoid division by zero
    if(fabs(yawd) < 0.001){
      Xsig_pred_(0, i) = px + (v * cos(yaw) * delta_t) + (0.5 * delta_t*delta_t*cos(yaw)*nu_a);  
      Xsig_pred_(1, i) = py + (v * sin(yaw) * delta_t) + (0.5 * delta_t*delta_t*sin(yaw)*nu_a);  
    }else{
      Xsig_pred_(0, i) = px + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw)) + (0.5 * delta_t*delta_t*cos(yaw)*nu_a);  
      Xsig_pred_(1, i) = py + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) ) + (0.5 * delta_t*delta_t*sin(yaw)*nu_a);   
    }
    Xsig_pred_(2, i) = v + delta_t * nu_a;
    Xsig_pred_(3, i) = yaw + 0.5 * delta_t * delta_t * nu_yawdd;
    Xsig_pred_(4, i) = yawd + delta_t * nu_yawdd;
  }
}

void UKF::PredictMeanAndCovariance(void) {
  // predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  // iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  VectorXd z_pred = Lidar_H_ * x_;
  VectorXd y = meas_package.raw_measurements_ - z_pred;
  MatrixXd Ht = Lidar_H_.transpose();
  MatrixXd S = Lidar_H_ * P_ * Ht + Lidar_R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  P_ = (I - K * Lidar_H_) * P_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // create matrix for sigma points in measurement space
  MatrixXd Zsig(n_z_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred(n_z_);
  
  // measurement covariance matrix S
  MatrixXd S(n_z_, n_z_);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
    Zsig(1,i) = atan2(p_y,p_x);                                // phi
    Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
  }

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S = S + Radar_R_;

  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z_);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}