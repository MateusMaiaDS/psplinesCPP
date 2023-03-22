#include<RcppArmadillo.h>
#include<vector>
// Creating the struct
struct modelParam;

struct modelParam{

  arma::mat y;
  arma::mat B_train;

  // BART prior param specification
  int p;
  double tau_b;
  double tau_b_intercept;
  double tau;
  double a_tau;
  double d_tau;
  double nu;
  double delta;
  double a_delta;
  double d_delta;
  arma::mat P;

  // MCMC spec.
  int n_mcmc;
  int n_burn;


  // Objects from the sampler calculator
  arma::mat Gamma_inv;
  arma::mat bt_ones;
  arma::mat btr;
  double s_b_0;

  // Defining the constructor for the model param
  modelParam(arma::mat y_,
             arma::mat B_train_,
             double tau_b_,
             double tau_b_intercept_,
             double tau_,
             double a_tau_,
             double d_tau_,
             double nu_,
             double delta_,
             double a_delta_,
             double d_delta_,
             int n_mcmc_,
             int n_burn_);

};
