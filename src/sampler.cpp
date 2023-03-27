#include "psplines_sampler.h"
#include <random>
#include <Rcpp.h>
using namespace std;

// Building the constructor
modelParam::modelParam(arma::mat B_train_,
                       arma::mat y_,
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
                       int n_burn_){

  // Create a matrix of ones
  arma::vec ones(y_.n_rows,arma::fill::ones);
  bt_ones = B_train_.t()*ones;
  btr = B_train_.t()*y_;

  // Filling the parameters
  y = y_;
  B_train = B_train_;
  tau_b = tau_b_;
  tau = tau_;
  a_tau = a_tau_;
  d_tau = d_tau_;
  nu = nu_;
  delta = delta_;
  a_delta = a_delta_;
  d_delta = d_delta_;
  n_mcmc = n_mcmc_;
  n_burn = n_burn_;
  p = B_train_.n_cols;


}

// // Create a function to generate matrix D (for penalisation)
arma::mat D(modelParam data){

        // Creating the matrix elements
        arma::mat D_m((data.p-2),data.p,arma::fill::zeros);

        for(int i=0;i<(data.p-2);i++){
                D_m(i,i) = 1;
                D_m(i,i+1) = -2;
                D_m(i,i+2) = 1;
        }

        return D_m;
}

// // Create a function to generate matrix D (for penalisation)
arma::mat D_diag(modelParam data){

  // Creating the matrix elements
  arma::mat D_m((data.p),data.p,arma::fill::zeros);

  for(int i=0;i<(data.p);i++){
    D_m(i,i) = 1;

  }

  return D_m;
}

// // Create a function to generate matrix D (for penalisation)
arma::mat D_first(modelParam data){

  // Creating the matrix elements
  arma::mat D_m((data.p-1),data.p,arma::fill::zeros);

  for(int i=0;i<(data.p-1);i++){
    D_m(i,i) = -1;
    D_m(i,i+1) = 1;
  }

  return D_m;
}

// Building the beta sampler
void beta_sampler(arma::vec& betas,
                  double& beta_0,
                  modelParam& data){


  // Updating data s_b_0
  data.s_b_0 = (data.y.n_rows+(data.tau_b_intercept/data.tau));

  // Calculating Gamma Inv
  // data.Gamma_inv = inv(data.B_train.t()*data.B_train + (data.tau_b/data.tau)*data.P - (1/data.s_b_0)*(data.bt_ones*data.bt_ones.t()));
  data.Gamma_inv = inv(data.B_train.t()*data.B_train + (data.tau_b/data.tau)*data.P );

  // Ones aux
  arma::mat ones(data.B_train.n_rows,1,arma::fill::ones);
  // Calculating mean and variance
  arma::mat beta_mean = data.Gamma_inv*(data.btr-beta_0*data.B_train.t()*ones);
  arma::mat beta_cov  = (1/data.tau)*data.Gamma_inv;

  arma::mat sample = arma::randn<arma::mat>(data.Gamma_inv.n_cols);
  betas = arma::chol(beta_cov,"lower")*sample + beta_mean;


  return;
}

// Building the beta_0 sampler
void beta_0_sampler(arma::vec& betas,
                    double& beta_0,
                    modelParam& data){

  // Calculating the mean
  arma::vec mean_aux = betas.t()*data.bt_ones;
  double beta_0_mean = (1/data.s_b_0)*(sum(data.y.col(0))-mean_aux(0));
  double beta_0_sd = sqrt(1/(data.tau*data.s_b_0));

  beta_0 = arma::randn()*beta_0_sd + beta_0_mean;

  return;
}


// Building the \tau_b sampler
void tau_b_sampler(arma::vec& betas,
                   modelParam& data){

  // Calculating the shape and rate parameter
  double tau_b_shape = 0.5*data.p+0.5*data.nu;
  arma::vec rate_aux = betas.t()*data.P*betas;
  double tau_b_rate = 0.5*rate_aux(0) + 0.5*data.delta*data.nu;

  data.tau_b = R::rgamma(tau_b_shape,1/tau_b_rate);

  return;
}

// Updating delta
void delta_sampler(modelParam& data){

  // Calculating shape and rate parameter
  double delta_shape = 0.5*data.nu+data.a_delta;
  double delta_rate = 0.5*data.nu*data.tau_b + data.d_delta;

  data.delta = R::rgamma(delta_shape,1/delta_rate);

  return;
}


// Updating the tau parameter
void tau_sampler(modelParam& data,
                 arma::vec& y_hat){

  double tau_res_sq_sum = dot((y_hat-data.y),(y_hat-data.y));

  data.tau = R::rgamma((0.5*data.y.n_rows+data.a_tau),1/(0.5*tau_res_sq_sum+data.d_tau));

  return;
}

// Generating the sample code for the sampler
//[[Rcpp::export]]
Rcpp::List sp_sampler(arma::mat B_train,
                     arma::mat y,
                     double tau_b,
                     double tau_b_intercept,
                     double tau,
                     double a_tau,
                     double d_tau,
                     double nu,
                     double delta,
                     double a_delta,
                     double d_delta,
                     int n_mcmc,
                     int n_burn){


    // cout << "Error 0" << endl;

    // Initalising the data object
    modelParam data(    B_train,
                        y,
                        tau_b,
                        tau_b_intercept,
                        tau,
                        a_tau,
                        d_tau,
                        nu,
                        delta,
                        a_delta,
                        d_delta,
                        n_mcmc,
                        n_burn);

    // Generating the P matrix
    arma::mat D_m = D_diag(data);
    // arma::mat D_m = D_first(data);

    data.P = D_m.t()*D_m;

    // Initializing the vector of betas
    // cout << "Error 1" << endl;
    arma::vec betas(data.p, arma::fill::ones);
    double beta_0 = 0;
    arma::vec y_hat(data.y.n_rows,arma::fill::zeros);

    // Storing the posteriors
    int n_post = n_mcmc-n_burn;
    arma::mat beta_post(n_post,data.p,arma::fill::ones);
    arma::vec beta_0_post(n_post,arma::fill::ones);
    arma::mat y_hat_post(n_post,data.y.n_rows,arma::fill::ones);

    arma::vec tau_post(n_post,arma::fill::ones);
    arma::vec tau_b_post(n_post,arma::fill::ones);
    arma::vec delta_post(n_post,arma::fill::ones);
    int post_iter = 0;


    // Initializing the sampling processes
    for(int i = 0; i < data.n_mcmc; i++){

      // cout << "Beta error" << endl;
      beta_sampler(betas,beta_0, data);
      // cout << "Beta_0 error" << endl;
      beta_0_sampler(betas, beta_0, data);
      // cout << "Tau_b error" << endl;
      tau_b_sampler(betas,data);
      // cout << "Delta error" << endl;
      delta_sampler(data);

      // Calculating the predictions
      // cout << "Y_hat error" << endl;
      y_hat = data.B_train*betas + beta_0;

      // cout << "Tau sampler error" << endl;
      tau_sampler(data,y_hat);


      // cout << " No error" << endl;
      // Iterating and storing the posterior samples
      if(i >= data.n_burn){
        beta_post.row(post_iter) = betas.t();
        beta_0_post(post_iter) = beta_0;
        y_hat_post.row(post_iter) = y_hat.t();
        tau_b_post(post_iter) = data.tau_b;
        delta_post(post_iter) = data.delta;
        tau_post(post_iter) = data.tau;
        post_iter++;
      }


    }

    return Rcpp::List::create(beta_post,
                              beta_0_post,
                              y_hat_post,
                              tau_b_post,
                              delta_post,
                              tau_post);
}




