#include "mex.h"
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include "math.h"
#include <vector>
using namespace Eigen;


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  if(nrhs!=8 || nlhs > 2) {
    mexErrMsgIdAndTxt("Drake:recompVmex:BadInputs","Usage: [c,dc] = recompV(x,coeff_match,coeff_power,coeff_M,dcoeff_match,dcoeff_power,dcoeff_M,cnstr_normalizer)");
  }
  int nx = static_cast<int>(mxGetNumberOfElements(prhs[0]));
  Map<VectorXd> x(mxGetPr(prhs[0]),nx);
 
  assert(mxGetM(prhs[1]) == nx);
  assert(mxGetM(prhs[4]) == nx);

  double cnstr_normalizer = *mxGetPr(prhs[7]);

  Map<VectorXd> coeff_match(mxGetPr(prhs[1]),nx);
  Map<VectorXd> dcoeff_match(mxGetPr(prhs[4]),nx);
  VectorXd x_coeff(nx);
  VectorXd x_dcoeff(nx);
  for(int i = 0;i<nx;i++) {
    x_coeff(i) = x(static_cast<int>(coeff_match(i))-1);
    x_dcoeff(i) = x(static_cast<int>(dcoeff_match(i))-1);
  }

  int nc = mxGetM(prhs[3]);
  int coeff_power_nnz = mxGetNzmax(prhs[2]);
  int coeff_M_nnz = mxGetNzmax(prhs[3]);
  int dcoeff_power_nnz = mxGetNzmax(prhs[5]);
  int dcoeff_M_nnz = mxGetNzmax(prhs[6]);

  mwIndex* coeff_power_ir = mxGetIr(prhs[2]);
  mwIndex* coeff_power_jc = mxGetJc(prhs[2]);
  Map<VectorXd> coeff_power_val(mxGetPr(prhs[2]),coeff_power_nnz);
  std::vector<int> coeff_power_row;
  std::vector<int> coeff_power_col;
  coeff_power_row.reserve(coeff_power_nnz);
  coeff_power_col.reserve(coeff_power_nnz);
  for(int i = 0;i<mxGetN(prhs[2]);i++) {
    mwSize nrow = coeff_power_jc[i+1]-coeff_power_jc[i];
    for(int j = 0;j<nrow;j++) {
      coeff_power_row.push_back(*coeff_power_ir++);
      coeff_power_col.push_back(i);
    }
  }

  mwIndex* coeff_M_ir = mxGetIr(prhs[3]);
  mwIndex* coeff_M_jc = mxGetJc(prhs[3]);
  double* coeff_M_pr = mxGetPr(prhs[3]);
  const int coeff_M_cols = mxGetN(prhs[3]);
  SparseMatrix<double> coeff_M(nc,coeff_M_cols);
  std::vector<Triplet<double>> coeff_M_triplet;
  coeff_M_triplet.reserve(coeff_M_nnz);
  for(int i = 0;i<mxGetN(prhs[3]);i++) {
    mwSize nrow = coeff_M_jc[i+1]-coeff_M_jc[i];
    for(int j = 0;j<nrow;j++) {
      coeff_M_triplet.push_back(Triplet<double>(*coeff_M_ir++,i,*coeff_M_pr++)); 
    }
  }
  coeff_M.setFromTriplets(coeff_M_triplet.begin(),coeff_M_triplet.end());

  mwIndex* dcoeff_power_ir = mxGetIr(prhs[5]);
  mwIndex* dcoeff_power_jc = mxGetJc(prhs[5]);
  Map<VectorXd> dcoeff_power_val(mxGetPr(prhs[5]),dcoeff_power_nnz);
  std::vector<int> dcoeff_power_row;
  std::vector<int> dcoeff_power_col;
  dcoeff_power_row.reserve(dcoeff_power_nnz);
  dcoeff_power_col.reserve(dcoeff_power_nnz);
  for(int i = 0;i<mxGetN(prhs[5]);i++) {
    mwSize nrow = dcoeff_power_jc[i+1]-dcoeff_power_jc[i];
    for(int j = 0;j<nrow;j++) {
      dcoeff_power_row.push_back(*dcoeff_power_ir++);
      dcoeff_power_col.push_back(i);
    }
  }

 
  
  const int dcoeff_M_cols = mxGetN(prhs[6]);
  SparseMatrix<double> dcoeff_M(mxGetM(prhs[6]),dcoeff_M_cols);
  std::vector<Triplet<double>> dcoeff_M_triplet;
  dcoeff_M_triplet.reserve(dcoeff_M_nnz);
  mwIndex* dcoeff_M_ir = mxGetIr(prhs[6]);
  mwIndex* dcoeff_M_jc = mxGetJc(prhs[6]);
  double* dcoeff_M_pr = mxGetPr(prhs[6]);
  for(int i = 0;i<dcoeff_M_cols;i++){
    mwSize nrow = dcoeff_M_jc[i+1]-dcoeff_M_jc[i];
    for(int j = 0;j<nrow;j++){
      dcoeff_M_triplet.push_back(Triplet<double>(*dcoeff_M_ir++,i,*dcoeff_M_pr++));
    }
  }
  dcoeff_M.setFromTriplets(dcoeff_M_triplet.begin(),dcoeff_M_triplet.end());

  VectorXd coeff_prod = VectorXd::Ones(coeff_M_cols);
  for(int i=0;i<coeff_power_nnz;i++) {
    coeff_prod(coeff_power_row[i]) *= pow(x_coeff(coeff_power_col[i]),coeff_power_val(i));
  }
  VectorXd c = coeff_M*coeff_prod/cnstr_normalizer;

  VectorXd dcoeff_prod = VectorXd::Ones(dcoeff_M_cols);
  for(int i = 0;i<dcoeff_power_nnz;i++) {
    dcoeff_prod(dcoeff_power_row[i]) *= pow(x_dcoeff(dcoeff_power_col[i]),dcoeff_power_val(i));
  }
  MatrixXd dc = dcoeff_M*dcoeff_prod/cnstr_normalizer;

  plhs[0] = mxCreateDoubleMatrix(nc,1,mxREAL);
  memcpy(mxGetPr(plhs[0]),c.data(),sizeof(double)*nc);
  plhs[1] = mxCreateDoubleMatrix(nc,nx,mxREAL);
  memcpy(mxGetPr(plhs[1]),dc.data(),sizeof(double)*nc*nx);
} 
