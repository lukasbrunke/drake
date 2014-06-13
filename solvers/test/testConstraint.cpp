#include "../Constraint.h"
#include <iostream>

using namespace std;
using namespace Eigen;
// implement a nonlinear constraint
// y(0) = x(0)^2+2*x(0)*x(1)+x(2)^2
// y(1) = x(1)*x(2)*x(3)
// y(2) = 5*x(0)*x(2) 
// y(3) = 10*x(1);
// y(4) = 0;
void cnstr1_eval(const ConstraintInput &in, VectorXd &y, MatrixXd &dy)
{
  y.resize(5);
  dy = Matrix<double,5,4>::Zero();
  y(0) = in.x(0)*in.x(0)+2*in.x(0)*in.x(1)+in.x(2)*in.x(2);
  y(1) = in.x(1)*in.x(2)*in.x(3);
  y(2) = 5*in.x(0)*in.x(2);
  y(3) = 10*in.x(1);
  y(4) = 0;
  dy(0,0) = 2*in.x(0)+2*in.x(1);
  dy(0,1) = 2*in.x(0);
  dy(0,2) = 2*in.x(2);
  dy(1,1) = in.x(2)*in.x(3);
  dy(1,2) = in.x(1)*in.x(3);
  dy(1,3) = in.x(1)*in.x(2);
  dy(2,0) = 5*in.x(2);
  dy(2,2) = 5*in.x(0);
  dy(3,1) = 10;
}

int main()
{
  cout<<"test NonlinearConstraint"<<endl;
  Matrix<double,5,1> cnstr1_lb;
  Matrix<double,5,1> cnstr1_ub;
  cnstr1_lb<<0.0,0.2,1.0,0.1,2;
  cnstr1_ub<<0.0,0.5,1.0,0.1,8;
  NonlinearConstraint* cnstr1 = new NonlinearConstraint(cnstr1_lb,cnstr1_ub,4);
  VectorXi ceq_idx;
  VectorXi cin_idx;
  cnstr1->getCeqIdx(ceq_idx);
  cnstr1->getCinIdx(cin_idx);
  if(ceq_idx(0) != 0 || ceq_idx(1) != 2 || ceq_idx(2) != 3 || ceq_idx.rows() != 3)
  {
    cerr<<"The indicies of equality constraints are wrong"<<endl;
  }
  else if(cin_idx(0) != 1 || cin_idx(1) != 4 || cin_idx.rows() != 2)
  {
    cerr<<"The indices of inequality constraints are wrong"<<endl;
  }
  cnstr1->setGevalHandle(cnstr1_eval);
  VectorXi iCfun(9);
  VectorXi jCvar(9);
  iCfun<< 0,0,0,1,1,1,2,2,3;
  jCvar<< 0,1,2,1,2,3,0,2,1;
  cnstr1->setSparseStructure(iCfun,jCvar); 
  Vector4d x;
  x<<-1,1,2,3;
  MatrixXd y1;
  VectorXd y2;
  MatrixXd dy;
  cnstr1->eval(ConstraintInput(x),y1);
  cnstr1->geval(ConstraintInput(x),y2,dy);
  cnstr1->checkGradient(1E-4,ConstraintInput(Vector4d::Random()));
  VectorXd lb,ub;
  cnstr1->getBounds(lb,ub);
  for(int i = 0;i<5;i++)
  {
    if(abs(lb(i)-cnstr1_lb(i))>1E-10 || abs(ub(i)-cnstr1_ub(i))>1E-10)
    {
      cerr<<"constraint lower bound or upper bound is incorrect in row "<<i<<endl;
    }
  }
  vector<string> name;
  name.resize(4);
  name.at(0) = string("y0");
  name.at(1) = string("y1");
  name.at(2) = string("y2");
  name.at(3) = string("y3");
  vector<string> name_ret;
  cnstr1->getName(name_ret);

  /////////////////////////////////////////////////////
  cout<<"Test LinearConstraint"<<endl;
  Matrix<double,4,5> A;
  A(0,0) = 1.0;
  A(0,3) = 1.5;
  A(2,1) = 3.0;
  A(2,2) = 4.0;
  A(2,4) = -0.1;
  A(3,0) = -1.0;
  A(3,4) = 0.5;
  Vector4d cnstr2_lb;
  Vector4d cnstr2_ub;
  cnstr2_lb<<-1,-2, 1, 3;
  cnstr2_ub<<-1, 2, 4, 3;
  LinearConstraint* cnstr2 = new LinearConstraint(cnstr2_lb,cnstr2_ub,A);
  MatrixXd A_hat;
  cnstr2->getA(A_hat);
  cnstr2->getBounds(lb,ub);
  for(int i = 0;i<4;i++)
  {
    if(abs(lb(i)-cnstr2_lb(i))>1E-10 || abs(ub(i)-cnstr2_ub(i))>1E-10)
    {
      cerr<<"constraint lower bound or upper bound is incorrect in row "<<i<<endl;
    }
  }
  VectorXi iAfun,jAvar;
  VectorXd A_val;
  int nnz;
  cnstr2->getSparseStructure(iAfun,jAvar,A_val,nnz);
  MatrixXd A_sparse = MatrixXd::Zero(4,5);
  if(iAfun.rows() != nnz)
  {
    cerr<<"The number of non-zero elements does not match with iAfun"<<endl;
  }
  for(int i = 0;i<nnz;i++)
  {
    A_sparse(iAfun(i),jAvar(i)) = A_val(i);
  }
  for(int j = 0;j<5;j++)
  {
    for(int i = 0;i<4;i++)
    {
      if(abs(A_sparse(i,j)-A(i,j))>1E-10)
      {
        cerr<<"The sparse matrix is incorrect in row "<<i<<" column "<<j<<endl;
      }
      if(abs(A_hat(i,j)-A(i,j))>1E-10)
      {
        cerr<<"The linear matrix returned by LinearConstraint is incorrect in row "<<i<<" column "<<j<<endl;
      }
    }
  }
  /// Test BoundingBoxConstraint
  cout<<"Test BoundingBoxConstraint"<<endl;
  Vector4d cnstr3_lb, cnstr3_ub;
  cnstr3_lb<<-0.1,1,2,3;
  cnstr3_ub<<-0,2,3,5;
  BoundingBoxConstraint* cnstr3 = new BoundingBoxConstraint(cnstr3_lb,cnstr3_ub);
  delete cnstr1,cnstr2,cnstr3;
  return 0;
}

