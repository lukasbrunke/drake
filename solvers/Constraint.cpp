#include "Constraint.h"
#include <iostream>
using namespace std;
using namespace Eigen;

ConstraintInput::ConstraintInput(const MatrixXd &x)
{
  this->x = x;
}

Constraint::Constraint(const MatrixXd &lb, const MatrixXd &ub)
{
  if(lb.rows() != ub.rows() || lb.cols() != ub.cols())
  {
    cerr<<"constraint lower bound and upper bound should have the same size"<<endl;
  }
  this->lb = lb;
  this->ub = ub;
}

void Constraint::setEvalHandle(void (*eval_handle)(const ConstraintInput &, MatrixXd &))
{
  this->eval_handle = eval_handle;
}

void Constraint::setName(const vector<string> &name)
{
  this->name = name;
}

void Constraint::getName(vector<string> &name) const
{
  name = this->name;
}
void Constraint::eval(const ConstraintInput & in, MatrixXd &y) const
{
  if(this->eval_handle == nullptr)
  {
    cerr<<"Drake:Constraint::eval:null eval pointer"<<endl;
  }
  else
  {
    this->eval_handle(in,y);
  }
}

void Constraint::getBounds(MatrixXd &lb, MatrixXd &ub) const
{
  lb = this->lb;
  ub = this->ub;
}

NonlinearConstraint::NonlinearConstraint(const VectorXd &lb, const VectorXd &ub, int xdim):Constraint(lb,ub)
{
   this->num_cnstr = this->lb.rows();
   this->ceq_idx.resize(this->num_cnstr);
   this->cin_idx.resize(this->num_cnstr);
   int num_ceq = 0;
   int num_cin = 0;
   for(int i = 0;i<this->num_cnstr;i++)
   {
     if(this->lb(i)>this->ub(i))
     {
       cerr<<"Drake:NonlinearConstraint:lb must be no larger than ub"<<endl;
     }
     else if(this->lb(i) == this->ub(i))
     {
       this->ceq_idx(num_ceq) = i;
       num_ceq++;
     }
     else
     {
       this->cin_idx(num_cin) = i;
       num_cin++;
     }
   }
   this->ceq_idx.conservativeResize(num_ceq);
   this->cin_idx.conservativeResize(num_cin);
   if(xdim<=0)
   {
     cerr<<"Drake:NonlinearConstraint:xdim should be positive"<<endl;
   }
   this->xdim = xdim;
   this->iCfun.resize(this->num_cnstr*this->xdim);
   this->jCvar.resize(this->num_cnstr*this->xdim);
   for(int j = 0;j<this->xdim;j++)
   {
     for(int i = 0;i<this->num_cnstr;i++)
     {
       this->iCfun(j*this->num_cnstr+i) = i;
       this->jCvar(j*this->num_cnstr+i) = j;
     }
   }
   this->nnz = this->num_cnstr*this->xdim;
   this->name.resize(this->num_cnstr);
   for(int i = 0;i<this->num_cnstr;i++)
   {
     this->name.at(i) = string("");
   }
   this->eval_handle = nullptr;
}

int NonlinearConstraint::getNumCnstr(void) const
{
  return this->num_cnstr;
}

int NonlinearConstraint::getXdim(void) const
{
  return this->xdim;
}

void NonlinearConstraint::getSparseStructure(VectorXi &iCfun, VectorXi &jCvar,int &nnz) const
{
  iCfun = this->iCfun;
  jCvar = this->jCvar;
  nnz = this->nnz;
}

void NonlinearConstraint::getCeqIdx(VectorXi &ceq_idx) const
{
  ceq_idx = this->ceq_idx;
}

void NonlinearConstraint::getCinIdx(VectorXi &cin_idx) const
{
  cin_idx = this->cin_idx;
}

void NonlinearConstraint::setSparseStructure(const VectorXi &iCfun, const VectorXi &jCvar)
{
  if(iCfun.rows() != jCvar.rows())
  {
    cerr<<"Drake:NonlinearConstraint:setSparseStructure:iCfun and jCvar should have the same size"<<endl;
  }
  for(int i = 0;i<iCfun.rows();i++)
  {
    if(iCfun(i)<0 || iCfun(i)>=this->num_cnstr || jCvar(i)<0 || jCvar(i)>=this->xdim)
    {
      cerr<<"Drake:NonlinearConstraint:setSparseStructure:iCfun and jCvar are not within the valid range"<<endl;
    }
  }
  this->iCfun = iCfun;
  this->jCvar = jCvar;
  this->nnz = iCfun.rows();
}

void NonlinearConstraint::eval(const ConstraintInput &in, MatrixXd &y) const
{
  if(this->eval_handle != nullptr)
  {
    this->eval_handle(in,y);
  }
  else if(this->geval_handle != nullptr)
  {
    VectorXd y_tmp;
    MatrixXd dy;
    this->geval_handle(in,y_tmp,dy);
    y = y_tmp;
  }
  else
  {
    cerr<<"Drake:NonlinearConstraint:eval:both eval_handle and geval_handle are null pointers"<<endl;
  }
}

void NonlinearConstraint::geval(const ConstraintInput &in, VectorXd &y, MatrixXd &dy) const
{
  if(this->geval_handle == nullptr)
  {
    cerr<<"Drake:NonlinearConstraint::geval:null geval_handle pointer"<<endl;
  }
  else
  {
    this->geval_handle(in,y,dy);
  }
}

void NonlinearConstraint::setGevalHandle(void (*geval_handle)(const ConstraintInput &, VectorXd & y, MatrixXd &dy))
{
  this->geval_handle = geval_handle;
}

bool NonlinearConstraint::checkGradient(double tol, const ConstraintInput &in) const
{
  VectorXd y;
  MatrixXd dy;
  this->geval(in,y,dy);
  MatrixXd dy_numeric(this->num_cnstr,this->xdim);
  for(int i = 0;i<this->xdim;i++)
  {
    double x_err = 1E-7;
    MatrixXd xi = in.x;
    xi(i) += x_err;
    MatrixXd yi;
    this->eval(ConstraintInput(xi),yi);
    dy_numeric.col(i) = (yi.col(0)-y)/x_err;
  }
  MatrixXd dy_sparse = MatrixXd::Zero(this->num_cnstr,this->xdim);
  VectorXi iCfun,jCvar;
  int nnz;
  this->getSparseStructure(iCfun,jCvar,nnz);
  for(int i = 0;i<nnz;i++)
  {
    dy_sparse(iCfun(i),jCvar(i)) = dy(iCfun(i),jCvar(i));
  }
  bool ret_flag = true;
  for(int j = 0;j<this->xdim;j++)
  {
    for(int i = 0;i<this->num_cnstr;i++)
    {
      if(abs(dy(i,j)-dy_numeric(i,j))>tol) 
      {
        cout<<"user_gradient("<<i<<","<<j<<")="<<dy(i,j)<<",numerical_gradient("<<i<<","<<j<<")="<<dy_numeric(i,j)<<endl;
        ret_flag = false;
      }
      if(abs(dy(i,j)-dy_sparse(i,j))>=1E-10)
      {
        cout<<"Gradient sparse pattern is incorrect in row "<<i<<" column "<<j<<endl;
        ret_flag = false;
      }
    }
  }
  return ret_flag;
}

void NonlinearConstraint::setName(const std::vector<std::string> &name)
{
  if(name.size() != this->num_cnstr)
  {
    cerr<<"Drake:NonlinearConstraint:setName:name string has incorrect size"<<endl;
  }
  Constraint::setName(name);
}

void NonlinearConstraint::getBounds(VectorXd &lb, VectorXd &ub) const
{
  lb.resize(this->num_cnstr);
  ub.resize(this->num_cnstr);
  memcpy(lb.data(),this->lb.data(),sizeof(double)*this->num_cnstr);
  memcpy(ub.data(),this->ub.data(),sizeof(double)*this->num_cnstr);
}

LinearConstraint::LinearConstraint(const VectorXd &lb, const VectorXd &ub, const MatrixXd &A):Constraint(lb,ub)
{
  if(lb.rows() != ub.rows())
  {
    cerr<<"Drake:LinearConstraint:lb and ub should have the same size"<<endl;
  }
  this->num_cnstr = lb.rows();
  if(A.rows() != this->num_cnstr)
  {
    cerr<<"Drake:LinearConstraint: matrix A and bounds have different number of rows"<<endl;
  }
  this->xdim = A.cols();
  this->ceq_idx.resize(this->num_cnstr);
  this->cin_idx.resize(this->num_cnstr);
  int num_ceq = 0;
  int num_cin = 0;
  for(int i = 0;i<this->num_cnstr;i++)
  {
    if(this->lb(i)>this->ub(i))
    {
      cerr<<"Drake:LinearConstraint:lb must be no larger than ub"<<endl;
    }
    else if(this->lb(i) == this->ub(i))
    {
      this->ceq_idx(num_ceq) = i;
      num_ceq++;
    }
    else
    {
      this->cin_idx(num_cin) = i;
      num_cin++;
    }
  }
  this->ceq_idx.conservativeResize(num_ceq);
  this->cin_idx.conservativeResize(num_cin);
  this->A = A;
  this->nnz = 0;
  int A_size = this->num_cnstr*this->xdim;
  this->iAfun.resize(A_size);
  this->jAvar.resize(A_size);
  this->A_val.resize(A_size);
  for(int j= 0;j<this->xdim;j++)
  {
    for(int i = 0;i<this->num_cnstr;i++)
    {
      if(A(i,j) != 0)
      {
        iAfun(this->nnz) = i;
        jAvar(this->nnz) = j;
        A_val(this->nnz) = A(i,j);
        this->nnz++;
      }
    }
  }
  this->iAfun.conservativeResize(this->nnz);
  this->jAvar.conservativeResize(this->nnz);
  this->A_val.conservativeResize(this->nnz);
  this->name.resize(this->num_cnstr);
  for(int i = 0;i<this->num_cnstr;i++)
  {
    this->name.at(i) = string("");
  }
}

void LinearConstraint::eval(const ConstraintInput &in, MatrixXd &y) const
{
  y = this->A*in.x;
}

void LinearConstraint::getSparseStructure(VectorXi &iAfun, VectorXi &jAvar, VectorXd &A_val,int &nnz) const
{
  iAfun = this->iAfun;
  jAvar = this->jAvar;
  A_val = this->A_val;
  nnz = this->nnz;
}

void LinearConstraint::getA(MatrixXd &A) const
{
  A = this->A;
}

void LinearConstraint::setName(const vector<string> name)
{
  if(name.size() != this->num_cnstr)
  {
    cerr<<"Drake:LinearConstraint:setName: name string has incorrect size"<<endl;
  }
  Constraint::setName(name);
}

void LinearConstraint::getBounds(VectorXd &lb, VectorXd &ub) const
{
  lb.resize(this->num_cnstr);
  ub.resize(this->num_cnstr);
  memcpy(lb.data(),this->lb.data(),sizeof(double)*this->num_cnstr);
  memcpy(ub.data(),this->ub.data(),sizeof(double)*this->num_cnstr);
}

BoundingBoxConstraint::BoundingBoxConstraint(const VectorXd &lb, const VectorXd &ub):LinearConstraint(lb,ub,MatrixXd::Identity(lb.rows(),lb.rows())){};
