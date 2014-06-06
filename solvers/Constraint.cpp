#include "Constraint.h"
#include <iostream>
using namespace std;
using namespace Eigen;

 ConstraintInput::ConstraintInput(const MatrixXd &x)
{
  this->x = x;
}

ConstraintOutput::ConstraintOutput(const MatrixXd &y)
{
  this->y = y;
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

void Constraint::setName(const vector<string> &name)
{
  this->name = name;
}

ConstraintOutputWGradient::ConstraintOutputWGradient(const VectorXd &y, const MatrixXd &dy):ConstraintOutput(y)
{
  this->dy = dy;
}

NonlinearConstraint::NonlinearConstraint(const VectorXd &lb, const VectorXd &ub, int xdim):Constraint(lb,ub)
{
   this->num_cnstr = this->lb.rows();
   for(int i = 0;i<this->num_cnstr;i++)
   {
     if(this->lb(i)>this->ub(i))
     {
       cerr<<"Drake:NonlinearConstraint:lb must be no larger than ub"<<endl;
     }
     else if(this->lb(i) == this->ub(i))
     {
       this->ceq_idx.push_back(i);
     }
     else
     {
       this->cin_idx.push_back(i);
     }
   }
   if(xdim<=0)
   {
     cerr<<"Drake:NonlinearConstraint:xdim should be positive"<<endl;
   }
}


