#ifndef __CONSTRAINT_H__
#define __CONSTRAINT_H__
#include <cstdlib>
#include <string>
#include <vector>
#include <Eigen/Dense>

class ConstraintInput
{
  public:
    ConstraintInput(const Eigen::MatrixXd  &x);
    virtual ~ConstraintInput(void){};
  public:
    Eigen::MatrixXd x;
};

class ConstraintOutput
{
  public:
    ConstraintOutput(const Eigen::MatrixXd &y);
    virtual ~ConstraintOutput(void){};
  public:
    Eigen::MatrixXd y;
};

class Constraint
{
  protected:
    Eigen::MatrixXd lb;
    Eigen::MatrixXd ub;
    std::vector<std::string> name;
  public:
    Constraint(const Eigen::MatrixXd &lb, const Eigen::MatrixXd &ub);
    virtual void eval(const ConstraintInput &in, ConstraintOutput &out) const ;
    virtual void setName(const std::vector<std::string> &name);
    virtual ~Constraint(void){};
};

class ConstraintOutputWGradient: public ConstraintOutput
{
  public:
    ConstraintOutputWGradient(const Eigen::VectorXd &y, const Eigen::MatrixXd &dy);
  public:
    Eigen::MatrixXd dy;
};

class NonlinearConstraint:Constraint
{
  protected:
    int num_cnstr;
    int xdim;
    Eigen::VectorXi iCfun;
    Eigen::VectorXi jCvar;
    int nnz;
    std::vector<int> ceq_idx;
    std::vector<int> cin_idx;
  public:
    NonlinearConstraint(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, int xdim);
    virtual void eval(const ConstraintInput &in, ConstraintOutputWGradient &out) const;
    virtual void setSparseStructure(const Eigen::VectorXi &iCfun, const Eigen::VectorXi &jCvar);
    virtual int getNumCnstr() const;
    virtual int getXdim() const;
    virtual void getSparseStructure(Eigen::VectorXi &iCfun, const Eigen::VectorXi &jCvar) const;
    virtual void getCeqIdx(Eigen::VectorXi ceq_idx) const;
    virtual void getCinIdx(Eigen::VectorXi cin_idx) const;
};
#endif

