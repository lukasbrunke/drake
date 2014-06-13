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

class Constraint
{
  protected:
    Eigen::MatrixXd lb;
    Eigen::MatrixXd ub;
    std::vector<std::string> name;
    void (*eval_handle)(const ConstraintInput &in, Eigen::MatrixXd &y);
  public:
    Constraint(const Eigen::MatrixXd &lb, const Eigen::MatrixXd &ub);
    virtual void setEvalHandle(void(*eval_handle)(const ConstraintInput &, Eigen::MatrixXd &));
    virtual void eval(const ConstraintInput &in, Eigen::MatrixXd &y) const;
    virtual void setName(const std::vector<std::string> &name);
    virtual void getName(std::vector<std::string> &name) const;
    virtual void getBounds(Eigen::MatrixXd &lb, Eigen::MatrixXd &ub) const;
    virtual ~Constraint(void){};
};

class NonlinearConstraint:public Constraint
{
  protected:
    int num_cnstr;
    int xdim;
    Eigen::VectorXi iCfun;
    Eigen::VectorXi jCvar;
    int nnz;
    Eigen::VectorXi ceq_idx;
    Eigen::VectorXi cin_idx;
    void (*geval_handle)(const ConstraintInput &, Eigen::VectorXd &y, Eigen::MatrixXd &dy);
  public:
    NonlinearConstraint(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, int xdim);
    virtual void eval(const ConstraintInput &in, Eigen::MatrixXd &y) const;
    virtual void geval(const ConstraintInput &in, Eigen::VectorXd &, Eigen::MatrixXd &) const ;
    virtual void setSparseStructure(const Eigen::VectorXi &iCfun, const Eigen::VectorXi &jCvar);
    virtual int getNumCnstr(void) const;
    virtual int getXdim(void) const;
    virtual void getSparseStructure(Eigen::VectorXi &iCfun, Eigen::VectorXi &jCvar,int &nnz) const;
    virtual void getCeqIdx(Eigen::VectorXi &ceq_idx) const;
    virtual void getCinIdx(Eigen::VectorXi &cin_idx) const;
    virtual void setGevalHandle(void (*geval_handle)(const ConstraintInput &, Eigen::VectorXd &y, Eigen::MatrixXd &dy));
    virtual void setName(const std::vector<std::string> &name);
    virtual void getBounds(Eigen::VectorXd &lb, Eigen::VectorXd &ub) const;
    bool checkGradient(double tol, const ConstraintInput & in) const;
    virtual ~NonlinearConstraint(void){};
};

class LinearConstraint:public Constraint
{
  protected:
    int num_cnstr;
    int xdim;
    Eigen::MatrixXd A;
    Eigen::VectorXi iAfun;
    Eigen::VectorXi jAvar;
    Eigen::VectorXd A_val;
    int nnz;
    Eigen::VectorXi ceq_idx;
    Eigen::VectorXi cin_idx;

  public:
    LinearConstraint(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, const Eigen::MatrixXd &A);
    virtual void eval(const ConstraintInput &in, Eigen::MatrixXd &y) const;
    virtual void getSparseStructure(Eigen::VectorXi &iAfun, Eigen::VectorXi &jAvar, Eigen::VectorXd &A_val, int &nnz) const;
    virtual void getA(Eigen::MatrixXd &A) const;
    virtual void setName(const std::vector<std::string> name);
    virtual void getBounds(Eigen::VectorXd &lb, Eigen::VectorXd &ub) const;
    virtual ~LinearConstraint(void){};
};

class BoundingBoxConstraint:public LinearConstraint
{
  public:
    BoundingBoxConstraint(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub);
    ~BoundingBoxConstraint(void){};
};

#endif

