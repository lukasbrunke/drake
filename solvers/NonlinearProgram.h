#include <cstdlib>
#include <math.h>
#include <float.h>
#include <cstring>

namespace snopt {
#include "snopt.hh"
#include "snfilewrapper.hh"
#include "snoptProblem.hh"
}
#undef min
#undef max
#undef abs

#include <Eigen/Dense>

class NonlinearProgram
{
  protected:
    int num_vars;
    int num_cin;
    int num_ceq;
    Eigen::MatrixXd Ain;
    Eigen::MatrixXd Aeq;
    Eigen::VectorXd bin;
    Eigen::VectorXd beq;
    Eigen::VectorXd cin_lb;
    Eigen::VectorXd cin_ub;
    Eigen::VectorXd x_lb;
    Eigen::VectorXd x_ub;
    Eigen::VectorXi iFfun;
    Eigen::VectorXi jFvar;
    Eigen::VectorXi iCinfun;
    Eigen::VectorXi jCinvar;
    Eigen::VectorXi iCeqfun;
    Eigen::VectorXi jCeqvar;
    int solver;

  private:
    mutable bool objcon_logic;
    mutable Eigen::VectorXi iGfun;
    mutable Eigen::VectorXi jGvar;
  public:
    static const int snopt = 0;
    static const int knitro = 1;
    static const int nlp = 2;

  public:
    NonlinearProgram(int num_vars, int num_nonlinear_inequality_constraints, int num_nonlinear_equality_constraints);
    virtual void objective(const Eigen::VectorXd & x, double &f, Eigen::RowVectorXd &df) const;
    virtual void nonlinearConstraints(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::VectorXd &h, Eigen::MatrixXd &dg, Eigen::MatrixXd &dh) const; 
    virtual void objectiveAndNonlinearConstraints(const Eigen::VectorXd &x, Eigen::VectorXd &fgh, Eigen::MatrixXd &dfgh) const;
    void addLinearInequalityConstraints(const Eigen::MatrixXd &Ain, const Eigen::VectorXd &bin);
    void addLinearEqualityConstraints(const Eigen::MatrixXd &Aeq, const Eigen::VectorXd &beq);
    void setVarBounds(const Eigen::VectorXd &x_lb, const Eigen::VectorXd &x_ub);
    void setObjectiveGradientSparsity(const Eigen::VectorXi &jFvar);
    void setNonlinearInequalityConstraintsGradientSparsity(const Eigen::VectorXi &iCinfun, const Eigen::VectorXi &jCinvar);
    void setNonlinearEqualityConstraintsGradientSparsity(const Eigen::VectorXi &iCeqfun, const Eigen::VectorXi &jCeqvar);
    void setSolver(int solver);
    void setNonlinearInequalityBounds(const Eigen::VectorXd &cin_lb, const Eigen::VectorXd &cin_ub, const Eigen::VectorXi &cin_idx);
    void setNonlinearInequalityBounds(const Eigen::VectorXd &cin_lb, const Eigen::VectorXd &cin_ub);
    void bounds(Eigen::VectorXd &lb, Eigen::VectorXd &ub) const;
    virtual void solve(const Eigen::VectorXd &x0, Eigen::VectorXd &x, double &objval, int &exitflag) const;
    void snopt_solve(const Eigen::VectorXd &x0, Eigen::VectorXd &x, double &objval, int &exitflag) const;
    int snopt_userfun(snopt::integer *Status, snopt::integer *n, snopt::doublereal x[],snopt::integer *needF, snopt::integer *neF, snopt::doublereal F[], snopt::integer *needG, snopt::integer *neG, snopt::doublereal G[], char *cu, snopt::integer *lencu, snopt::integer iu[], snopt::integer *leniu,snopt::doublereal ru[], snopt::integer *lenru);
    virtual ~NonlinearProgram(void) = 0;

  private:
    void setNonlinearGradientSparsity() const;
};
