#include "NonlinearProgram.h"
#include <iostream>

using namespace std;
using namespace Eigen;

NonlinearProgram::NonlinearProgram(int num_vars, int num_nonlinear_inequality_constraints, int num_nonlinear_equality_constraints)
{
  if(num_vars<0)
  {
    cerr<<"Drake:NonlinearProgram:num_vars should be non-negative"<<endl;
    exit(EXIT_FAILURE);
  }
  if(num_nonlinear_inequality_constraints<0)
  {
    cerr<<"Drake:NonlinearProgram:num_nonlinear_inequality_constraints should be non-negative"<<endl;
    exit(EXIT_FAILURE);
  }
  if(num_nonlinear_equality_constraints<0)
  {
    cerr<<"Drake:NonlinearProgram:num_nonlinear_equality_constraints should be non-negative"<<endl;
    exit(EXIT_FAILURE);
  }
  this->num_vars = num_vars;
  this->num_cin = num_nonlinear_inequality_constraints;
  this->num_ceq = num_nonlinear_equality_constraints;
  this->x_lb = VectorXd::Constant(this->num_vars,-1.0/0.0);
  this->x_lb = VectorXd::Constant(this->num_vars,1.0/0.0);
  this->cin_lb = VectorXd::Constant(this->num_cin,-1.0/0.0);
  this->cin_ub = VectorXd::Zero(this->num_cin);
  this->iFfun = VectorXi::Zero(this->num_vars);
  this->jFvar = VectorXi::LinSpaced(Sequential,this->num_vars,0,this->num_vars-1);
  this->iCinfun.resize(this->num_cin*this->num_vars);
  this->jCinvar.resize(this->num_cin*this->num_vars);
  for(int j=0;j<this->num_vars;j++)
  {
    for(int i = 0;i<this->num_cin;i++)
    {
      this->iCinfun(j*this->num_cin+i) = i;
      this->jCinvar(j*this->num_cin+i) = j;
    }
  }
  this->iCeqfun.resize(this->num_ceq*this->num_vars);
  this->jCeqvar.resize(this->num_ceq*this->num_vars);
  for(int j = 0;j<this->num_vars;j++)
  {
    for(int i = 0;i<this->num_ceq;i++)
    {
      this->iCeqfun(j*this->num_ceq+i) = i;
      this->jCeqvar(j*this->num_ceq+i) = j;
    }
  }
  this->objcon_logic = false;
  this->solver = NonlinearProgram::snopt;
}


void NonlinearProgram::objective(const VectorXd &x, double &f, RowVectorXd &df) const
{
  if(this->objcon_logic)
  {
    cerr<<"Drake:NonlinearProgram:AbstractMethod:all derived classes must implement objective or objectiveAndNonlinearConstraints"<<endl;
    exit(EXIT_FAILURE);
  }
  this->objcon_logic = true;
  VectorXd fgh;
  MatrixXd dfgh;
  this->objectiveAndNonlinearConstraints(x,fgh,dfgh);
  f = fgh(0);
  df = dfgh.row(0);

}

void NonlinearProgram::nonlinearConstraints(const VectorXd &x, VectorXd &g, VectorXd &h, MatrixXd &dg, MatrixXd &dh) const
{
  if(this->objcon_logic)
  {
    cerr<<"Drake:NonlinearProgram:AbstractMethod:all derived classes must implement objective or objectiveAndNonlinearConstraints"<<endl;
    exit(EXIT_FAILURE);
  }
  this->objcon_logic = true;
  VectorXd fgh;
  MatrixXd dfgh;
  this->objectiveAndNonlinearConstraints(x,fgh,dfgh);
  g = fgh.block(1,0,this->num_cin,1);
  h = fgh.block(1+this->num_cin,0,this->num_ceq,1);
}
void NonlinearProgram::addLinearInequalityConstraints(const MatrixXd &Ain, const VectorXd &bin)
{
  if(Ain.rows() != bin.rows())
  {
    cerr<<"Drake:NonlinearProgram:addLinearInequalityConstraints:Ain and bin should have same number of rows"<<endl;
  }
  if(Ain.cols() != this->num_vars)
  {
    cerr<<"Drake:NonlinearProgram:addLinearInequalityConstraints:The number of columns of Ain should be the same as the number of decision variables"<<endl;
    exit(EXIT_FAILURE);
  }
  MatrixXd Ain_tmp = this->Ain;
  this->Ain.resize(this->Ain.rows()+Ain.rows(),this->num_vars);
  this->Ain << Ain_tmp, Ain;
  this->bin.conservativeResize(this->bin.rows()+bin.rows());
  this->bin.block(Ain_tmp.rows(),0,bin.rows(),1) = bin;
}

void NonlinearProgram::objectiveAndNonlinearConstraints(const VectorXd &x, VectorXd &fgh, MatrixXd &dfgh) const
{
  if(this->objcon_logic)
  {
    cerr<<"Drake:NonlinearProgram:AbstractMethd:all derived classes must implement objective or objectiveAndNonlinearConstraints"<<endl;
    exit(EXIT_FAILURE);
  }
  this->objcon_logic = true;
  double f;
  RowVectorXd df;
  this->objective(x,f,df);
  VectorXd g,h;
  MatrixXd dg,dh;
  this->nonlinearConstraints(x,g,h,dg,dh);
  fgh.resize(1+this->num_cin+this->num_ceq);
  dfgh.resize(1+this->num_cin+this->num_ceq,this->num_vars);
  fgh(0) = f;
  fgh.block(1,0,this->num_cin,1) = g;
  fgh.block(1+this->num_cin,0,this->num_ceq,1) = h;
  dfgh.row(0) = df;
  dfgh.block(1,0,this->num_cin,this->num_vars) = dg;
  dfgh.block(1+this->num_cin,0,this->num_ceq,this->num_vars) = dh;
}

void NonlinearProgram::addLinearEqualityConstraints(const MatrixXd &Aeq, const VectorXd &beq)
{
  if(Aeq.rows() != beq.rows())
  {
    cerr<<"Drake:NonlinearProgram:addLinearInequalityConstraints:Aeq and beq should have same number of rows"<<endl;
    exit(EXIT_FAILURE);
  }
  if(Aeq.cols() != this->num_vars)
  {
    cerr<<"Drake:NonlinearProgram:addLinearInequalityConstraints:The number of columns of Aeq should be the same as the number of decision variables"<<endl;
    exit(EXIT_FAILURE);
  }
  MatrixXd Aeq_tmp = this->Aeq;
  this->Aeq.resize(this->Aeq.rows()+Aeq.rows(),this->num_vars);
  this->Aeq << Aeq_tmp, Aeq;
  this->beq.conservativeResize(this->beq.rows()+beq.rows());
  this->beq.block(Aeq_tmp.rows(),0,beq.rows(),1) = beq;
}

void NonlinearProgram::setVarBounds(const VectorXd &x_lb, const VectorXd &x_ub)
{
  if(x_lb.rows() != this->num_vars)
  {
    cerr<<"Drake:NonlinearProgram:setVarBounds:the number of rows in x_lb should be this->num_vars"<<endl;
    exit(EXIT_FAILURE);
  }
  if(x_ub.rows() != this->num_vars)
  {
    cerr<<"Drake:NonlinearProgram:setVarBounds:the number of rows in x_ub should be this->num_vars"<<endl;
    exit(EXIT_FAILURE);
  }
  for(int i = 0;i<this->num_vars;i++)
  {
    if(x_lb(i)>x_ub(i))
    {
      cerr<<"Drake:NonlinearProgram:setVarBounds:x_lb("<<i<<") should be no larger than x_ub("<<i<<")"<<endl;
      exit(EXIT_FAILURE);
    }
  }
  this->x_lb = x_lb;
  this->x_ub = x_ub;
}

void NonlinearProgram::setObjectiveGradientSparsity(const VectorXi &jFvar)
{
  for(int i = 0;i<jFvar.rows();i++)
  {
    if(jFvar(i)<0 || jFvar(i)>=this->num_vars)
    {
      cerr<<"Drake:NonlinearProgram:setObjectiveGradientSparsity:jFvar("<<i<<")="<<jFvar(i)<<", out of bounds"<<endl;
      exit(EXIT_FAILURE);
    }
  }
  this->iFfun = VectorXi::Zero(jFvar.rows());
  this->jFvar = jFvar;
}

void NonlinearProgram::setNonlinearInequalityConstraintsGradientSparsity(const VectorXi &iCinfun, const VectorXi &jCinvar)
{
  int nnz = iCinfun.rows();
  if(jCinvar.rows() != nnz)
  {
    cerr<<"Drake:NonlinearProgram:setNonlinearInequalityConstraintsGradientSparsity:iCinfun and jCinvar should have the same size"<<endl;
    exit(EXIT_FAILURE);
  }
  for(int i = 0;i<nnz;i++)
  {
    if(iCinfun(i)<0 || iCinfun(i)>=this->num_cin || jCinvar(i)<0 || jCinvar(i)>=this->num_vars)
    {
      cerr<<"Drake:NonlinearProgram:setNonlinearInequalityConstraintsGradientSparsity:iCinfun("<<i<<") or jCinvar("<<i<<") is out of bounds"<<endl;
      exit(EXIT_FAILURE);
    }
  }
  this->iCinfun = iCinfun;
  this->jCinvar = jCinvar;
}

void NonlinearProgram::setNonlinearEqualityConstraintsGradientSparsity(const VectorXi &iCeqfun, const VectorXi &jCeqvar)
{
  int nnz = iCeqfun.rows();
  if(jCeqvar.rows() != nnz)
  {
    cerr<<"Drake:NonlinearProgram:setNonlinearEqualityConstraintsGradientSparsity:iCeqfun and jCeqvar should have the same size"<<endl;
    exit(EXIT_FAILURE);
  }
  for(int i = 0;i<nnz;i++)
  {
    if(iCeqfun(i)<0 || iCeqfun(i)>=this->num_cin || jCeqvar(i)<0 || jCeqvar(i)>=this->num_vars)
    {
      cerr<<"Drake:NonlinearProgram:setNonlinearInequalityConstraintsGradientSparsity:iCeqfun("<<i<<") or jCeqvar("<<i<<") is out of bounds"<<endl;
      exit(EXIT_FAILURE);
    }
  }
  this->iCeqfun = iCeqfun;
  this->jCeqvar = jCeqvar;
}

void NonlinearProgram::setSolver(int solver)
{
  if(solver<0 || solver >2)
  {
    cerr<<"NonlinearProgram:setSolver:unsupported solver type"<<endl;
    exit(EXIT_FAILURE);
  }
  this->solver = solver;
}

void NonlinearProgram::setNonlinearInequalityBounds(const VectorXd &cin_lb, const VectorXd &cin_ub, const VectorXi &cin_idx)
{
  int num_cin_set = cin_lb.rows(); 
  if(cin_ub.rows() != num_cin_set || cin_idx.rows() != num_cin_set)
  {
    cerr<<"NonlinearProgram:setNonlinearInequalityBounds:cin_lb, cin_ub and cin_idx should have the same size"<<endl;
    exit(EXIT_FAILURE);
  }
  for(int i = 0;i<num_cin_set;i++)
  {
    if(cin_idx(i)<0 || cin_idx(i)>=this->num_cin)
    {
      cerr<<"NonlinearProgram:setNonlinearInequalityBounds:cin_idx("<<i<<")="<<cin_idx(i)<<", out of bounds"<<endl;
      exit(EXIT_FAILURE);
    }
    if(cin_lb(i)>cin_ub(i))
    {
      cerr<<"NonlinearProgram:setNonlinearInequalityBounds:cin_lb("<<i<<") should be no larger than cin_ub("<<i<<")"<<endl;
      exit(EXIT_FAILURE);
    }
    this->cin_lb(cin_idx(i)) = cin_lb(i);
    this->cin_ub(cin_idx(i)) = cin_ub(i);
  }
}

void NonlinearProgram::setNonlinearInequalityBounds(const VectorXd &cin_lb, const VectorXd &cin_ub)
{
  VectorXi cin_idx = VectorXi::LinSpaced(Sequential,this->num_cin, 0, this->num_cin-1);
  this->setNonlinearInequalityBounds(cin_lb,cin_ub,cin_idx);
}

void NonlinearProgram::setNonlinearGradientSparsity() const
{
  int iGfun_length = this->iFfun.rows()+this->iCinfun.rows()+this->iCeqfun.rows();
  this->iGfun.resize(iGfun_length);
  this->jGvar.resize(iGfun_length);
  this->iGfun.block(0,0,this->iFfun.rows(),1) = this->iFfun;
  this->jGvar.block(0,0,this->iFfun.rows(),1) = this->jFvar;
  this->iGfun.block(this->iFfun.rows(),0,this->iCinfun.rows(),1) = this->iCinfun+VectorXi::Ones(this->iCinfun.rows());
  this->jGvar.block(this->iFfun.rows(),0,this->iCinfun.rows(),1) = this->jCinvar;
  this->iGfun.block(this->iFfun.rows()+this->iCinfun.rows(),0,this->iCeqfun.rows(),1) = this->iCeqfun+VectorXi::Constant(1+this->num_cin,this->iCeqfun.rows());
  this->jGvar.block(this->iFfun.rows()+this->iCinfun.rows(),0,this->iCeqfun.rows(),1) = this->jCeqvar;
}

void NonlinearProgram::bounds(VectorXd &lb, VectorXd &ub) const
{
  int bound_size = 1+this->num_cin+this->num_ceq+this->bin.rows()+this->beq.rows();
  lb.resize(bound_size);
  ub.resize(bound_size);
  lb(0) = -1.0/0.0;
  ub(0) = 1.0/0.0;
  lb.block(1,0,this->num_cin,1) = this->cin_lb;
  ub.block(1,0,this->num_cin,1) = this->cin_ub;
  lb.block(1+this->num_cin,0,this->num_ceq,1) = VectorXd::Zero(this->num_ceq);
  ub.block(1+this->num_cin,0,this->num_ceq,1) = VectorXd::Zero(this->num_ceq);
  lb.block(1+this->num_cin+this->num_ceq,0,this->bin.rows(),1) = VectorXd::Constant(-1.0/0.0,this->bin.rows());
  ub.block(1+this->num_cin+this->num_ceq,0,this->bin.rows(),1) = this->bin;
  lb.block(1+this->num_cin+this->num_ceq+this->bin.rows(),0,this->beq.rows(),1) = this->beq;
  ub.block(1+this->num_cin+this->num_ceq+this->bin.rows(),0,this->beq.rows(),1) = this->beq;
}

void NonlinearProgram::solve(const VectorXd &x0, VectorXd &x, double &objval, int &exitflag) const
{
  switch(this->solver)
  {
    case this->snopt:
      this->snopt_solve(x0,x,objval,exitflag);
      break;
    default:
      cerr<<"Unsupported solver"<<endl;
  }
}

void NonlinearProgram::snopt_solve(const VectorXd &x0, VectorXd &x, double &objval, int &exitflag) const
{
  MatrixXd A(this->bin.rows()+this->beq.rows(),this->num_vars);
  A.block(0,0,this->bin.rows(),this->num_vars) = this->Ain;
  A.block(this->bin.rows(),0,this->beq.rows(),this->num_vars) = this->Aeq;
  VectorXi iAfun((this->bin.rows()+this->beq.rows())*this->num_vars);
  VectorXi jAvar((this->bin.rows()+this->beq.rows())*this->num_vars);
  VectorXd A_val((this->bin.rows()+this->beq.rows())*this->num_vars);
  int nnz_A = 0;
  for(int j = 0;j<this->num_vars;j++)
  {
    for(int i = 0;i<this->bin.rows()+this->beq.rows();i++)
    {
      if(A(i,j) != 0)
      {
        iAfun(nnz_A) = i;
        jAvar(nnz_A) = j;
        A_val(nnz_A) = A(i,j);
        nnz_A++;
      }
    }
  }
  iAfun.conservativeResize(nnz_A);
  jAvar.conservativeResize(nnz_A);
  A_val.conservativeResize(nnz_A);
  iAfun = iAfun+VectorXi::Constant(1+this->num_cin+this->num_ceq,iAfun.rows());
  this->setNonlinearGradientSparsity();
  VectorXd lb,ub;
  this->bounds(lb,ub);

  // set up snopt
  snopt::integer nF = static_cast<snopt::integer>(1+this->num_cin+this->num_ceq);
  snopt::doublereal *Flow = new snopt::doublereal[nF];
  snopt::doublereal *Fupp = new snopt::doublereal[nF];
  for(int i = 0;i<nF;i++)
  {
    Flow[i] = lb(i);
    Fupp[i] = ub(i);
  }
  snopt::integer nG = iGfun.rows();
  snopt::integer *iGfun_snopt = new snopt::integer(nG);
  snopt::integer *jGvar_snopt = new snopt::integer(nG);
  for(int i = 0;i<nG;i++)
  {
    iGfun_snopt[i] = static_cast<snopt::integer>(this->iGfun(i)+1);//snopt is using 1-index
    jGvar_snopt[i] = static_cast<snopt::integer>(this->jGvar(i)+1);
  }
  snopt::integer lenA = iAfun.rows();
  snopt::integer *iAfun_snopt = new snopt::integer(lenA);
  snopt::integer *jAvar_snopt = new snopt::integer(lenA);
  snopt::doublereal *Aval_snopt = new snopt::doublereal(lenA);
  for(int i = 0;i<lenA;i++)
  {
    iAfun_snopt[i] = static_cast<snopt::integer>(iAfun(i)+1);
    jAvar_snopt[i] = static_cast<snopt::integer>(jAvar(i)+1);
    Aval_snopt[i] = static_cast<snopt::doublereal>(A_val(i));
  }
  snopt::integer nx = static_cast<snopt::integer>(this->num_vars);
  snopt::doublereal *xlow = new snopt::doublereal[nx];
  snopt::doublereal *xupp = new snopt::doublereal[nx];
  for(int i = 0;i<nx;i++)
  {
    xlow[i] = this->x_lb(i);
    xupp[i] = this->x_ub(i);
  }

  snopt::snoptProblem snopt_problem;
  snopt_problem.setUserFun(&NonlinearProgram::snopt_userfun);
//  snopt::integer minrw, miniw, mincw;
//  snopt::integer lenrw = (this->num_cin+this->num_ceq+this->bin.rows()+this->beq.rows())*this->num_vars*1000;
//  snopt::integer leniw = (this->num_cin+this->num_ceq+this->bin.rows()+this->beq.rows())*this->num_vars*100;
//  snopt::integer lencw = 500;
//
//  snopt::doublereal *rw;
//  rw = (snopt::doublereal*) std::calloc(lenrw,sizeof(snopt::doublereal));
//  snopt::integer *iw;
//  iw = (snopt::integer*) std::calloc(leniw,sizeof(snopt::integer));
//  char cw[8*lencw];
//
//  snopt::integer Cold = 0; //, Basis = 1, Warm = 2;
//  snopt::doublereal *xmul = new snopt::doublereal[nx];
//  snopt::integer    *xstate = new snopt::integer[nx];
//  for(int i = 0;i<nx;i++)
//  {
//    xstate[i] = 0;
//  }
//  snopt::doublereal *F      = new snopt::doublereal[nF];
//  snopt::doublereal *Fmul   = new snopt::doublereal[nF];
//  snopt::integer    *Fstate = new snopt::integer[nF];
//  for(int i = 0;i<nF;i++)
//  {
//    Fstate[i] = 0;
//  }
//
//  snopt::doublereal ObjAdd = 0.0;
//  snopt::integer ObjRow = 1;
//
//  snopt::integer nxname = 1, nFname = 1, npname = 0;
//  char* xnames = new char[nxname*8];
//  char* Fnames = new char[nFname*8];
//
//  char Prob[200];
//  snopt::integer iSumm = -1;
//  snopt::integer iPrint = -1;
//
//  snopt::integer nS,nInf;
//  snopt::doublereal sInf;
//
//  snopt::integer INFO_snopt;
//  snopt::sninit_(&iPrint,&iSumm,cw,&lencw,iw,&leniw,rw,&lenrw,8*500);
//  char strOpt1[200] = "Derivative option";
//  snopt::integer DerOpt = 1, strOpt_len = strlen(strOpt1);
//  snopt::snseti_(strOpt1,&DerOpt,&iPrint,&iSumm,&INFO_snopt,cw,&lencw,iw,&leniw,rw,&lenrw,strOpt_len,8*500);
//
//
//  typedef int(*snopt_fp) (snopt::integer *Status, snopt::integer *n, snopt::doublereal x[],
//    snopt::integer *needF, snopt::integer *neF, snopt::doublereal F[],
//    snopt::integer *needG, snopt::integer *neG, snopt::doublereal G[],
//    char *cu, snopt::integer *lencu,
//    snopt::integer iu[], snopt::integer *leniu,
//    snopt::doublereal ru[],snopt::integer *lenru);
//  snopt::snopta_
//    ( &Cold, &nF, &nx, &nxname, &nFname,
//      &ObjAdd, &ObjRow, Prob, static_cast<snopt_fp>(&NonlinearProgram::snopt_userfun),
//      iAfun_snopt, jAvar_snopt, &lenA, &lenA, Aval_snopt,
//      iGfun_snopt, jGvar_snopt, &nG, &nG,
//      xlow, xupp, xnames, Flow, Fupp, Fnames,
//      x, xstate, xmul, F, Fstate, Fmul,
//      &INFO_snopt, &mincw, &miniw, &minrw,
//      &nS, &nInf, &sInf,
//      cw, &lencw, iw, &leniw, rw, &lenrw,
//      cw, &lencw, iw, &leniw, rw, &lenrw,
//      npname, 8*nxname, 8*nFname,
//      8*500, 8*500);
}

int NonlinearProgram::snopt_userfun(snopt::integer *Status, snopt::integer *n, snopt::doublereal x[],
    snopt::integer *needF, snopt::integer *neF, snopt::doublereal F[],
    snopt::integer *needG, snopt::integer *neG, snopt::doublereal G[],
    char *cu, snopt::integer *lencu,
    snopt::integer iu[], snopt::integer *leniu,
    snopt::doublereal ru[],snopt::integer *lenru)
{
  VectorXd x_vec(this->num_vars);
  memcpy(x_vec.data(),x,sizeof(snopt::doublereal)*this->num_vars);
  VectorXd fgh;
  MatrixXd dfgh;
  this->objectiveAndNonlinearConstraints(x_vec,fgh,dfgh);
  memcpy(F,fgh.data(),sizeof(snopt::doublereal)*(*neF));
  for(int i = 0;i<*neG;i++)
  {
    G[i] = dfgh(this->iGfun(i),this->jGvar(i));
  }
  return 0;
}

