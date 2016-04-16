function testRecompV()
x = msspoly('x',8);
y = msspoly('y',4);
p = msspoly('p',5);
q = msspoly('q',3);
xypq_monomial = monomials([x;y;p;q],0:4);
M = randn(1,length(xypq_monomial));
expr = M*xypq_monomial;
[indet,expr_power,expr_coeff] = decomp(expr,[p;q]);

dexpr_coeff = diff(expr_coeff',[p;q]);

[coeff_var,coeff_power,coeff_M] = decomp(expr_coeff');
coeff_match = match([p;q],coeff_var);
[dcoeff_var,dcoeff_power,dcoeff_M] = decomp(dexpr_coeff);
dcoeff_match = match([p;q],dcoeff_var);
pq_val = randn(length(p)+length(q),1);
cnstr_normalizer = 100;
[c,dc] = recompVmex(pq_val,coeff_match,coeff_power,coeff_M,dcoeff_match,dcoeff_power,dcoeff_M,cnstr_normalizer);
[~,dc_num] = geval(@(pq) recompVmex(pq,coeff_match,coeff_power,coeff_M,dcoeff_match,dcoeff_power,dcoeff_M,cnstr_normalizer),pq_val,struct('grad_method','numerical'));
valuecheck(dc,dc_num,1e-4);
valuecheck(double(clean(recomp(indet,expr_power,c'*cnstr_normalizer)-subs(expr,[p;q],pq_val),1e-5)),0);
end