function [P,U,Z]= solve1DWaveEqn(X,k,M,p_lb,p_rb,rho,c)

kc_p = k/(1+M);
kc_m = k/(1-M);

C1 = (p_rb-p_lb*exp(1j*kc_m))/(exp(-1j*kc_p)-exp(1j*kc_m));
C2 = 1-C1;

% Initialize solutions.
P = zeros(size(X));
U = zeros(size(X));
Z = zeros(size(X));

% Loop over x values.
for i = 1:numel(X)
    x = X(i);

    % Calculate the solutions using the integral function. The boundary
    % conditions in x = -1 and x = 1 are known, so leave 0 as they are
    % given by initialization of U.
        p = C1*exp(-1j*kc_p*x)+C2*exp(1j*kc_m*x);
        u = (C1*exp(-1j*kc_p*x)-C2*exp(1j*kc_m*x))/(rho*c);
        P(i) = p;
        U(i) = u;
        Z(i) = p/u;
end

end
