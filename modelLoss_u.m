function [loss_u,gradients_u] = modelLoss_u(parameters_u,parameters_p,X,X0,UV0,k,M,rho,c)
% Make predictions with the initial conditions.
U = model(parameters_u,X);
Ur = U(1,:);
Ui = U(2,:);

% Trial neural network
P = model(parameters_p,X);
Pr = (1-X)*UV0(1)+X*UV0(3)/X0(4)+(X0(4)-X).*X.*P(1,:);
Pi = (1-X)*UV0(2)+X*UV0(4)/X0(4)+(X0(4)-X).*X.*P(2,:);

% Calculate derivatives with respect to X.
Urx = dlgradient(sum(Ur,'all'),X,'EnableHigherDerivatives',true);
Uix = dlgradient(sum(Ui,'all'),X,'EnableHigherDerivatives',true);
Prx = dlgradient(sum(Pr,'all'),X,'EnableHigherDerivatives',true);
Pix = dlgradient(sum(Pi,'all'),X,'EnableHigherDerivatives',true);

% Calculate loss.
f_r = M*Urx-k*Ui+(1/(rho*c))*Prx;
f_i = M*Uix+k*Ur+(1/(rho*c))*Pix;
zeroTarget_r = zeros(size(f_r),"like",f_r);
zeroTarget_i = zeros(size(f_i),"like",f_i);
loss_r = l2loss(f_r, zeroTarget_r);
loss_i = l2loss(f_i, zeroTarget_i);

% Calculate total loss
loss_u = loss_r + loss_i;

% Calculate gradients with respect to the learnable parameters.
gradients_u = dlgradient(loss_u,parameters_u);

end
