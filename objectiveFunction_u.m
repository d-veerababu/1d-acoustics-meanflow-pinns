function [loss_u,gradientsV_u] = objectiveFunction_u(parametersV_u,parameters_p,X,X0,UV0,parameterNames_u,parameterSizes_u,k,M,rho,c)

% Convert parameters to structure of dlarray objects.
parametersV_u = dlarray(parametersV_u);
parameters_u = parameterVectorToStruct(parametersV_u,parameterNames_u,parameterSizes_u);

% Evaluate model loss and gradients.
[loss_u,gradients_u] = dlfeval(@modelLoss_u,parameters_u,parameters_p,X,X0,UV0,k,M,rho,c);

% Return loss and gradients for fmincon.
gradientsV_u = parameterStructToVector(gradients_u);
gradientsV_u = extractdata(gradientsV_u);
loss_u = extractdata(loss_u);

end
