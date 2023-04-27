function [ q,  output ] = dataDriven_IsingModel( N, B, Bmin, logTd, p, chi, eta, alpha, epsThreshold, nStepMore, nWork ,verbose )
% Infer the Ising couplings via the data-driven algorithm with L2
% regularization
% N = Number of spins
% B = number of snapshots
% Bmin = minimal number of MCMC steps
% logTd = decorrelation time for the MCMC
% mean of sufficient statistics
% chi = covariance of sufficient statistics
% alpha = initial learning rate
% epsThreshold = early stop condition
% nStepMore = number of step after convergence
% nWork = numebr of parallel workers

% Code for inferring a pairwise Ising model from neural data using
% the data-driven algorithm presented in
% 'Learning Maximal Entropy Models from finite size datasets: 
% a fast Data-Driven algorithm allows to sample from the posterior distribution'
% by U. Ferrari, Phys. Rev. E, 94, 2, 2018
% Please cite this paper if this code has been useful for you



p=p(:)';
global params;

D=N*(N+1)/2;
alphaMin = 10^(-10);
alphaMax=64.0;

time=0;
%INITIALIZATION LATTICE
lattice = uint32(rand([nWork N]) < repmat(p(1:N) ,[nWork 1]));

qMat= zeros([nWork D]);
parfor (nW=1:nWork,nWork)
    [qMat(nW,:), lattice(nW,:)] = MCMC_IsingModel(N, Bmin/nWork,floor( log2(B)-2 ),logTd,params,lattice(nW,:));
end
q = mean(qMat);

chiEta = chi + eta ;

fOfX = -(eta*params)';
%fOfX = fOfX + mvnrnd( zeros([1,D]), eta/B ) ; % for non-diagonal regularization
fOfX = fOfX + normrnd( zeros([1,D]), sqrt(diag(eta/B))' ) ;

grad = p - q + fOfX;
contraGrad = ( chiEta \  grad' );
epsOpMc = sqrt( 0.5*B*(  grad* contraGrad )/ D);
epsOpMcOld = epsOpMc;


tic
step=0;
Beff = max(min(B/(epsOpMcOld),B),Bmin);
if verbose
    disp('    Step - learning rate - ratio of MCMC steps - eps old - eps new');
    disp([step , alpha, Beff/B, log(epsOpMcOld), log(epsOpMc)]);
end
output = [ step, alpha , epsOpMc , Beff];
while (epsOpMc > epsThreshold) && (step<150)
    

    jListTry = params + alpha * contraGrad;
  
    qMatTry = zeros([nWork D]);
    parfor (nW=1:nWork,nWork)
        [qMatTry(nW,:), lattice(nW,:)] = MCMC_IsingModel(N, Beff/nWork,floor( log2(B)-4 ),logTd,jListTry,lattice(nW,:));
    end
    qTry = mean(qMatTry);
    step = step+1;
    
    fOfX = -(eta*jListTry)';
    %fOfX = fOfX + mvnrnd( zeros([1,D]), eta/B ) ; % for non-diagonal regularization
    fOfX = fOfX + normrnd( zeros([1,D]), sqrt(diag(eta/B))' ) ;

    gradTry = p - qTry + fOfX;
    contraGradTry = ( chiEta \  gradTry' ) ;
    
    epsOpMc = sqrt(  0.5*B*(  ( p - qTry + fOfX )  * ( chiEta \  ( p - qTry + fOfX)' )  )  / D) ;
    dEps = epsOpMc - epsOpMcOld;
    
    if dEps<0
        alpha = min(alphaMax,alpha*1.05);
        q = qTry;
        grad = gradTry;
        contraGrad = contraGradTry;
        epsOpMcOld = epsOpMc;
        params = jListTry;
    else
        alpha = max(alpha/sqrt(2.0),alphaMin);
    end
    
    if verbose
        disp([step , alpha, Beff/B, log(epsOpMcOld), log(epsOpMc)]);
    end
    
    Beff = max(min(B/(epsOpMcOld),B),Bmin);

    output = [output; step, alpha , epsOpMc , Beff ];
    
end
time=toc;

if verbose
    disp(['Inference done, now thermalization']);
end
alpha = alpha/1.05;
grad = p - q + fOfX ;
for ss=1:nStepMore
        
    params = params + alpha * (chiEta \ grad' );
    
    qMat= zeros([nWork D]);
    parfor (nW=1:nWork,nWork)
        [qMat(nW,:), lattice(nW,:)] = MCMC_IsingModel(N,Beff/nWork,floor( log2(B)-3 ),logTd+1,params,lattice(nW,:));
    end
    q = mean(qMat);
    
    fOfX = -(eta*params)';
    %fOfX = fOfX + mvnrnd( zeros([1,D]), eta/B ) ; % for non-diagonal regularization
    fOfX = fOfX + normrnd( zeros([1,D]), sqrt(diag(eta/B))' ) ;
        
        
    grad = p - q + fOfX ;
    epsOpMc = sqrt( 0.5*B*(grad * ( chiEta \ grad' ) )/ D );
    if epsOpMc > epsThreshold
        alpha = alpha/sqrt(2.0);
    else
       alpha = alpha*1.05; 
    end

    if verbose
        disp([step , alpha, ss, log(epsOpMc)])
    end
end
%toc;




end

