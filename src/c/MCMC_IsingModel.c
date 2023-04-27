/* 
 * mex file for compiling a function that performs Markov-chain Monte-Carlo sampling 
 * from a pairwise Ising model
 *
 * This is part of the package for inferring a pairwise Ising model from neural data using
 * the data-driven algorithm presented in
 * 'Learning Maximal Entropy Models from finite size datasets: 
 * a fast Data-Driven algorithm allows to sample from the posterior distribution'
 * by U. Ferrari, Phys. Rev. E, 94, 2, 2018
 * Please cite this paper if this code has been useful for you
 *
 *
 */


#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int offset(int i, int length) 
{
    return (length + i * (length - 2) - (i * (i - 1)) / 2 - 1); 
}

double intPow(double b,int exp)
{
 int i;
 double result=b;
 for(i=1;i<exp;i++){ result=result*b; }
 return result;
}

void MCMC_IsingModel(int N, int B, int first_steps, int logTd, double *J, int *latticeIn, double* p, int *latticeOut)
{
    
    FILE *devran = fopen("/dev/urandom","r");
    unsigned int myrand;
    fread(&myrand, 4, 1, devran);
    fclose(devran);
    /* myrand=1234567 */;
    srand(myrand);
    
    
    
    int lenght = (N*(N+1))/2;
    int i,j,k,n;
    for (i=0;i<lenght;i++) { p[i]  = 0.0; }
    
    int lattice[N];
    for(i=0;i<N;i++)   lattice[i]=latticeIn[i];

    double expJ[N][N];
    for (i=0;i<N;i++) {
        expJ[i][i] = 1.0;
        int off=offset(i, N);
        for (j=i+1;j<N;j++) {
            expJ[i][j] = exp( J[off + j] );
            expJ[j][i] = exp( J[off + j] );
        }
    }
    
    
    
    double h_local[N];
    double expHlocal[N];
    
    int decTime =intPow(2,logTd);
    
    for (i=0;i<N;i++) {
        expHlocal[i] = exp(-J[i] );
        for (j=0;j<N;j++) if(lattice[j]==1) expHlocal[i] /= expJ[i][j];
    }
    
    
    int nWindows,steps,loopTd;
    
    /*  Thermalization  */
    for (nWindows=0;nWindows< first_steps;nWindows++) {
        for(steps=intPow(2,nWindows); steps < intPow(2,nWindows+1); steps++ ){
            for (loopTd=0; loopTd< intPow(2,logTd);loopTd++){
                for(n=0; n<N;n++){
                    k=rand()%N;
                    double expMinusDelta = (lattice[k]==1 ? expHlocal[k] : 1.0/expHlocal[k]); /* = exp( -DeltaE ) */
                    if ( expMinusDelta>1.0 || rand() < expMinusDelta * RAND_MAX  ) {
                        lattice[k]=1.0 - lattice[k];
                        if(lattice[k]==1) for(i=0;i<N;i++) expHlocal[ i ] = expHlocal[ i ] / expJ[k][i];
                        else for(i=0;i<N;i++) expHlocal[ i ] = expHlocal[ i ] * expJ[k][i];
                    }
                }
            }
        }
    }
    
    
    /*  Sampling  */
    
    int b;
    for (b=0;b< B;b++) {
        for(steps=0; steps < intPow(2,logTd); steps++ ){
            for(n=0; n<N;n++){
                k=rand()%N;
                double expMinusDelta = (lattice[k]==1 ? expHlocal[k] : 1.0/expHlocal[k]); /* = exp( -DeltaE ) */
                if ( expMinusDelta>1.0 || rand() < expMinusDelta * RAND_MAX  ) {
                    lattice[k]=1-lattice[k];
                    if(lattice[k]==1) for(i=0;i<N;i++) expHlocal[ i ] = expHlocal[ i ] / expJ[k][i];
                    else for(i=0;i<N;i++) expHlocal[ i ] = expHlocal[ i ] * expJ[k][i];
                }
            }
        }
        for (i=0;i<N;i++) {
            p[i] = p[i] + lattice[i];
            if (lattice[i]==1) {
                int off=offset(i, N);
                for (j=i+1;j<N;j++) p[off + j] += lattice[j];
            }
        }
    }
    for (i=0;i<lenght;i++)  p[i]/= B;
    
    for (i=0;i<N;i++) latticeOut[i]=lattice[i];
}



/* The gateway function 
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    
    int N;
    int B;
    int first_steps;
    int logTd;
    int *latticeIn;
    int *latticeOut;
    double *J;
    double *p;
    
    if(nrhs!=6) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","six inputs required.");
    }
    if(nlhs!=2) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","Two output required.");
    }
    
    N = mxGetScalar(prhs[0]);
    B = mxGetScalar(prhs[1]);
    first_steps = mxGetScalar(prhs[2]);
    logTd = mxGetScalar(prhs[3]);
    
    J = mxGetPr(prhs[4]);
    latticeIn = mxGetPr(prhs[5]);
    
    plhs[0] = mxCreateDoubleMatrix(1, N*(N+1)/2,mxREAL);
    p = mxGetPr(plhs[0]);
    
    plhs[1] = mxCreateDoubleMatrix(1, N,mxREAL);
    latticeOut = mxGetPr(plhs[1]);
    
    MCMC_IsingModel(N, B,first_steps, logTd, J, latticeIn,  p, latticeOut);
    
}


*/


