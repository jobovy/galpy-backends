__global__ void k_div(double *a){ int i=threadIdx.x; a[i]=a[i]/a[i+1]; }
__global__ void k_sqrt(double *a){ int i=threadIdx.x; a[i]=sqrt(a[i]); }
__global__ void k_pow(double *a){ int i=threadIdx.x; a[i]=pow(a[i],a[i+1]); }
__global__ void k_pow15(double *a){ int i=threadIdx.x; a[i]=pow(a[i],1.5); }
__global__ void k_log(double *a){ int i=threadIdx.x; a[i]=log(a[i]); }
__global__ void k_exp(double *a){ int i=threadIdx.x; a[i]=exp(a[i]); }
