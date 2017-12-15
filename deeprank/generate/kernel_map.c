#include <math.h>
__global__ void AddGrid(float alpha, float x0, float y0, float z0, float *xvect, float *yvect, float *zvect, float *out)
{
	
	// 3D thread 
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    int tz = threadIdx.z + blockDim.z * blockIdx.z;

    float beta = 1.0/%(RES)s;

    if ( ( tx < %(nx)s ) && (ty < %(ny)s) && (tz < %(nz)s) )
    {

    	float dx = xvect[tx] - x0;
    	float dy = yvect[ty] - y0;
    	float dz = zvect[tz] - z0;
    	float d = sqrt(dx*dx + dy*dy + dz*dz);
    	out[ty * %(nx)s * %(nz)s + tx * %(nz)s + tz] += alpha*exp(-beta*d);
    }
}