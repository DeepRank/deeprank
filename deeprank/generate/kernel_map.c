#include <math.h>

__global__ void gaussian(float alpha, float x0, float y0, float z0, float *xvect, float *yvect, float *zvect, float *out)
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

__global__ void atomic_densities(float vdw_radius, float x0, float y0, float z0, float *xvect, float *yvect, float *zvect, float *out)
{

    /*
    the formula is equation (1) of the Koes paper
    Protein-Ligand Scoring with Convolutional NN Arxiv:1612.02751v1
    */

    // 3D thread
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    int tz = threadIdx.z + blockDim.z * blockIdx.z;

    if ( ( tx < %(nx)s ) && (ty < %(ny)s) && (tz < %(nz)s) )
    {

        float dx = xvect[tx] - x0;
        float dy = yvect[ty] - y0;
        float dz = zvect[tz] - z0;
        float d = sqrt(dx*dx + dy*dy + dz*dz);

        float e = exp(1.0);
        float e2 = e*e;
        float d2 = d*d;
        float vdw2 = vdw_radius*vdw_radius;

        if (d < vdw_radius)
            out[ty * %(nx)s * %(nz)s + tx * %(nz)s + tz] += exp(-2.*d2/vdw2);
        else if (d < 1.5*vdw_radius)
            out[ty * %(nx)s * %(nz)s + tx * %(nz)s + tz] += 4.*d2/(e2*vdw2) - 12.*d/(e2*vdw_radius) + 9./e2;

    }

}