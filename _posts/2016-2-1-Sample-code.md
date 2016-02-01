---
layout: post
title: Sample code
---

A simple CUDA kernel:

{% highlight Cuda %}
template <int direction, bool conjugate>
__global__ void pack_unpack_z_cols_gpu_kernel
(
    cuDoubleComplex* z_cols_packed__,
    cuDoubleComplex* fft_buf__,
    int size_x__,
    int size_y__,
    int size_z__,
    int num_z_cols__,
    int const* z_columns_pos__
)
{
    int icol = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y;
    /* comment */
    if (icol < num_z_cols__)
    {
        int x, y;

        if (conjugate)
        {
            x = (-z_columns_pos__[array2D_offset(0, icol, 2)] + size_x__) % size_x__;
            y = (-z_columns_pos__[array2D_offset(1, icol, 2)] + size_y__) % size_y__;
        }
        else
        {
            x = (z_columns_pos__[array2D_offset(0, icol, 2)] + size_x__) % size_x__;
            y = (z_columns_pos__[array2D_offset(1, icol, 2)] + size_y__) % size_y__;
        }

        /* load into buffer */
        if (direction == 1)
        {
            if (conjugate)
            {
                fft_buf__[array3D_offset(x, y, iz, size_x__, size_y__)] = cuConj(z_cols_packed__[array2D_offset(iz, icol, size_z__)]);
            }
            else
            {
                fft_buf__[array3D_offset(x, y, iz, size_x__, size_y__)] = z_cols_packed__[array2D_offset(iz, icol, size_z__)];
            }
        }
        if (direction == -1)
        {
            z_cols_packed__[array2D_offset(iz, icol, size_z__)] = fft_buf__[array3D_offset(x, y, iz, size_x__, size_y__)];
        }
    }
}
{% endhighlight %}
