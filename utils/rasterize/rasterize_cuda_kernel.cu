#include <torch/extension.h>
#include <torch/data/iterator.h>
#include <cuda.h>
#include <cuda_runtime.h>
  
#include <device_launch_parameters.h>
#include <curand_kernel.h> 

#include <vector>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}


__device__ void substract3d(float results[3], float arr1[3], float arr2[3])
{
    results[0] = arr1[0] - arr2[0];
    results[1] = arr1[1] - arr2[1];
    results[2] = arr1[2] - arr2[2];
}

__device__ void compute_I(float I[3], float P0[3], float dir[3], float r)
{
    // I = P0 + r * dir;            # intersect point of ray and plane
    I[0] = P0[0] + r * dir[0];
    I[1] = P0[1] + r * dir[1];
    I[2] = P0[2] + r * dir[2];
}

__device__ void cross(float results[3], float arr1[3], float arr2[3])
{

    float a1 = arr1[0];
    float a2 = arr1[1];
    float a3 = arr1[2];

    float b1 = arr2[0];
    float b2 = arr2[1];
    float b3 = arr2[2];
    results[0] = a2 * b3 - a3 * b2;
    results[1] = a3 * b1 - a1 * b3;
    results[2] = a1 * b2 - a2 * b1;
}

__device__ float dot(float arr1[3], float arr2[3])
{
    return arr1[0] * arr2[0] + arr1[1] * arr2[1] + arr1[2] * arr2[2];
}

__device__ float abs(float val)
{
    if (val>=0)
        return val;
    else
        return -val; 
}

__device__ float grad_updat_rule(float V2_shifted, float V2, float voxel, float grad_voxel, float diag )
{

    float d = V2_shifted - V2;
    float grad_V2 = 0;
    if (voxel == 0) // voxel(x,y,z) is outside the mesh
    {
        if (d * grad_voxel < 0) 
            grad_V2 = -grad_voxel * abs(V2_shifted - V2)/diag; 
    }
    else // voxel(x,y,z) is insde the mesh
    {
        if (d * grad_voxel > 0)
            grad_V2 = -grad_voxel * abs(V2_shifted - V2)/diag; 
    }

    return grad_V2;
}
 
__device__ void plane_from_3_ponts(float n[3], float P1[3], float P2[3], float P3[3])
{
    float u[3];
    float v[3]; 

    substract3d(u, P2, P1);
    substract3d(v, P3, P1);

    cross(n, u, v); 

    // n is the normal to the plane
}

__device__ bool P_is_on_face(float P[3], float V1[3], float V0[3], float V2_x[3])
{

    float w1[3]; // V0-V2_new
    float w2[3]; // V1-V2_new
    float w3[3]; // P-V2_new
    bool is_on_face = false;

    substract3d(w1, V0, V2_x);
    substract3d(w2, V1, V2_x);
    substract3d(w3, P, V2_x);

    float a = (w3[0]/w2[0] - w3[1]/w2[1])/(w1[0]/w2[0] - w1[1]/w2[1]);
    float b = (w3[0]/w1[0] - w3[1]/w1[1])/(w2[0]/w1[0] - w2[1]/w1[1]);

    if (a >= 0 && b >= 0 && a+b<=1)
      is_on_face = true;

    return is_on_face;

    // n is the normal to the plane
}

__device__ void compute_grad(float grad_V2[3], float V0[3], float V1[3], float V2[3], float P[3], float voxel, float grad_voxel, float diag)
{
    // First get the plane going through P, V0 and V1
    float n[3];
    plane_from_3_ponts(n, P, V0, V1);

    // Now, find the intersectoin x,y,z values
    // when shifting V2 along three axes with above plane
    // eg: when shifting along x-axis and finding x (V2_new[0]),
    // x = -(ny*V2y + nz*V2z)/nx
    float V2_x[3];
    float V2_y[3];
    float V2_z[3];

    V2_x[0] = -(n[1]*V2[1] + n[2]*V2[2])/n[0];
    V2_x[1] = V2[1];
    V2_x[2] = V2[2];

    V2_y[0] = V2[0];
    V2_y[1] = -(n[0]*V2[0] + n[2]*V2[2])/n[1];
    V2_y[2] = V2[2];        
    
    V2_z[0] = V2[0];
    V2_z[1] = V2[1];
    V2_z[2] = -(n[1]*V2[1] + n[0]*V2[0])/n[2];

    // Check whether P is inside face V0, V1 and V2_new
    // we solve the equation a * (V0-V2_new) + b * (V1-V2_new) = P - V2_new
    // if a, b has solution, P is on the plane (this should be siincee we found P to be so)
    // if a >= 0, b >= 0 and a + b <= 1, P is on edges or within (and on) the face
    // This gives us three equations and has two variables(a, b)
    if (n[0] != 0 && (V2_x[0] - V2[0]) != 0 && P_is_on_face(P, V1, V0, V2_x)==true)
    {
        grad_V2[0] = grad_updat_rule(V2_x[0], V2[0], voxel, grad_voxel, diag); 
    }
    else
      grad_V2[0] = 0;

    if (n[1] != 0 && (V2_y[1] - V2[1]) != 0 && P_is_on_face(P, V1, V0, V2_y)==true)
    { 
      grad_V2[1] = grad_updat_rule(V2_y[1], V2[1], voxel, grad_voxel, diag); 
    }
    else
      grad_V2[1] = 0;
    
    if (n[2] != 0 && (V2_z[2] - V2[2]) != 0 && P_is_on_face(P, V1, V0, V2_z)==true)
    { 
      grad_V2[2] = grad_updat_rule(V2_z[2], V2[2], voxel, grad_voxel, diag); 
    }
    else
      grad_V2[2] = 0;

    // n is the normal to the plane
}

template <typename scalar_t>
__global__ void rasterize_cuda_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> volume,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_vertices,
    const torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> d_faces,
    const torch::PackedTensorAccessor32<int32_t,1,torch::RestrictPtrTraits> shape
    ) 
{


    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    int D = shape[0];
    int H = shape[1];
    int W = shape[2];
    // auto v = torch::tensor({z, y, x});
    if (x >= W || y >= H || z >= D) {
        return;
    }
    // float D = shape[0];
    // float shape_[3];
    // shape_ = shape.data<float>();



    float P1[3]; 
    float P0[] = {x, y, z}; 
    float u[3];
    float v[3];
    float n[3];
    float w0[3];
    float w[3];
    float dir[3];

    float r, s, t, a, b, uu, vv, uv, wu, wv, d;
    float V0[3];
    float V1[3];
    float V2[3];

    float I[3];

    int done = 0;


    curandState state;
    //    curand_init((unsigned long long)clock() + H * W * z + W * y + x, 0, 0, &state);
    curand_init(H * W * z + W * y + x, 0, 0, &state);
    //    curand_init(0, 0, 0, &state);


    //   d_debug[H * W * z + W * y + x] = -100000;
    int debug_count = 0;
    while (done==0){
        done = 1;
        int border_intersection = 0;
        int face_intersection = 0;

        P1[0] = W * (float)curand_uniform_double(&state);;
        P1[1] = H * (float)curand_uniform_double(&state);
        P1[2] = D;


        for (int i = 0; i < d_faces.size(0); i++)
        {

            V0[0] = d_vertices[d_faces[i][0]][0];          
            V0[1] = d_vertices[d_faces[i][0]][1];
            V0[2] = d_vertices[d_faces[i][0]][2];

            V1[0] = d_vertices[d_faces[i][1]][0];          
            V1[1] = d_vertices[d_faces[i][1]][1];
            V1[2] = d_vertices[d_faces[i][1]][2];

            V2[0] = d_vertices[d_faces[i][2]][0];          
            V2[1] = d_vertices[d_faces[i][2]][1];
            V2[2] = d_vertices[d_faces[i][2]][2];


            // get triangle edge vectors and plane normal
            substract3d(u, V1, V0);
            substract3d(v, V2, V0);

            cross(n, u, v);

            substract3d(dir, P1, P0);   // ray direction vector
            substract3d(w0, P0, V0);


            a = -dot(n,w0);
            b = dot(n,dir);


            if (abs(b) < 0.00000001)     // ray is  parallel to triangle plane ---
                continue;

            // get intersect point of ray with triangle plane
            r = a / b;
            if (r < 0.0)                 // ray goes away from triangle
                continue;                   // => no intersect

            compute_I(I, P0, dir, r);

            // is I inside T?
            uu = dot(u,u);
            uv = dot(u,v);
            vv = dot(v,v);

            substract3d(w, I, V0);

            wu = dot(w,u);
            wv = dot(w,v);

            d = uv * uv - uu * vv;

            // get and test parametric coordsc
            s = (uv * wv - vv * wu) / d;
            if (s < 0.0 || s > 1.0)        // I is outside T
                continue;

            t = (uv * wu - uu * wv) / d;
            if (t < 0.0 || (s + t) > 1.0)  // I is outside T
                continue;

            if (s == 0 || t == 0 || s+t == 1)
            {
                border_intersection = border_intersection + 1;
                // face_intersection = face_intersection + 1;
            }
            else
            {
                face_intersection = face_intersection + 1;



            }
        }


        if (border_intersection > 0)
              continue;

        if (face_intersection % 2 == 1)
            // volume[H * W * x + W * y + z] = 255;
            volume[z][y][x] = 1;
        else
            // volume[H * W * x + W * y + z] = 0;
            volume[z][y][x] = 0;



    //        d_volume[H * W * z + W * y + x] = face_intersection;
        done = 1;


    }
}

template <typename scalar_t>
__global__ void rasterize_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_vertices,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> volume,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_volume,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> vertices,
    const torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> faces,
    const torch::PackedTensorAccessor32<int32_t,1,torch::RestrictPtrTraits> shape
    ) 
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    float P[] = {x, y, z}; 

    int D = shape[0];
    int H = shape[1];
    int W = shape[2];

    // auto v = torch::tensor({z, y, x});
    if (x >= W || y >= H || z >= D) {
        return;
    }

    float diag = 1.732 * ((float) D); // (D * D + H * H + W * W);

    float voxel = volume[z][y][x];
    float grad_voxel = grad_volume[z][y][x];
  
    float V0[3];
    float V1[3];
    float V2[3];
    for (int i = 0; i < faces.size(0); i++)
    {
 
        V0[0] = vertices[faces[i][0]][0];          
        V0[1] = vertices[faces[i][0]][1];
        V0[2] = vertices[faces[i][0]][2];

        V1[0] = vertices[faces[i][1]][0];          
        V1[1] = vertices[faces[i][1]][1];
        V1[2] = vertices[faces[i][1]][2];

        V2[0] = vertices[faces[i][2]][0];          
        V2[1] = vertices[faces[i][2]][1];
        V2[2] = vertices[faces[i][2]][2];


        float grad_V0[3];
        float grad_V1[3];
        float grad_V2[3];

        // TODO compute_grad: should take volume, grad_volume at loc P as input when computing grad_V0
        compute_grad(grad_V0, V1, V2,  V0, P, voxel, grad_voxel, diag);
        compute_grad(grad_V1, V0, V2,  V1, P, voxel, grad_voxel, diag);
        compute_grad(grad_V2, V0, V1,  V2, P, voxel, grad_voxel, diag);

        // grad_vertices[faces[i][0]][0] = grad_V0[0];
        // grad_vertices[faces[i][0]][1] = grad_V0[1];
        // grad_vertices[faces[i][0]][2] = grad_V0[2]; 

        // grad_vertices[faces[i][1]][0] = grad_V1[0];
        // grad_vertices[faces[i][1]][1] = grad_V1[1];
        // grad_vertices[faces[i][1]][2] = grad_V1[2];

        // grad_vertices[faces[i][2]][0] = grad_V2[0];
        // grad_vertices[faces[i][2]][1] = grad_V2[1];
        // grad_vertices[faces[i][2]][2] = grad_V2[2]; 

        atomicAdd( &(grad_vertices[faces[i][0]][0]), grad_V0[0]);
        atomicAdd( &(grad_vertices[faces[i][0]][1]), grad_V0[1]);
        atomicAdd( &(grad_vertices[faces[i][0]][2]), grad_V0[2]); 

        atomicAdd( &(grad_vertices[faces[i][1]][0]), grad_V1[0]);
        atomicAdd( &(grad_vertices[faces[i][1]][1]), grad_V1[1]);
        atomicAdd( &(grad_vertices[faces[i][1]][2]), grad_V1[2]); 

        atomicAdd( &(grad_vertices[faces[i][2]][0]), grad_V2[0]);
        atomicAdd( &(grad_vertices[faces[i][2]][1]), grad_V2[1]);
        atomicAdd( &(grad_vertices[faces[i][2]][2]), grad_V2[2]); 
        

    }




    

}
} // namespace

std::vector<torch::Tensor> rasterize_cuda_forward(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor shape) {

  int D = shape[0].item<int>();;
  int H = shape[1].item<int>();;
  int W = shape[2].item<int>();;
  // https://pytorch.org/cppdocs/notes/tensor_creation.html
  auto options =torch::TensorOptions()
                  .dtype(torch::kFloat32)
                  .layout(torch::kStrided)
                  .device(torch::kCUDA, 0)
                  .requires_grad(false);
  torch::Tensor volume = torch::zeros({D, H, W}, options);  

  const dim3 block(32, 16, 2);
  const dim3 grid(ceil(W / 32.0), ceil(H / 16.0), ceil(D / 2.0));

  AT_DISPATCH_FLOATING_TYPES(volume.type(), "rasterize_forward_cuda", ([&] {
    rasterize_cuda_forward_kernel<scalar_t><<<grid, block>>>(
        volume.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        vertices.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        faces.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
        shape.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>());
  }));
 
  return {volume};
}

std::vector<torch::Tensor> rasterize_cuda_backward(
    torch::Tensor volume,
    torch::Tensor grad_volume, 
    torch::Tensor vertices, 
    torch::Tensor faces, 
    torch::Tensor shape) {

  int D = shape[0].item<int>();;
  int H = shape[1].item<int>();;
  int W = shape[2].item<int>();;
  // // https://pytorch.org/cppdocs/notes/tensor_creation.html
  int Vn = vertices.size(0);
  // printf("v count %d\n", Vn );
  auto options =torch::TensorOptions()
                  .dtype(torch::kFloat32)
                  .layout(torch::kStrided)
                  .device(torch::kCUDA, 0)
                  .requires_grad(false);
  torch::Tensor grad_vertices = torch::zeros({Vn, 3}, options);  

  const dim3 block(32, 16, 2);
  const dim3 grid(ceil(W / 32.0), ceil(H / 16.0), ceil(D / 2.0));

 

  AT_DISPATCH_FLOATING_TYPES(grad_vertices.type(), "rasterize_forward_cuda", ([&] {
    rasterize_cuda_backward_kernel<scalar_t><<<grid, block>>>(
        grad_vertices.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        volume.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_volume.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        vertices.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        faces.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
        shape.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>());
  }));
 

  return {grad_vertices};
}
