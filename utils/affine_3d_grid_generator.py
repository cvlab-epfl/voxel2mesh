import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.backends.cudnn as cudnn

MODE_ZEROS = 0
MODE_BORDER = 1


def affine_grid(theta, size):
    return AffineGridGenerator.apply(theta, size)


# TODO: Port these completely into C++
class AffineGridGenerator(Function):

    @staticmethod
    def _enforce_cudnn(input):
        if not cudnn.enabled:
            raise RuntimeError("AffineGridGenerator needs CuDNN for "
                               "processing CUDA inputs, but CuDNN is not enabled")
        assert cudnn.is_acceptable(input)

    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size
        N, C, D, H, W = size
        ctx.size = size

        #ctx.is_cuda = True

        base_grid = theta.new(N, D, H, W, 4)

        w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
        h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
        d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)

        base_grid[:, :, :, :, 0] = w_points
        base_grid[:, :, :, :, 1] = h_points
        base_grid[:, :, :, :, 2] = d_points
        base_grid[:, :, :, :, 3] = 1
        ctx.base_grid = base_grid
        grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
        grid = grid.view(N, D, H, W, 3)
        return grid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        N, C, D, H, W = ctx.size
        assert grad_grid.size() == torch.Size([N, D, H, W, 3])
        base_grid = ctx.base_grid
        grad_theta = torch.bmm(
            base_grid.view(N, D * H * W, 4).transpose(1, 2),
            grad_grid.view(N, D * H * W, 3))
        grad_theta = grad_theta.transpose(1, 2)
        return grad_theta, None
