from typing import Tuple
import numbers
from typing import Optional, Union

import torch
from torch import Size
from torch.nn import Module
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter

import transformer_engine
from transformer_engine_torch import rmsnorm_bwd, rmsnorm_fwd


if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    # TODO
    pass


SM_MARGIN = 0
ZERO_CENTERED_GAMMA = False

_shape_t = Union[int, list[int], Size]


@_torch_custom_op_wrapper("te::_rmsnorm_fwd", mutates_args=(), device_types="cuda")
def _rmsnorm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    eps:float ,
) -> Tuple[torch.Tensor, torch.Tensor]:
    y, rstdevs = rmsnorm_fwd(x, w, eps, SM_MARGIN, ZERO_CENTERED_GAMMA)

    return y, rstdevs


@_torch_register_fake_wrapper("te::_rmsnorm_fwd")
def _rmsnorm_fwd_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    eps:float ,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty(x.size(0), device=x.device)


@_torch_custom_op_wrapper("te::_rmsnorm_bwd", mutates_args=("dy",), device_types="cuda")
def _rmsnorm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    rstdevs: torch.Tensor,
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dx, dw = rmsnorm_bwd(dy ,x, rstdevs, w, SM_MARGIN, ZERO_CENTERED_GAMMA)

    return dx, dw


@_torch_register_fake_wrapper("te::_rmsnorm_bwd")
def _rmsnorm_bwd_fake(
    dy: torch.Tensor,
    x: torch.Tensor,
    rstdevs: torch.Tensor,
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(w)


if torch.__version__ >= "2.4.0":
    _wrapped_rmsnorm_fwd = torch.ops.te._rmsnorm_fwd
    _wrapped_rmsnorm_bwd = torch.ops.te._rmsnorm_bwd
else:
    # TODO:
    pass


class TEFusedRMSNormFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        eps:float ,
    ) -> torch.Tensor:
        b, s, h = x.size()
        x = x.view(-1, h)
        out, rstdevs = _wrapped_rmsnorm_fwd(x, w, eps)

        ctx.save_for_backward(rstdevs, x, w,)

        return out.view(b, s, h)

    @staticmethod
    def backward(
        ctx, 
        grad_output: torch.Tensor
    ):
        (rstdevs, x, w) = ctx.saved_tensors
        b, s, h = grad_output.size()
        grad_output = grad_output.view(-1, h)
        dx, dw = _wrapped_rmsnorm_bwd(grad_output, x, rstdevs, w)

        return dx.view(b, s, h), dw, None 


class TERMSNorm(Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: Optional[float]
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs forward pass.
        """
        return TEFusedRMSNormFunc.apply(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        """
        Extra information about the module.
        """
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


if __name__ == "__main__":
    b, s ,h = 6, 8192, 8192
    torch.cuda.manual_seed(1234)

    x_orig = torch.randn(b, s, h, dtype=torch.bfloat16, device='cuda')
    w = torch.randn(h, dtype=torch.bfloat16, device='cuda')

    eps = 1e-6
    x = x_orig.view(b*s, h)
    y, rstdevs = rmsnorm_fwd(x, w, eps, SM_MARGIN, ZERO_CENTERED_GAMMA)
    y = y.view(b,s,h)

    x_ref = x_orig.transpose(0, 1).contiguous().view(s*b,h)
    y_ref, rstdevs_ref = rmsnorm_fwd(x_ref, w, eps, SM_MARGIN, ZERO_CENTERED_GAMMA)

    y_ref = y_ref.view(s,b,h).transpose(0,1).contiguous()
    
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-7)
