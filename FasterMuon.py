"""
FastMuon: 一个把 Newton-Schulz 正交化批量化的 Muon 优化器实现。
 
背景
----
Muon 的更新分两种完全不同性质的计算：
 
1. 逐元素部分（动量更新、权重衰减、应用最终更新）——很便宜，并且跟参数
   的具体形状无关，可以用 `torch._foreach_*` 批量处理任意形状混杂的参数
   列表（跟 Adam 的 foreach 路径是同一套机制）。
 
2. Newton-Schulz 正交化——真正烧 GPU 时间的部分，需要实打实的矩阵乘法
   （`X @ X.T`、`A @ A` ...）。`torch._foreach_*` 这个算子族里没有"批量
   矩阵乘法"，所以 PyTorch 官方的 Muon 实现在这一步只能退回 Python
   for 循环，每个参数单独 matmul 一次（源码里写的是
   `"Foreach is not supported for Muon yet"`）。
 
这份实现对 (2) 做了批量化：把形状完全相同的参数的梯度（经过动量更新后）
堆叠成一个 `(K, out, in)` 的 3D tensor，用 `torch.bmm` 一次性跑完整批的
Newton-Schulz 迭代，而不是 K 次串行的 matmul。在 GPU 上，这相当于把 K 次
kernel launch 合并成几次，对 transformer 里大量"形状相同"的权重矩阵
（比如每一层的 Q/K/V/O 投影、每一层同维度的 MLP 矩阵）收益最明显。
 
形状在整批参数里只出现一次、找不到同伴可以batch的，会自动退回普通的
单矩阵路径——没什么可批的，批不出收益。
"""
 
import math
from collections import defaultdict
 
import torch
from torch import Tensor
 
__all__ = ["FasterMuon"]
 
# @torch.compile
def _zeropower_via_newtonschulz5_single(
    G: Tensor,
    steps: int,
    eps: float = 1e-7,
    coeffs: tuple = (3.4445, -4.7750, 2.0315),
) -> Tensor:
    """参考版本：一次只处理一个矩阵（用于落单形状的 fallback）。"""
    a, b, c = coeffs
    X = G.bfloat16()
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X
 
# @torch.compile
def _zeropower_via_newtonschulz5_batched(
    G: Tensor,
    steps: int,
    eps: float = 1e-7,
    coeffs: tuple = (3.4445, -4.7750, 2.0315),
) -> Tensor:
    """
    同样的迭代，但 G 的形状是 (K, out, in) —— K 个完全同形状的矩阵，
    用 torch.bmm 一次跑完，而不是 K 次 Python 循环里各自 matmul 一次。
    """
    a, b, c = coeffs
    X = G.bfloat16()
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.transpose(-2, -1)
    # 注意：这里是每个矩阵各自的 Frobenius norm，不是整个 batch 混在一起的 norm
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = torch.bmm(X, X.transpose(-2, -1))
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)
    if transpose:
        X = X.transpose(-2, -1)
    return X
 
 
class FasterMuon(torch.optim.Optimizer):
    """
    Muon 优化器，按参数形状分组后用 bmm 批量做 Newton-Schulz 正交化。
 
    只负责 Muon（==2D，隐藏层权重）那部分参数。其它的bias、norm、embedding、
    lm_head 这些请照常配一个独立的 torch.optim.AdamW，用法和官方 Muon
    实现完全一致。
    """
 
    def __init__(
        self,
        params,
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ):
        params = list(params)
        for p in params:
            if p.ndim != 2:
                raise ValueError(
                    f"FasterMuon 只支持 2D 参数，但收到了形状 {tuple(p.shape)}"
                )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
        )
        super().__init__(params, defaults)
 
    @staticmethod
    def _adjust_lr(lr: float, shape: torch.Size) -> float:
        A, B = shape[:2]
        return lr * 0.2 * math.sqrt(max(A, B))
 
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
 
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
 
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]
 
            # ---- 1. 动量更新：逐元素，跟形状无关，直接 foreach 批量做 ----
            bufs = []
            for p, g in zip(params, grads):
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                bufs.append(state["momentum_buffer"])
 
            torch._foreach_lerp_(bufs, grads, 1 - momentum)
            if nesterov:
                updates = torch._foreach_lerp(grads, bufs, momentum)
            else:
                updates = [b.clone() for b in bufs]
 
            # ---- 2. Newton-Schulz 正交化：按形状分组，同形状的一起 bmm ----
            shape_groups = defaultdict(list)
            for idx, p in enumerate(params):
                shape_groups[tuple(p.shape)].append(idx)
            # print(len(shape_groups))
            orthogonalized: list = [None] * len(params)
            for shape, idxs in shape_groups.items():
                if len(idxs) == 1:
                    i = idxs[0]
                    orthogonalized[i] = _zeropower_via_newtonschulz5_single(
                        updates[i], steps=ns_steps, eps=eps
                    )
                else:
                    stacked = torch.stack([updates[i] for i in idxs], dim=0)
                    batched = _zeropower_via_newtonschulz5_batched(
                        stacked, steps=ns_steps, eps=eps
                    )
                    for j, i in enumerate(idxs):
                        orthogonalized[i] = batched[j]
 
            # ---- 3. 权重衰减 + 应用更新：逐元素，per-param 的 lr 缩放先用
            #          ScalarList 的 foreach_mul_ 折进 update 里，再用一次
            #          统一 alpha 的 foreach_add_ 写回参数 ----
            torch._foreach_mul_(params, 1 - lr * weight_decay)
            adjusted_lrs = [self._adjust_lr(lr, p.shape) for p in params]
            torch._foreach_mul_(orthogonalized, adjusted_lrs)
            torch._foreach_add_(params, orthogonalized, alpha=-1.0)
 
        return loss
