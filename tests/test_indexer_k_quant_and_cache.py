
import torch
import torch.nn.functional as F
import pytest
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types

# Mocking or extracting necessary parts from vllm for the test
eps = 1e-4


def _pytorch_group_quant(
    x: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_ue8m0 is None:
        use_ue8m0 = False

    if dtype is None:
        dtype = torch.float8_e4m3fn

    # Validate inputs
    assert x.shape[-1] % group_size == 0, (
        f"Last dimension {x.shape[-1]} must be divisible by group_size {group_size}"
    )
    assert x.stride(-1) == 1, "Input tensor groups must be contiguous"

    # Prepare output tensor
    if out_q is None:
        x_q = torch.empty_like(x, dtype=dtype)
    else:
        assert out_q.shape == x.shape
        x_q = out_q

    # Reshape input for group processing
    # Original shape: (..., last_dim)
    # Target shape: (..., num_groups, group_size)
    original_shape = x.shape
    num_groups = original_shape[-1] // group_size

    # Reshape to separate groups
    group_shape = original_shape[:-1] + (num_groups, group_size)
    x_grouped = x.view(group_shape)

    # Compute per-group absolute maximum values
    # Shape: (..., num_groups)
    abs_max = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=False)
    abs_max = torch.maximum(abs_max, torch.tensor(eps, device=x.device, dtype=x.dtype))

    # Compute scales
    FP8_MAX = torch.finfo(dtype).max
    FP8_MIN = torch.finfo(dtype).min
    scale_raw = abs_max / FP8_MAX

    if use_ue8m0:
        # For UE8M0 format, scales must be powers of 2
        scales = torch.pow(2.0, torch.ceil(torch.log2(scale_raw)))
    else:
        scales = scale_raw

    # Expand scales for broadcasting with grouped data
    # Shape: (..., num_groups, 1)
    scales_expanded = scales.unsqueeze(-1)

    # Quantize the grouped data
    x_scaled = x_grouped / scales_expanded
    x_clamped = torch.clamp(x_scaled, FP8_MIN, FP8_MAX)
    x_quantized = x_clamped.to(dtype)

    # Reshape back to original shape
    x_q.copy_(x_quantized.view(original_shape))

    # Prepare scales tensor in requested format
    if column_major_scales:
        # Column-major: (num_groups,) + batch_dims
        # Transpose the scales to put group dimension first
        scales_shape = (num_groups,) + original_shape[:-1]
        x_s = scales.permute(-1, *range(len(original_shape) - 1))
        x_s = x_s.contiguous().view(scales_shape)
    else:
        # Row-major: batch_dims + (num_groups,)
        x_s = scales.contiguous()

    # Ensure scales are float32
    return x_q, x_s.float()

def _pytorch_indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
) -> None:
    head_dim = k.shape[-1]
    num_tokens = k.shape[0]
    num_groups = head_dim // quant_block_size
    block_size = kv_cache.shape[1]
    cache_stride = kv_cache.shape[2]

    k_fp8, k_scale = _pytorch_group_quant(
        k,
        group_size=quant_block_size,
        column_major_scales=False,
        use_ue8m0=(scale_fmt == "ue8m0"),
    )

    # [num_tokens, head_dim]
    k_fp8_bytes = k_fp8.view(torch.uint8)  

    # [num_blocks * block_size * cache_stride]
    kv_cache_flat_bytes = kv_cache.view(-1)         
    kv_cache_flat_float = kv_cache_flat_bytes.view(torch.float32)  


    for i, slot_idx in enumerate(slot_mapping.flatten().tolist()):
        if slot_idx < 0:
            continue
        block_idx = slot_idx // block_size
        block_offset = slot_idx % block_size

        fp8_start = block_idx * block_size * cache_stride + block_offset * head_dim
        kv_cache_flat_bytes[fp8_start: fp8_start + head_dim] = k_fp8_bytes[i]

        for g in range(num_groups):
            scale_float_idx = block_idx * block_size * cache_stride + block_size * head_dim + (block_offset * head_dim + g * quant_block_size) * 4 // quant_block_size
            kv_cache_flat_float[scale_float_idx // 4] = k_scale[i, g]
    kv_cache.copy_(kv_cache_flat_bytes.view(kv_cache.shape))

def test_indexer_k_quant_and_cache_correctness():
    device = "xpu"
    if not torch.xpu.is_available():
        print("XPU not available, skipping")
        return

    num_tokens = 17   # test multi-block
    head_dim = 256    # multi-group
    quant_block_size = 128
    block_size = 16
    num_blocks = (num_tokens + block_size - 1) // block_size  # = 8
    # scale_fmt = "ue8m0"
    scale_fmt = "fp8e4m3" 

    torch.manual_seed(42)
    k = torch.randn((num_tokens, head_dim), device=device, dtype=torch.float32)

    num_groups = head_dim // quant_block_size
    cache_stride = head_dim + num_groups * 4  

    kv_cache_ref = torch.zeros((num_blocks, block_size, cache_stride), dtype=torch.uint8, device=device)
    kv_cache_xpu = torch.zeros((num_blocks, block_size, cache_stride), dtype=torch.uint8, device=device)

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # Run Reference
    _pytorch_indexer_k_quant_and_cache(
        k, kv_cache_ref, slot_mapping, quant_block_size, scale_fmt
    )

    # Run XPU kernel
    from vllm import _custom_ops as ops
    ops.indexer_k_quant_and_cache(
        k, kv_cache_xpu, slot_mapping, quant_block_size, scale_fmt
    )
    print(f"kv_cache_xpu: {kv_cache_xpu}")


    # in block, fp8 should be in first block_size*head_dim bytes, scale in the rest
    for block_idx in range(num_blocks):
        block_ref = kv_cache_ref[block_idx].view(-1)  # [block_size * cache_stride]
        block_xpu = kv_cache_xpu[block_idx].view(-1)

        fp8_end = block_size * head_dim
        ref_fp8 = block_ref[:fp8_end]
        out_fp8 = block_xpu[:fp8_end]

        # reinterpret scale to float
        ref_scale = block_ref[fp8_end:].view(torch.float32)
        out_scale = block_xpu[fp8_end:].view(torch.float32)

        print(f"Block {block_idx} FP8 match: {torch.equal(ref_fp8, out_fp8)}")
        if not torch.equal(ref_fp8, out_fp8):
            diff = (ref_fp8.view(torch.float8_e4m3fn).float() -
                    out_fp8.view(torch.float8_e4m3fn).float()).abs()
            print(f"  Max FP8 diff: {diff.max()}")

        print(f"Block {block_idx} Scale match: {torch.allclose(ref_scale, out_scale, atol=1e-5)}")
        if not torch.allclose(ref_scale, out_scale, atol=1e-5):
            print(f"  Ref scale: {ref_scale[:num_tokens]}")
            print(f"  Out scale: {out_scale[:num_tokens]}")

if __name__ == "__main__":
    test_indexer_k_quant_and_cache_correctness()
