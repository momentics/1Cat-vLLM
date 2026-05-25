# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch


def _require_flash_attn_v100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for flash_attn_v100")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (7, 0):
        pytest.skip("flash_attn_v100 kernels are only validated on SM70")
    return pytest.importorskip("flash_attn_v100")


def _prefill_as_decode(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    softmax_scale: float,
) -> torch.Tensor:
    from flash_attn_v100 import flash_attn_decode_paged

    q_len = q.shape[1]
    q_flat = q.squeeze(0).contiguous()
    seq_lens = seq_len - q_len + torch.arange(
        1, q_len + 1, device=q.device, dtype=torch.int32)
    decode_block_table = block_table.expand(q_len, -1).contiguous()
    out = torch.empty_like(q_flat)
    flash_attn_decode_paged(
        q_flat,
        key_cache,
        value_cache,
        decode_block_table,
        seq_lens,
        softmax_scale=softmax_scale,
        out=out,
        kv_cache_dtype="auto",
    )
    return out.unsqueeze(0)


def _prefill_as_decode_with_padding(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    real_q_len: int,
    softmax_scale: float,
) -> torch.Tensor:
    from flash_attn_v100 import flash_attn_decode_paged

    padded_q_len = q.shape[1]
    q_flat = q.squeeze(0).contiguous()
    token_indices = torch.arange(
        padded_q_len, device=q.device, dtype=torch.int32)
    seq_lens = seq_len - real_q_len + token_indices + 1
    padding_mask = token_indices >= real_q_len
    seq_lens = torch.where(padding_mask, torch.zeros_like(seq_lens), seq_lens)
    decode_block_table = block_table.expand(padded_q_len, -1).contiguous()
    decode_block_table = torch.where(
        padding_mask[:, None],
        torch.zeros_like(decode_block_table),
        decode_block_table,
    ).contiguous()
    out = torch.empty_like(q_flat)
    flash_attn_decode_paged(
        q_flat,
        key_cache,
        value_cache,
        decode_block_table,
        seq_lens,
        softmax_scale=softmax_scale,
        out=out,
        kv_cache_dtype="auto",
    )
    return out.unsqueeze(0)


@pytest.mark.parametrize("q_len", [2, 4, 8, 9])
@pytest.mark.parametrize("block_size", [16, 832])
def test_flash_attn_v100_small_query_prefill_matches_decode(
    q_len: int,
    block_size: int,
):
    _require_flash_attn_v100()
    from flash_attn_v100 import flash_attn_prefill_paged

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    seq_len = 4096
    num_blocks = (seq_len + block_size - 1) // block_size
    num_q_heads = 6
    num_kv_heads = 1
    head_dim = 256
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn((1, q_len, num_q_heads, head_dim),
                    device=device,
                    dtype=dtype)
    key_cache = torch.randn((num_blocks, block_size, num_kv_heads, head_dim),
                            device=device,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(num_blocks, device=device,
                               dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)

    ref = flash_attn_prefill_paged(
        q,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        softmax_scale=softmax_scale,
        out=torch.empty_like(q),
        kv_cache_dtype="auto",
        causal=True,
    )
    actual = _prefill_as_decode(q, key_cache, value_cache, block_table,
                                seq_len, softmax_scale)

    torch.testing.assert_close(actual, ref, atol=1e-3, rtol=1e-2)


def test_flash_attn_v100_small_query_padding_rows_are_dummy():
    _require_flash_attn_v100()
    from flash_attn_v100 import flash_attn_prefill_paged

    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.float16
    real_q_len = 5
    padded_q_len = 8
    seq_len = 4096
    block_size = 16
    num_blocks = (seq_len + block_size - 1) // block_size
    num_q_heads = 6
    num_kv_heads = 1
    head_dim = 256
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn((1, padded_q_len, num_q_heads, head_dim),
                    device=device,
                    dtype=dtype)
    key_cache = torch.randn((num_blocks, block_size, num_kv_heads, head_dim),
                            device=device,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(num_blocks, device=device,
                               dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)

    ref = flash_attn_prefill_paged(
        q[:, :real_q_len],
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        softmax_scale=softmax_scale,
        out=torch.empty_like(q[:, :real_q_len]),
        kv_cache_dtype="auto",
        causal=True,
    )
    actual = _prefill_as_decode_with_padding(q, key_cache, value_cache,
                                             block_table, seq_len, real_q_len,
                                             softmax_scale)

    torch.testing.assert_close(actual[:, :real_q_len], ref, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(
        actual[:, real_q_len:],
        torch.zeros_like(actual[:, real_q_len:]),
        atol=0,
        rtol=0,
    )
