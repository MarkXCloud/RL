# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types

import pytest
import torch

pytestmark = pytest.mark.vllm


@pytest.fixture()
def fp8_module():
    pytest.importorskip("vllm")

    from nemo_rl.models.generation.vllm.quantization import fp8

    old_config = fp8.global_fp8_config
    old_state = fp8.fp8_state
    old_patches_applied = fp8.fp8_patches_applied
    fp8.global_fp8_config = None
    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False

    try:
        yield fp8
    finally:
        fp8.global_fp8_config = old_config
        fp8.fp8_state = old_state
        fp8.fp8_patches_applied = old_patches_applied


def test_init_fp8_uses_mxfp8_quantization_config(fp8_module, monkeypatch):
    fp8 = fp8_module
    applied_configs = []

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(
        fp8,
        "monkey_patch_vllm_ray_executor",
        lambda fp8_config: applied_configs.append(fp8_config),
    )
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM_E8M0", raising=False)

    vllm_kwargs = fp8.init_fp8(
        {
            "precision": "fp8",
            "kv_cache_dtype": "auto",
            "async_engine": False,
            "is_mx": True,
            "use_deep_gemm": True,
        },
        "dummy-model",
        model_parallel_size=1,
    )

    assert vllm_kwargs == {
        "quantization": "fp8",
        "kv_cache_dtype": "auto",
        "hf_overrides": {"quantization_config": fp8.MXFP8_BLOCK_QUANT_KWARGS},
    }
    assert applied_configs == [fp8.global_fp8_config]
    assert fp8.global_fp8_config.is_mx is True
    assert "VLLM_USE_DEEP_GEMM" not in fp8.os.environ
    assert "VLLM_USE_DEEP_GEMM_E8M0" not in fp8.os.environ


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("pow2_weight_scaling_factors", "only pow2 weight scaling factors"),
        ("pow2_activation_scaling_factors", "only pow2 activation scaling factors"),
    ],
)
def test_init_fp8_rejects_non_pow2_mxfp8_scales(fp8_module, monkeypatch, field, error):
    fp8 = fp8_module

    monkeypatch.setattr(
        fp8.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: types.SimpleNamespace(num_hidden_layers=4),
    )
    monkeypatch.setattr(fp8, "monkey_patch_vllm_ray_executor", lambda _fp8_config: None)

    with pytest.raises(ValueError, match=error):
        fp8.init_fp8(
            {
                "precision": "fp8",
                "kv_cache_dtype": "auto",
                "async_engine": False,
                "is_mx": True,
                field: False,
            },
            "dummy-model",
            model_parallel_size=1,
        )


def test_apply_fp8_patches_registers_modelopt_patches_only_for_mxfp8(
    fp8_module, monkeypatch
):
    fp8 = fp8_module
    patched_paths = []

    class FakePatch:
        def __init__(self, path):
            self.path = path
            self.started = False

        def start(self):
            self.started = True

    def fake_patch(path, _replacement):
        patched_paths.append(path)
        return FakePatch(path)

    monkeypatch.setattr(fp8, "patch", fake_patch)

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(use_fp8_weights=True, model_parallel_size=1, is_mx=False),
    )
    assert not any("ModelOptMxFp8" in path for path in patched_paths)
    assert all(patcher.started for patcher in fp8.fp8_state.vllm_patches)

    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False
    patched_paths.clear()

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(
            use_fp8_weights=True,
            model_parallel_size=1,
            use_activation_pow2_scale=True,
        ),
    )
    assert any("per_token_group_quant_fp8" in path for path in patched_paths)
    assert all(patcher.started for patcher in fp8.fp8_state.vllm_patches)

    fp8.fp8_state = fp8.FP8State()
    fp8.fp8_patches_applied = False
    patched_paths.clear()

    fp8.apply_fp8_patches(
        None,
        fp8.FP8Config(use_fp8_weights=True, model_parallel_size=1, is_mx=True),
    )

    assert any("ModelOptMxFp8LinearMethod" in path for path in patched_paths)
    assert any("ModelOptMxFp8FusedMoE.create_weights" in path for path in patched_paths)
    assert any(
        "ModelOptMxFp8FusedMoE.process_weights_after_loading" in path
        for path in patched_paths
    )
    assert all(patcher.started for patcher in fp8.fp8_state.vllm_patches)


def test_get_module_from_param_name_applies_qwen35_prefix_mapper(fp8_module):
    fp8 = fp8_module
    model = torch.nn.Module()
    model.packed_modules_mapping = {}
    model.hf_to_vllm_mapper = types.SimpleNamespace(
        apply_list=lambda names: [
            name.replace("model.language_model.", "language_model.model.", 1)
            for name in names
        ]
    )
    model.language_model = torch.nn.Module()
    model.language_model.model = torch.nn.Module()
    model.language_model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    expected_module = torch.nn.Linear(4, 4, bias=False)
    model.language_model.model.layers[0].projection = expected_module

    actual_module = fp8._get_module_from_param_name(
        model, "model.language_model.layers.0.projection.weight"
    )

    assert actual_module is expected_module


@pytest.mark.parametrize(
    ("target_dtype", "expected"),
    [(torch.float8_e4m3fn, True), (torch.bfloat16, False)],
)
def test_is_fp8_weight_recognizes_grouped_experts(
    fp8_module, monkeypatch, target_dtype, expected
):
    fp8 = fp8_module

    class FakeFusedMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = types.SimpleNamespace(dtype=target_dtype)
            self.w2_weight = types.SimpleNamespace(dtype=target_dtype)

    monkeypatch.setattr(fp8, "FusedMoE", FakeFusedMoE)
    model = torch.nn.Module()
    model.packed_modules_mapping = {}
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.experts = FakeFusedMoE()

    assert fp8._is_fp8_weight("model.layers.0.mlp.experts.down_proj", model) is expected


def test_quantize_grouped_experts_blockwise_preserves_expert_boundaries(
    fp8_module, monkeypatch
):
    fp8 = fp8_module
    grouped_weight = torch.arange(2 * 128 * 128, dtype=torch.float32).reshape(
        2, 128, 128
    )
    flat_scale = torch.tensor([1.0, 2.0]).reshape(2, 1, 1)

    def fake_cast(flat_weight, weight_block_size):
        assert flat_weight.shape == (256, 128)
        assert weight_block_size == [128, 128]
        return flat_weight, flat_scale

    monkeypatch.setattr(fp8, "cast_tensor_to_fp8_blockwise", fake_cast)

    quantized, scale = fp8._quantize_grouped_experts_blockwise(grouped_weight)

    torch.testing.assert_close(quantized, grouped_weight)
    torch.testing.assert_close(scale, flat_scale.squeeze(-1).reshape(2, 1, 1))


@pytest.mark.parametrize(
    ("projection", "expected_calls"),
    [
        ("gate_up_proj", 2),
        ("down_proj", 1),
    ],
)
def test_quantize_grouped_moe_expert_blockwise(
    fp8_module, monkeypatch, projection, expected_calls
):
    fp8 = fp8_module
    fp8.global_fp8_config = fp8.FP8Config(is_mx=False)
    num_experts = 2
    out_features = 8 if projection == "gate_up_proj" else 4
    weight = torch.arange(num_experts * out_features * 3, dtype=torch.float32).reshape(
        num_experts, out_features, 3
    )
    grouped_projections = []

    def fake_quantize(grouped_projection):
        grouped_projections.append(grouped_projection.clone())
        scale = torch.full(
            (num_experts, 1, 1), len(grouped_projections), dtype=torch.float32
        )
        return grouped_projection, scale

    monkeypatch.setattr(fp8, "_quantize_grouped_experts_blockwise", fake_quantize)

    key = f"model.layers.0.mlp.experts.{projection}"
    quantized = fp8._quantize_grouped_moe_expert(key, weight)

    assert [name for name, _ in quantized] == [key, f"{key}_scale_inv"]
    torch.testing.assert_close(quantized[0][1], weight)
    assert len(grouped_projections) == expected_calls

    if projection == "gate_up_proj":
        torch.testing.assert_close(grouped_projections[0], weight[:, :4, :])
        torch.testing.assert_close(grouped_projections[1], weight[:, 4:, :])
        expected_scale = torch.cat(
            (torch.ones(num_experts, 1, 1), torch.full((num_experts, 1, 1), 2)),
            dim=1,
        )
    else:
        torch.testing.assert_close(grouped_projections[0], weight)
        expected_scale = torch.ones(num_experts, 1, 1)
    torch.testing.assert_close(quantized[1][1], expected_scale)


def test_quantize_grouped_moe_expert_mxfp8(fp8_module, monkeypatch):
    fp8 = fp8_module
    fp8.global_fp8_config = fp8.FP8Config(is_mx=True)
    original_weight = torch.ones(2, 4, 3)
    quantized_weight = torch.zeros_like(original_weight, dtype=torch.float8_e4m3fn)
    scale = torch.ones(2, 4, 1, dtype=torch.uint8)
    from vllm.model_executor.layers.quantization.utils import mxfp8_utils

    monkeypatch.setattr(
        mxfp8_utils,
        "mxfp8_e4m3_quantize",
        lambda weight: (quantized_weight, scale),
    )

    key = "model.layers.0.mlp.experts.down_proj"
    quantized = fp8._quantize_grouped_moe_expert(key, original_weight)

    assert [name for name, _ in quantized] == [
        key,
        f"{key}_scale_from_checkpoint",
    ]
    assert quantized[0][1] is quantized_weight
    assert quantized[1][1] is scale


def test_load_weights_routes_grouped_experts_to_backend_quantizer(
    fp8_module, monkeypatch
):
    fp8 = fp8_module
    fp8.global_fp8_config = fp8.FP8Config(is_mx=False)
    original_weight = torch.ones(2, 4, 3)
    quantized_weight = torch.zeros(4, 3)
    loaded_weights = []
    model = types.SimpleNamespace(
        load_weights=lambda weights: loaded_weights.extend(weights)
    )
    model_runner = types.SimpleNamespace(model=model)

    monkeypatch.setattr(fp8, "_is_fp8_weight", lambda _name, _model: True)
    monkeypatch.setattr(
        fp8,
        "_quantize_grouped_moe_expert",
        lambda _name, _weight: [("quantized.weight", quantized_weight)],
    )

    key = "model.layers.0.mlp.experts.down_proj"
    fp8.load_weights([(key, original_weight)], model_runner)

    assert loaded_weights[0][0] == "quantized.weight"
    assert loaded_weights[0][1] is quantized_weight
