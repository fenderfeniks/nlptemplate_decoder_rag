import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from transformers import BitsAndBytesConfig

from src.decoder_pipeline.core.models.builder import HFModelBuilder
from src.decoder_pipeline.core.models.modifiers import (
    EmbeddingResizeModifier,
    FullFinetuningModifier,
    PEFTModifier,
)
from src.decoder_pipeline.core.models.tokenization import HFTokenizerBuilder


# ИСПРАВЛЕНИЕ: Теперь заглушка принимает любые kwargs (например, r=8), как и оригинальный класс
class DummyLoraConfig:
    def __init__(self, *args, **kwargs):
        pass


class TestHFModelBuilder:
    @patch.dict(sys.modules, {"flash_attn": MagicMock()})
    @patch("src.decoder_pipeline.core.models.builder.torch.cuda.is_available", return_value=False)
    def test_attn_implementation_fallback_on_cpu(self, mock_cuda):
        impl = HFModelBuilder._resolve_attn_implementation("flash_attention_2")
        assert impl is None

    @patch.dict(sys.modules, {"flash_attn": MagicMock()})
    @patch("src.decoder_pipeline.core.models.builder.torch.cuda.is_available", return_value=True)
    def test_attn_implementation_fallback_on_old_gpu(self, mock_cuda):
        with patch("src.decoder_pipeline.core.models.builder.torch.cuda.get_device_capability", return_value=(7, 5)):
            impl = HFModelBuilder._resolve_attn_implementation("flash_attention_2")
            assert impl == "sdpa"

    @patch("src.decoder_pipeline.core.models.builder.importlib.import_module")
    @patch("src.decoder_pipeline.core.models.builder.torch.cuda.is_available", return_value=True)
    @patch("src.decoder_pipeline.core.models.builder.torch.cuda.current_device", return_value=0)
    def test_build_converts_bnb_dtypes(self, mock_current_device, mock_cuda_avail, mock_import):
        raw_cfg = OmegaConf.create({"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"})
        builder = HFModelBuilder("test", quantization_config=raw_cfg)
        
        mock_class = MagicMock()
        mock_import.return_value.AutoModelForCausalLM = mock_class
        
        builder.build()
        kwargs = mock_class.from_pretrained.call_args[1]
        assert kwargs["quantization_config"].bnb_4bit_compute_dtype == torch.bfloat16


class TestModifiers:
    def test_embedding_resize_modifier(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.__len__.return_value = 32005
        
        mock_model = MagicMock()
        mock_model.config.vocab_size = 32000
        
        weight_mock = torch.rand(32005, 128)
        mock_model.get_input_embeddings.return_value.weight.data = weight_mock
        mock_model.get_output_embeddings.return_value = None
        
        modifier = EmbeddingResizeModifier(tokenizer=mock_tokenizer)
        modifier(mock_model)
        
        mock_model.resize_token_embeddings.assert_called_once_with(32005)

    def test_peft_modifier_initializes_lora(self):
        mock_peft = MagicMock()
        mock_peft.LoraConfig = DummyLoraConfig
        mock_model = MagicMock()
        mock_peft.prepare_model_for_kbit_training.return_value = mock_model
        mock_peft.get_peft_model.return_value.get_nb_trainable_parameters.return_value = (100, 1000)
        
        with patch.dict(sys.modules, {"peft": mock_peft}):
            modifier = PEFTModifier(peft_config={"r": 8}, is_quantized=True)
            modifier(mock_model)
            
            mock_peft.prepare_model_for_kbit_training.assert_called_once()
            mock_peft.get_peft_model.assert_called_once()

    def test_full_finetuning_modifier_unfreezes_all_params(self):
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.requires_grad = False
        mock_param.numel.return_value = 100
        mock_model.parameters.return_value = [mock_param]
        
        modifier = FullFinetuningModifier(gradient_checkpointing=True)
        modifier(mock_model)
        
        mock_model.gradient_checkpointing_enable.assert_called_once_with({"use_reentrant": False})
        assert mock_param.requires_grad is True