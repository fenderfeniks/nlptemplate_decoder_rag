import sys
import pytest
import torch
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from src.pipelines.base.core.models.modifiers import (
    EmbeddingResizeModifier,
    PEFTModifier,
    FullFinetuningModifier,
)


class TestEmbeddingResizeModifier:
    def test_no_resize_needed(self):
        """Если размеры словаря совпадают, модель не должна меняться."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.__len__.return_value = 1000
        
        mock_model = MagicMock()
        mock_model.config.vocab_size = 1000

        modifier = EmbeddingResizeModifier(tokenizer=mock_tokenizer)
        result = modifier(mock_model)

        mock_model.resize_token_embeddings.assert_not_called()
        assert result is mock_model

    def test_resize_tied_embeddings(self):
        """Проверка ресайза, когда input и output эмбеддинги — это одна матрица (tied)."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.__len__.return_value = 100
        
        mock_model = MagicMock()
        mock_model.config.vocab_size = 50

        emb_weight = torch.zeros((100, 10))
        emb_weight[:50, :] = 1.0
        
        mock_input_emb = MagicMock()
        mock_input_emb.weight.data = emb_weight
        mock_model.get_input_embeddings.return_value = mock_input_emb
        mock_model.get_output_embeddings.return_value = mock_input_emb

        modifier = EmbeddingResizeModifier(tokenizer=mock_tokenizer, pad_to_multiple_of=8)
        modifier(mock_model)

        mock_model.resize_token_embeddings.assert_called_once_with(100, pad_to_multiple_of=8)
        assert torch.all(emb_weight[50:] == 1.0).item()

    def test_resize_separate_output_embeddings(self):
        """Проверка инициализации, когда у модели отдельный lm_head (не tied weights)."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.__len__.return_value = 100
        mock_model = MagicMock()
        mock_model.config.vocab_size = 50

        mock_input = MagicMock()
        mock_input.weight.data = torch.zeros((100, 10))
        mock_input.weight.data[:50, :] = 1.0

        mock_output = MagicMock()
        mock_output.weight.data = torch.zeros((100, 10))
        mock_output.weight.data[:50, :] = 2.0

        mock_model.get_input_embeddings.return_value = mock_input
        mock_model.get_output_embeddings.return_value = mock_output

        modifier = EmbeddingResizeModifier(tokenizer=mock_tokenizer)
        modifier(mock_model)

        # Новые строки input должны стать 1.0, а output — 2.0
        assert torch.all(mock_input.weight.data[50:] == 1.0).item()
        assert torch.all(mock_output.weight.data[50:] == 2.0).item()


class MockLoraConfig:
    """Фейковый класс для обхода ошибки isinstance()."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class TestPEFTModifier:
    def test_new_adapter_quantized(self):
        mock_peft = MagicMock()
        mock_peft.LoraConfig = MockLoraConfig  # Подменяем на настоящий класс
        
        mock_peft_model = MagicMock()
        mock_peft_model.get_nb_trainable_parameters.return_value = (100, 1000)
        
        mock_peft.prepare_model_for_kbit_training.return_value = MagicMock()
        mock_peft.get_peft_model.return_value = mock_peft_model

        with patch.dict(sys.modules, {'peft': mock_peft}):
            peft_cfg = {"r": 8, "lora_alpha": 16}
            modifier = PEFTModifier(peft_config=peft_cfg, lora_resume_path=None, is_quantized=True)
            
            mock_model = MagicMock()
            result = modifier(mock_model)

            mock_peft.prepare_model_for_kbit_training.assert_called_once_with(
                mock_model, use_gradient_checkpointing=True
            )
            mock_peft.get_peft_model.assert_called_once()
            assert result is mock_peft_model

    def test_resume_adapter_unquantized(self):
        mock_peft = MagicMock()
        mock_peft.LoraConfig = MockLoraConfig
        mock_peft_model = MagicMock()
        mock_peft_model.get_nb_trainable_parameters.return_value = (100, 1000)
        
        mock_peft.PeftModel.from_pretrained.return_value = mock_peft_model

        with patch.dict(sys.modules, {'peft': mock_peft}):
            modifier = PEFTModifier(
                peft_config={}, 
                lora_resume_path="/path/to/lora", 
                is_quantized=False,
                gradient_checkpointing=True
            )
            
            mock_model = MagicMock()
            result = modifier(mock_model)

            mock_model.gradient_checkpointing_enable.assert_called_once_with({"use_reentrant": False})
            mock_peft.PeftModel.from_pretrained.assert_called_once_with(
                mock_model, "/path/to/lora", is_trainable=True
            )
            assert result is mock_peft_model

    def test_peft_config_dictconfig_and_loraconfig(self):
        mock_peft = MagicMock()
        mock_peft.LoraConfig = MockLoraConfig
        mock_peft_model = MagicMock()
        mock_peft_model.get_nb_trainable_parameters.return_value = (1, 10)
        mock_peft.get_peft_model.return_value = mock_peft_model

        with patch.dict(sys.modules, {'peft': mock_peft}):
            # 1. Поведение при DictConfig
            dict_cfg = OmegaConf.create({"r": 16})
            mod1 = PEFTModifier(peft_config=dict_cfg, is_quantized=False, gradient_checkpointing=False)
            mod1(MagicMock())
            # Проверяем, что get_peft_model был вызван с инстансом нашего MockLoraConfig и параметром r=16
            call_args = mock_peft.get_peft_model.call_args[0]
            assert isinstance(call_args[1], MockLoraConfig)
            assert call_args[1].kwargs["r"] == 16

            # 2. Поведение при готовом LoraConfig
            lora_cfg = MockLoraConfig(r=32)
            mod2 = PEFTModifier(peft_config=lora_cfg, is_quantized=False, gradient_checkpointing=False)
            mod2(MagicMock())
            call_args = mock_peft.get_peft_model.call_args[0]
            assert call_args[1] is lora_cfg


class TestFullFinetuningModifier:
    def test_full_finetuning_unfreezes_all_params(self):
        """Проверка разморозки параметров и включения checkpointing."""
        mock_model = MagicMock()
        
        param1 = MagicMock(requires_grad=False)
        param1.numel.return_value = 10
        param2 = MagicMock(requires_grad=True)
        param2.numel.return_value = 20
        mock_model.parameters.return_value = [param1, param2]

        modifier = FullFinetuningModifier(gradient_checkpointing=True)
        result = modifier(mock_model)

        mock_model.gradient_checkpointing_enable.assert_called_once_with({"use_reentrant": False})
        assert param1.requires_grad is True
        assert param2.requires_grad is True
        assert result is mock_model

    def test_full_finetuning_no_checkpointing(self):
        """Проверка ветки, когда checkpointing выключен."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = []

        modifier = FullFinetuningModifier(gradient_checkpointing=False)
        modifier(mock_model)

        mock_model.gradient_checkpointing_enable.assert_not_called()