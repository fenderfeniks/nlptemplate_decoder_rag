import sys
import pytest
import torch
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from transformers import BitsAndBytesConfig

from src.pipelines.base.core.models.builder import HFModelBuilder


class TestHFModelBuilderUtils:
    def test_parse_torch_dtype(self):
        """Проверка парсинга типов весов."""
        assert HFModelBuilder._parse_torch_dtype("auto") == "auto"
        assert HFModelBuilder._parse_torch_dtype("float16") is torch.float16
        assert HFModelBuilder._parse_torch_dtype("bfloat16") is torch.bfloat16
        
        with pytest.raises(ValueError, match="Недопустимое значение torch_dtype"):
            HFModelBuilder._parse_torch_dtype("invalid_dtype")

    def test_parse_bnb_config(self):
        """Проверка обработки конфигурации квантизации."""
        # Пустой конфиг
        assert HFModelBuilder._parse_bnb_config(None) is None
        assert HFModelBuilder._parse_bnb_config({}) is None
        
        # DictConfig из Hydra с конвертацией dtype
        dict_cfg = OmegaConf.create({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16"
        })
        bnb = HFModelBuilder._parse_bnb_config(dict_cfg)
        assert isinstance(bnb, BitsAndBytesConfig)
        assert bnb.load_in_4bit is True
        assert bnb.bnb_4bit_compute_dtype is torch.float16

        # Ошибка при неизвестном dtype
        bad_cfg = OmegaConf.create({"bnb_4bit_compute_dtype": "unknown"})
        with pytest.raises(ValueError, match="Недопустимый dtype"):
            HFModelBuilder._parse_bnb_config(bad_cfg)

    @patch("torch.cuda.is_available")
    def test_resolve_device_map(self, mock_cuda_is_available):
        """Проверка резолва device_map."""
        builder = HFModelBuilder("fake", device_map="auto_cuda")
        
        # CUDA доступна
        mock_cuda_is_available.return_value = True
        with patch("torch.cuda.current_device", return_value=0):
            assert builder._resolve_device_map(None) == {"": 0}
            
        # CUDA недоступна
        mock_cuda_is_available.return_value = False
        assert builder._resolve_device_map(None) is None
        
        # Запрос квантизации без CUDA должен кидать ошибку
        with pytest.raises(RuntimeError, match="квантизация требует CUDA"):
            builder._resolve_device_map(MagicMock(spec=BitsAndBytesConfig))

    def test_parse_bnb_config_instance(self):
        """Проверка, что строковые dtype в готовом инстансе BitsAndBytesConfig заменяются на torch.*."""
        # Создаем фейковый инстанс, где dtype передан строкой
        bnb = BitsAndBytesConfig()
        bnb.bnb_4bit_compute_dtype = "float16"
        
        result = HFModelBuilder._parse_bnb_config(bnb)
        assert result.bnb_4bit_compute_dtype is torch.float16


class TestHFModelBuilderAttention:
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_capability")
    def test_resolve_attn_implementation(self, mock_get_cap, mock_is_avail):
        """Проверка логики выбора Flash Attention 2."""
        # Если запрошено не FA2, возвращаем as-is
        assert HFModelBuilder._resolve_attn_implementation("sdpa") == "sdpa"
        assert HFModelBuilder._resolve_attn_implementation(None) is None

        # Эмулируем отсутствие пакета flash_attn
        with patch.dict(sys.modules, {'flash_attn': None}):
            assert HFModelBuilder._resolve_attn_implementation("flash_attention_2") == "sdpa"
        
        # Эмулируем наличие пакета, но нет CUDA
        mock_flash = MagicMock()
        with patch.dict(sys.modules, {'flash_attn': mock_flash}):
            mock_is_avail.return_value = False
            assert HFModelBuilder._resolve_attn_implementation("flash_attention_2") is None
            
            # Есть CUDA, но старая архитектура (< 8)
            mock_is_avail.return_value = True
            mock_get_cap.return_value = (7, 5)
            assert HFModelBuilder._resolve_attn_implementation("flash_attention_2") == "sdpa"

            # Есть CUDA и подходящая архитектура
            mock_get_cap.return_value = (8, 0)
            assert HFModelBuilder._resolve_attn_implementation("flash_attention_2") == "flash_attention_2"


class TestHFModelBuilderBuild:
    @patch("src.pipelines.base.core.models.builder.importlib.import_module")
    def test_build_full_flow(self, mock_import):
        """Проверка основного метода build()."""
        # Мокаем класс модели и токенизатор
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model_instance
        
        # Мокаем модуль (transformers)
        mock_module = MagicMock()
        mock_module.AutoModelForCausalLM = mock_model_class
        mock_import.return_value = mock_module

        # Фейковый модификатор
        mock_modifier = MagicMock()
        mock_modifier.return_value = mock_model_instance

        # Подготавливаем билдер
        builder = HFModelBuilder(
            model_name_or_path="fake/model",
            auto_model_class="transformers.AutoModelForCausalLM",
            torch_dtype="float32",
            attn_implementation="sdpa",
            device_map="auto"
        )
        
        # Подменяем метод _build_modifiers, чтобы не тестировать внутренности Hydra
        with patch.object(builder, "_build_modifiers", return_value=[mock_modifier]):
            result = builder.build(tokenizer=MagicMock())

        # Проверяем, что была вызвана загрузка базовой модели с правильными параметрами
        mock_model_class.from_pretrained.assert_called_once_with(
            "fake/model",
            cache_dir=None,
            quantization_config=None,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            device_map="auto",
            attn_implementation="sdpa"
        )
        
        # Проверяем, что модификатор был применен
        mock_modifier.assert_called_once_with(mock_model_instance)
        assert result is mock_model_instance

    @patch("src.pipelines.base.core.models.builder.importlib.import_module")
    def test_build_rope_scaling_and_none_attn(self, mock_import):
        """Проверка передачи rope_scaling и отсутствия ключа attn_implementation при None."""
        mock_model_class = MagicMock()
        mock_module = MagicMock()
        mock_module.AutoModel = mock_model_class
        mock_import.return_value = mock_module

        # DictConfig для RoPE
        rope_cfg = OmegaConf.create({"type": "dynamic", "factor": 2.0})

        builder = HFModelBuilder(
            model_name_or_path="fake",
            auto_model_class="transformers.AutoModel",
            rope_scaling=rope_cfg,
            attn_implementation=None, # Отключаем attention
            device_map=None
        )
        
        with patch.object(builder, "_build_modifiers", return_value=[]):
            builder.build()

        call_kwargs = mock_model_class.from_pretrained.call_args[1]
        
        # Проверяем, что RoPE конвертировался в словарь
        assert call_kwargs["rope_scaling"] == {"type": "dynamic", "factor": 2.0}
        # Проверяем, что attn_implementation вообще не передан в kwargs
        assert "attn_implementation" not in call_kwargs
        

class TestHFModelBuilderModifiers:
    def test_build_modifiers_empty(self):
        """Если modifiers_cfg пуст или None, возвращается пустой список."""
        builder = HFModelBuilder("fake", modifiers=None)
        assert builder._build_modifiers(tokenizer=None) == []

    @patch("src.pipelines.base.core.models.builder._locate")
    @patch("src.pipelines.base.core.models.builder.instantiate")
    def test_build_modifiers_logic(self, mock_instantiate, mock_locate):
        """Проверка резолва маркеров и инстанцирования модификаторов."""
        # Создаем класс-заглушку с нужными маркерами
        class DummyModifier:
            _needs_tokenizer = True
            _needs_lora_path = True

        mock_locate.return_value = DummyModifier
        mock_instantiate.return_value = "modifier_instance"

        builder = HFModelBuilder("fake", modifiers={"test_mod": {"_target_": "dummy.Target"}})
        builder.lora_resume_path = "/path/to/lora"

        # 1. Проверяем ошибку, если модификатор требует токенизатор, а его нет
        with pytest.raises(ValueError, match="требует tokenizer"):
            builder._build_modifiers(tokenizer=None)

        # 2. Проверяем успешный проход
        mock_tokenizer = MagicMock()
        result = builder._build_modifiers(tokenizer=mock_tokenizer)

        assert result == ["modifier_instance"]
        mock_instantiate.assert_called_once_with(
            {"_target_": "dummy.Target"},
            tokenizer=mock_tokenizer,
            lora_resume_path="/path/to/lora"
        )