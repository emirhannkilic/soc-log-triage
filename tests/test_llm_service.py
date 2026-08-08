"""Unit tests for src/llm/service.py (PHISHING_ROUTING_PLAN.md hybrid
workflow follow-up — "ortak Qwen model servisi"). No real mlx_vlm call
anywhere in this file — QwenService is constructed with injected
load_fn/generate_fn/apply_chat_template_fn throughout, per the module's
own testability contract (see its docstring's TESTABILITY section)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.service import LLMServiceError, QwenService, get_service  # noqa: E402


def _mock_model_and_processor():
    model = MagicMock()
    model.config = "CONFIG"
    processor = MagicMock()
    return model, processor


# --- load() ---------------------------------------------------------------

def test_load_calls_load_fn_with_model_path():
    calls = []

    def load_fn(path):
        calls.append(path)
        return _mock_model_and_processor()

    service = QwenService(model_path=Path("/fake/model"), load_fn=load_fn)
    model, processor = service.load()

    assert calls == ["/fake/model"]
    assert model is not None
    assert processor is not None


def test_load_is_idempotent_only_loads_once():
    """A 4-bit 9B model reload costs minutes — load() must not call
    load_fn again once the model is already loaded in this instance."""
    call_count = {"n": 0}

    def load_fn(path):
        call_count["n"] += 1
        return _mock_model_and_processor()

    service = QwenService(load_fn=load_fn)
    service.load()
    service.load()
    service.load()

    assert call_count["n"] == 1


def test_load_failure_wrapped_as_llm_service_error():
    original = OSError("model file not found")

    def load_fn(path):
        raise original

    service = QwenService(load_fn=load_fn)
    try:
        service.load()
        raise AssertionError("expected LLMServiceError")
    except LLMServiceError as e:
        assert e.__cause__ is original


# --- generate() -------------------------------------------------------------

def test_generate_calls_load_apply_chat_template_and_generate_in_order():
    calls = []

    def load_fn(path):
        calls.append("load")
        return _mock_model_and_processor()

    def apply_chat_template_fn(processor, config, messages, num_images):
        calls.append("apply_chat_template")
        assert config == "CONFIG"
        assert num_images == 0
        return "PROMPT"

    def generate_fn(model, processor, prompt, **kwargs):
        calls.append("generate")
        assert prompt == "PROMPT"
        result = MagicMock()
        result.text = "OUTPUT"
        return result

    service = QwenService(
        load_fn=load_fn,
        apply_chat_template_fn=apply_chat_template_fn,
        generate_fn=generate_fn,
    )
    out = service.generate([{"role": "user", "content": "x"}], max_tokens=10, temperature=0)

    assert out == "OUTPUT"
    assert calls == ["load", "apply_chat_template", "generate"]


def test_generate_reuses_already_loaded_model():
    load_calls = {"n": 0}

    def load_fn(path):
        load_calls["n"] += 1
        return _mock_model_and_processor()

    def apply_chat_template_fn(processor, config, messages, num_images):
        return "PROMPT"

    def generate_fn(model, processor, prompt, **kwargs):
        result = MagicMock()
        result.text = "OUTPUT"
        return result

    service = QwenService(
        load_fn=load_fn,
        apply_chat_template_fn=apply_chat_template_fn,
        generate_fn=generate_fn,
    )
    service.load()
    service.generate([{"role": "user", "content": "x"}], max_tokens=10, temperature=0)
    service.generate([{"role": "user", "content": "y"}], max_tokens=10, temperature=0)

    assert load_calls["n"] == 1


def test_generate_passes_logits_processor_through_when_given():
    captured = {}

    def load_fn(path):
        return _mock_model_and_processor()

    def apply_chat_template_fn(processor, config, messages, num_images):
        return "PROMPT"

    def generate_fn(model, processor, prompt, **kwargs):
        captured["logits_processors"] = kwargs.get("logits_processors")
        result = MagicMock()
        result.text = "OUTPUT"
        return result

    service = QwenService(
        load_fn=load_fn,
        apply_chat_template_fn=apply_chat_template_fn,
        generate_fn=generate_fn,
    )
    sentinel = object()
    service.generate(
        [{"role": "user", "content": "x"}],
        max_tokens=10,
        temperature=0,
        logits_processor=sentinel,
    )

    assert captured["logits_processors"] == [sentinel]


def test_generate_without_logits_processor_passes_none():
    captured = {}

    def load_fn(path):
        return _mock_model_and_processor()

    def apply_chat_template_fn(processor, config, messages, num_images):
        return "PROMPT"

    def generate_fn(model, processor, prompt, **kwargs):
        captured["logits_processors"] = kwargs.get("logits_processors")
        result = MagicMock()
        result.text = "OUTPUT"
        return result

    service = QwenService(
        load_fn=load_fn,
        apply_chat_template_fn=apply_chat_template_fn,
        generate_fn=generate_fn,
    )
    service.generate([{"role": "user", "content": "x"}], max_tokens=10, temperature=0)

    assert captured["logits_processors"] is None


def test_generate_handles_plain_string_result_without_text_attribute():
    """generate_fn's return value isn't always a rich result object with
    a .text attribute — some mocks/backends may return a plain string.
    generate() must handle both."""
    def load_fn(path):
        return _mock_model_and_processor()

    def apply_chat_template_fn(processor, config, messages, num_images):
        return "PROMPT"

    def generate_fn(model, processor, prompt, **kwargs):
        return "raw string output"

    service = QwenService(
        load_fn=load_fn,
        apply_chat_template_fn=apply_chat_template_fn,
        generate_fn=generate_fn,
    )
    out = service.generate([{"role": "user", "content": "x"}], max_tokens=10, temperature=0)

    assert out == "raw string output"


def test_generate_failure_wrapped_as_llm_service_error_with_cause_preserved():
    original = RuntimeError("[METAL] Command buffer execution failed: GPU Timeout Error")

    def load_fn(path):
        return _mock_model_and_processor()

    def apply_chat_template_fn(processor, config, messages, num_images):
        return "PROMPT"

    def generate_fn(model, processor, prompt, **kwargs):
        raise original

    service = QwenService(
        load_fn=load_fn,
        apply_chat_template_fn=apply_chat_template_fn,
        generate_fn=generate_fn,
    )
    try:
        service.generate([{"role": "user", "content": "x"}], max_tokens=10, temperature=0)
        raise AssertionError("expected LLMServiceError")
    except LLMServiceError as e:
        assert e.__cause__ is original


def test_generate_propagates_load_failure_as_llm_service_error():
    """generate() calls load() internally — a load failure must surface
    as the same LLMServiceError type, not a raw exception from load_fn."""
    def load_fn(path):
        raise OSError("disk full")

    service = QwenService(load_fn=load_fn)
    try:
        service.generate([{"role": "user", "content": "x"}], max_tokens=10, temperature=0)
        raise AssertionError("expected LLMServiceError")
    except LLMServiceError:
        pass


# --- get_service() singleton -----------------------------------------------

def test_get_service_returns_same_instance_across_calls():
    a = get_service()
    b = get_service()
    assert a is b


if __name__ == "__main__":
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items())
              if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {name}: {e}")
            failed += 1
        except Exception:
            print(f"ERROR: {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
