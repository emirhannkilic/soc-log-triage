"""
Shared Qwen3.5-9B model service (PHISHING_ROUTING_PLAN.md hybrid workflow
follow-up — "ortak Qwen model servisi").

WHY THIS EXISTS
    Before this module, src/semantic/analyze.py owned its own
    load()/generate() calls and its own module-level `_model`/`_processor`
    singleton directly. The plan's section 10.5 ("tek model, iki çağrı")
    already anticipated a second caller — a future final-report generator
    — needing the SAME loaded model inside the SAME phishing request,
    not a second 9B load. This module is that shared seam: one lazy-load,
    one process-lifetime instance, one generate() entry point, usable by
    any caller that needs Qwen3.5-9B without re-implementing mlx_vlm
    plumbing.

WHAT THIS DOES NOT DO (deliberately, this turn)
    - No teacher/ script refactor. src/teacher/generate.py,
      generate_training_data.py, and constrained_test.py keep their own
      load()/generate() calls untouched — they are separate-process,
      one-off training-data tools, not part of a live phishing request,
      and CLAUDE.md's guidance against touching finished training
      pipelines applies. Using the same model checkpoint does not mean
      sharing a lifecycle.
    - No final-report prompt/generation. This service is model plumbing
      only; report generation is a separate, not-yet-built task.
    - No JSON-schema/logits-processor logic. Building a constrained
      decoding schema is caller-specific (src/semantic/analyze.py's
      SemanticFindingCandidate array schema today, a Report schema for
      the future report generator) — this module accepts an already-built
      logits_processor, it does not construct one.
    - No CLI/web wiring, no real-model smoke test.

LAZY LOAD, ONE INSTANCE PER PROCESS
    QwenService.generate() (and .load(), if a caller wants to force
    loading early) both go through _ensure_loaded(), which only calls
    mlx_vlm.load() once — same reasoning src/semantic/analyze.py's
    previous load_model() already had (src/web.py's _load_model for
    Seneca does the same thing for a different model): a 4-bit 9B model
    reload costs minutes, so it must happen at most once per process.

    get_service() is a module-level lazy singleton on top of that — the
    default way callers obtain a QwenService without constructing one
    themselves. A caller that wants a distinct instance (tests, or a
    future caller needing an isolated lifecycle) can still construct
    QwenService() directly; nothing about the class requires the module
    singleton.

TESTABILITY — NO MODEL LOAD REQUIRED
    QwenService.generate() calls self._load_fn / self._generate_fn /
    self._apply_chat_template_fn — constructor-injectable, defaulting to
    the real mlx_vlm functions (imported lazily inside _ensure_loaded()/
    generate(), not at module import time, so importing this module never
    touches mlx_vlm or the filesystem). A test can construct
    QwenService(load_fn=..., generate_fn=..., apply_chat_template_fn=...)
    with plain MagicMocks and never load a real model — the same
    mocking shape tests/test_semantic_analyze.py already used against
    src.semantic.analyze.load_model, just moved one layer down.

ERRORS
    LLMServiceError wraps any failure from the load/generate
    infrastructure itself (model load failure, mlx_vlm/Metal runtime
    errors such as the GPU timeouts PROGRESS.md documents) — always with
    `raise LLMServiceError(...) from exc` so the original traceback is
    never discarded. This module has no opinion on what a malformed
    JSON *output* means (that is caller-specific — see
    src/semantic/analyze.py's SemanticExtractionError, which wraps
    LLMServiceError as one of several failure codes it normalizes to).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3.5-9B-MLX-4bit"


class LLMServiceError(Exception):
    """Model yükleme veya generate altyapısı başarısız."""


class QwenService:
    """One lazily-loaded Qwen3.5-9B instance (mlx_vlm, NOT mlx_lm — same
    constraint every other Qwen3.5-9B caller in this repo already
    documents: src/teacher/generate.py, src/semantic/analyze.py). Safe to
    construct multiple times in tests; in production code, use
    get_service() so the whole process shares one instance."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        load_fn=None,
        generate_fn=None,
        apply_chat_template_fn=None,
    ):
        self._model_path = model_path
        # Real mlx_vlm callables are the default, imported lazily (only
        # when actually needed) so constructing/importing this module
        # never requires mlx_vlm to be installed or touches the
        # filesystem — tests inject plain mocks instead.
        self._load_fn = load_fn
        self._generate_fn = generate_fn
        self._apply_chat_template_fn = apply_chat_template_fn
        self._model = None
        self._processor = None

    def _resolve_load_fn(self):
        if self._load_fn is not None:
            return self._load_fn
        from mlx_vlm import load

        return load

    def _resolve_generate_fn(self):
        if self._generate_fn is not None:
            return self._generate_fn
        from mlx_vlm import generate

        return generate

    def _resolve_apply_chat_template_fn(self):
        if self._apply_chat_template_fn is not None:
            return self._apply_chat_template_fn
        from mlx_vlm.prompt_utils import apply_chat_template

        return apply_chat_template

    def load(self):
        """Loads the model if it hasn't been loaded yet in this instance,
        and returns (model, processor). Idempotent — safe to call more
        than once, only the first call actually loads."""
        if self._model is None:
            load_fn = self._resolve_load_fn()
            try:
                self._model, self._processor = load_fn(str(self._model_path))
            except Exception as exc:
                raise LLMServiceError(
                    f"Qwen3.5-9B yüklenemedi ({self._model_path}): {exc}"
                ) from exc
        return self._model, self._processor

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        logits_processor=None,
    ) -> str:
        """Runs one chat-template + generate call and returns the raw
        text output. No JSON parsing, no schema validation — that is the
        caller's job (e.g. src/semantic/analyze.py's
        extract_json_array + validate_raw_findings)."""
        model, processor = self.load()

        try:
            apply_chat_template_fn = self._resolve_apply_chat_template_fn()
            prompt = apply_chat_template_fn(processor, model.config, messages, num_images=0)

            generate_fn = self._resolve_generate_fn()
            logits_processors = [logits_processor] if logits_processor is not None else None
            result = generate_fn(
                model,
                processor,
                prompt,
                image=None,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
                logits_processors=logits_processors,
            )
        except Exception as exc:
            raise LLMServiceError(f"Qwen3.5-9B generate() başarısız: {exc}") from exc

        return result.text if hasattr(result, "text") else str(result)


_service: QwenService | None = None


def get_service() -> QwenService:
    """Process-lifetime lazy singleton — the default way production code
    (src/semantic/analyze.py today, a future report generator) obtains a
    QwenService. Tests should construct QwenService(...) directly with
    injected mocks instead of going through this singleton, so one
    test's mock never leaks into another test via shared module state."""
    global _service
    if _service is None:
        _service = QwenService()
    return _service
