from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from threading import Lock
from urllib.request import urlopen
from zipfile import ZipFile

from .token_usage import MESSAGE_OVERHEAD_TOKENS, REPLY_PRIMING_TOKENS


TOKENIZER_URL = "https://cdn.deepseek.com/api-docs/deepseek_v4_tokenizer.zip"
DEFAULT_TOKENIZER_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "deepseek-tokenizer.json"
)


class TokenizerSetupError(RuntimeError):
    pass


def download_official_tokenizer(destination: Path = DEFAULT_TOKENIZER_PATH) -> Path:
    """Download the pinned official DeepSeek tokenizer asset into local cache."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(TOKENIZER_URL, timeout=60) as response:
            archive = response.read()
        with ZipFile(BytesIO(archive)) as bundle:
            tokenizer_member = next(
                name for name in bundle.namelist() if name.endswith("/tokenizer.json")
            )
            payload = bundle.read(tokenizer_member)
    except Exception as error:
        raise TokenizerSetupError(
            "Не удалось скачать официальный токенизатор DeepSeek"
        ) from error

    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return destination


class DeepSeekTokenCounter:
    """Exact text counter backed by DeepSeek's published tokenizer.json.

    Message envelope tokens remain an estimate because the API owns the final
    chat template. The provider-reported usage is the source of truth.
    """

    def __init__(self, tokenizer_path: Path = DEFAULT_TOKENIZER_PATH) -> None:
        self._path = tokenizer_path
        self._tokenizer = None
        self._lock = Lock()

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        with self._lock:
            if self._tokenizer is not None:
                return self._tokenizer
            if not self._path.exists():
                download_official_tokenizer(self._path)
            try:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_file(str(self._path))
            except Exception as error:
                raise TokenizerSetupError(
                    "Официальный токенизатор DeepSeek не удалось загрузить"
                ) from error
            return self._tokenizer

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._get_tokenizer().encode(text).ids)

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        if not messages:
            return 0
        return (
            sum(
                self.count_text(message.get("content", ""))
                + MESSAGE_OVERHEAD_TOKENS
                for message in messages
            )
            + REPLY_PRIMING_TOKENS
        )
