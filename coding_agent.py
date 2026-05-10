import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_PATH = str(BASE_DIR / "gemma_lora.weights.h5")
DEFAULT_PRESET = str(BASE_DIR / "gemma-keras-gemma_1.1_instruct_2b_en-v4")
DEFAULT_LORA_RANK = 16
DEFAULT_MAX_LENGTH = 256
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TASK = "List the files in the current directory and suggest one coding task."


def load_env_with_fallback() -> None:
    """Load .env with encoding fallback for Windows PowerShell-created files."""
    try:
        load_dotenv(encoding="utf-8")
        return
    except UnicodeDecodeError:
        # Windows PowerShell redirection often creates UTF-16LE text files.
        load_dotenv(encoding="utf-16")


def load_gemma_with_lora(
    weights_path: str,
    preset: str = DEFAULT_PRESET,
    lora_rank: int = 16,
):
    """Load a Gemma model preset and apply local LoRA fine-tuned weights."""
    import keras_nlp  # type: ignore[import-not-found]

    preset_path = Path(preset)
    resolved_preset = str(preset_path.resolve()) if preset_path.exists() else preset

    try:
        model = keras_nlp.models.GemmaCausalLM.from_preset(resolved_preset)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to load Gemma preset. Use a valid local preset path or configure "
            "Kaggle credentials/consent for Gemma model access. "
            f"preset={resolved_preset}"
        ) from exc
    model.backbone.enable_lora(rank=lora_rank)
    model.load_lora_weights(weights_path)
    return model


class GemmaLangChainChatModel(BaseChatModel):
    """LangChain chat model wrapper around KerasNLP GemmaCausalLM."""

    model: Any
    max_length: int = 256
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "gemma_keras_nlp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "max_length": self.max_length,
            "temperature": self.temperature,
        }

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        lines: List[str] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            lines.append(f"{role}: {message.content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = run_manager, kwargs
        prompt = self._messages_to_prompt(messages)

        output = self.model.generate(
            prompt,
            max_length=self.max_length,
            temperature=self.temperature,
        )
        text = output if isinstance(output, str) else str(output)

        if stop:
            for token in stop:
                idx = text.find(token)
                if idx != -1:
                    text = text[:idx]
                    break

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class RemoteGemmaChatModel(BaseChatModel):
    """Chat model wrapper that forwards prompts to a remote Gemma HTTP endpoint."""

    endpoint: str
    max_length: int = 256
    temperature: float = 0.2

    def __init__(self, endpoint: str, max_length: int = 256, temperature: float = 0.2):
        self.endpoint = endpoint
        self.max_length = max_length
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "gemma_remote_http"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"endpoint": self.endpoint, "max_length": self.max_length, "temperature": self.temperature}

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        lines: List[str] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            lines.append(f"{role}: {message.content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def bind_tools(self, tools: List[Any], tool_choice: Optional[str] = None, **kwargs: Any) -> "RemoteGemmaChatModel":
        # Minimal bind implementation so LangChain agent can attach tools.
        self._bound_tools = tools
        self._tool_choice = tool_choice
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = run_manager, kwargs
        prompt = self._messages_to_prompt(messages)

        payload = {"prompt": prompt, "max_length": self.max_length}
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("text", "")
        except Exception as exc:  # noqa: BLE001
            text = f"<remote request failed: {exc}>"

        if stop:
            for token in stop:
                idx = text.find(token)
                if idx != -1:
                    text = text[:idx]
                    break

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file from disk by path."""
    p = Path(path)
    if not p.exists():
        return f"File not found: {p}"
    return p.read_text(encoding="utf-8")


@tool
def write_file(spec: str) -> str:
    """
    Input format:
    <absolute-or-relative-path>\n\n<file-content>
    """
    if "\n\n" not in spec:
        return "Invalid input. Expected: <path>\\n\\n<content>"

    raw_path, content = spec.split("\n\n", 1)
    p = Path(raw_path.strip())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote file: {p}"


@tool
def list_directory(path: str = ".") -> str:
    """List entries inside a directory path."""
    p = Path(path)
    if not p.exists():
        return f"Directory not found: {p}"
    if not p.is_dir():
        return f"Not a directory: {p}"
    entries = sorted(child.name for child in p.iterdir())
    return "\n".join(entries) if entries else "<empty>"


def build_coding_agent(chat_model: BaseChatModel):
    return create_agent(
        model=chat_model,
        tools=[read_file, write_file, list_directory],
        system_prompt=(
            "You are a coding assistant. Prefer using tools when user asks about project files. "
            "Return concise, actionable responses."
        ),
        debug=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain coding agent using a remote Gemma HTTP endpoint")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Remote Gemma HTTP /generate endpoint (e.g. https://.../generate). Can also be provided via GEMMA_ENDPOINT in .env.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Max generated length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Task prompt for the coding agent",
    )
    args = parser.parse_args()

    # Load .env (if present) and environment variables
    load_env_with_fallback()
    endpoint = args.endpoint or os.environ.get("GEMMA_ENDPOINT")
    if not endpoint:
        raise SystemExit("A remote endpoint must be provided via --endpoint or GEMMA_ENDPOINT in .env")

    chat_model = RemoteGemmaChatModel(endpoint=endpoint, max_length=args.max_length, temperature=args.temperature)
    agent = build_coding_agent(chat_model)

    result = agent.invoke({"messages": [{"role": "user", "content": args.task}]})
    print("\n=== Agent Output ===")
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)
    else:
        print(result)


if __name__ == "__main__":
    main()
