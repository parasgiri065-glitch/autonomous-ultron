"""Telegram cockpit and approval bridge.

This module intentionally contains no framework or bot-library dependency. The
Bot API is small JSON-over-HTTPS, so a narrow httpx wrapper keeps the interface
easy to mock and makes authorization happen before any command or goal logic.
"""

from __future__ import annotations

import json
import logging
import shlex
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx

from ..agent import Agent, AgentResult
from ..cache import Cache
from ..config import Settings, get_settings
from ..domains.intel import IntelResearchEngine
from ..extractor import ExtractionError, GroundedDataExtractor
from ..ledger import FailureLedger
from ..policy import ApprovalPrompt, PolicyGate, Prompter
from ..registry import Registry
from ..scavenger import Scavenger

LOG = logging.getLogger(__name__)


class TelegramAPIError(RuntimeError):
    """Telegram returned an error or the HTTP request failed."""


Transport = Callable[[str, str, dict[str, Any]], dict[str, Any]]


class TelegramBotClient:
    """Minimal Bot API client with injectable transport and fail-closed auth."""

    def __init__(
        self,
        token: str | None = None,
        allowed_user_ids: Iterable[int] | None = None,
        *,
        settings: Settings | None = None,
        transport: Transport | None = None,
        timeout: float = 15.0,
        api_base: str = "https://api.telegram.org",
    ) -> None:
        self.settings = settings or get_settings()
        self.token = token if token is not None else self.settings.telegram_bot_token
        configured = (
            list(allowed_user_ids)
            if allowed_user_ids is not None
            else list(self.settings.telegram_allowed_user_ids)
        )
        self.allowed_user_ids = frozenset(int(value) for value in configured)
        self.timeout = timeout
        self.api_base = api_base.rstrip("/")
        self.transport = transport
        self.last_update_offset: int | None = None

    @property
    def configured(self) -> bool:
        return bool(self.token) and bool(self.allowed_user_ids)

    def authorized_user(self, user_id: Any) -> bool:
        try:
            authorized = int(user_id) in self.allowed_user_ids
        except (TypeError, ValueError):
            authorized = False
        if not authorized:
            LOG.warning("ignoring Telegram update from unauthorized user id=%r", user_id)
        return authorized

    def authorized_update(self, update: dict[str, Any]) -> bool:
        user_id = _update_user_id(update)
        return self.authorized_user(user_id)

    def filter_updates(self, updates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop unauthorized updates before command, goal, or callback handling."""
        return [update for update in updates if self.authorized_update(update)]

    def get_updates(self, *, offset: int | None = None, timeout: int = 20) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": max(0, int(timeout))}
        if offset is not None:
            payload["offset"] = offset
        result = self._call("getUpdates", payload)
        updates = result.get("result", [])
        if not isinstance(updates, list):
            return []
        valid_updates = [item for item in updates if isinstance(item, dict)]
        update_ids = [int(item["update_id"]) for item in valid_updates if "update_id" in item]
        if update_ids:
            # Advance past unauthorized updates too, otherwise one hostile
            # update would be returned forever by long polling.
            self.last_update_offset = max(update_ids) + 1
        return self.filter_updates(valid_updates)

    def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
        parse_mode: str = "Markdown",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return self._call("sendMessage", payload).get("result", {})

    def answer_callback_query(self, callback_query_id: str, *, text: str = "") -> dict[str, Any]:
        return self._call(
            "answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text}
        ).get("result", {})

    def send_document(
        self, chat_id: int | str, document: str | Path, *, caption: str | None = None
    ) -> dict[str, Any]:
        path = Path(document)
        if not path.is_file():
            raise TelegramAPIError(f"artifact does not exist: {path}")
        payload: dict[str, Any] = {"chat_id": chat_id}
        if caption:
            payload["caption"] = caption
        files = {"document": (path.name, path.read_bytes(), "application/octet-stream")}
        return self._call("sendDocument", payload, files=files).get("result", {})

    def _call(
        self,
        method: str,
        payload: dict[str, Any],
        *,
        files: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.token:
            raise TelegramAPIError("TELEGRAM_BOT_TOKEN is not configured")
        url = f"{self.api_base}/bot{self.token}/{method}"
        kwargs: dict[str, Any] = {"json": payload}
        if files is not None:
            kwargs = {"data": payload, "files": files}
        if self.transport is not None:
            response = self.transport(method, url, kwargs)
        else:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    raw = client.post(url, **kwargs)
                    raw.raise_for_status()
                    response = raw.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise TelegramAPIError(f"Telegram {method} request failed: {exc}") from exc
        if not isinstance(response, dict) or not response.get("ok", False):
            description = (
                response.get("description", "unknown Telegram error")
                if isinstance(response, dict)
                else response
            )
            raise TelegramAPIError(f"Telegram {method} failed: {description}")
        return response


class TelegramPrompter:
    """Policy ``Prompter`` that resolves one approval through an inline button."""

    def __init__(
        self,
        client: TelegramBotClient,
        chat_id: int | str,
        *,
        timeout_s: float = 120.0,
        poll_timeout_s: int = 5,
    ) -> None:
        self.client = client
        self.chat_id = chat_id
        self.timeout_s = max(0.0, timeout_s)
        self.poll_timeout_s = max(0, poll_timeout_s)
        self.prompts: list[ApprovalPrompt] = []

    def __call__(self, prompt: ApprovalPrompt) -> bool:
        self.prompts.append(prompt)
        token = uuid.uuid4().hex[:16]
        markup = {
            "inline_keyboard": [
                [
                    {"text": "✅ Approve", "callback_data": f"ultron:approve:{token}"},
                    {"text": "❌ Deny", "callback_data": f"ultron:deny:{token}"},
                ]
            ]
        }
        self.client.send_message(self.chat_id, self.format_prompt(prompt), reply_markup=markup)
        deadline = time.monotonic() + self.timeout_s
        offset: int | None = None
        while time.monotonic() <= deadline:
            remaining = max(0.0, deadline - time.monotonic())
            updates = self.client.get_updates(
                offset=offset, timeout=min(self.poll_timeout_s, int(remaining))
            )
            if self.client.last_update_offset is not None:
                offset = max(offset or 0, self.client.last_update_offset)
            for update in updates:
                if "update_id" in update:
                    offset = int(update["update_id"]) + 1
                callback = update.get("callback_query")
                if not isinstance(callback, dict):
                    continue
                data = str(callback.get("data", ""))
                if data not in {f"ultron:approve:{token}", f"ultron:deny:{token}"}:
                    continue
                callback_message = callback.get("message") or {}
                callback_chat = (
                    callback_message.get("chat", {}).get("id")
                    if isinstance(callback_message, dict)
                    else None
                )
                if callback_chat is not None and str(callback_chat) != str(self.chat_id):
                    continue
                callback_id = callback.get("id")
                if callback_id:
                    self.client.answer_callback_query(str(callback_id), text="Recorded")
                return data.startswith("ultron:approve:")
        LOG.warning("Telegram approval timed out for %s", prompt.tool)
        return False

    @staticmethod
    def format_prompt(prompt: ApprovalPrompt) -> str:
        preview = json.dumps(prompt.inputs_preview, sort_keys=True, default=str)
        if len(preview) > 800:
            preview = preview[:800] + "…"
        return (
            "*Ultron approval required*\n\n"
            f"*Tool:* `{prompt.tool}@{prompt.version}`\n"
            f"*Risk:* `{prompt.risk}`\n"
            f"*Reason:* {prompt.reason}\n"
            f"*Network:* `{prompt.network}`\n"
            f"*Goal:* {prompt.goal[:200]}\n"
            f"*Inputs:* `{preview}`\n"
            f"*Run:* `{prompt.run_id}`"
        )


class TelegramCockpit:
    """Authorized Telegram command/goal loop around the existing Agent."""

    def __init__(
        self,
        client: TelegramBotClient,
        *,
        settings: Settings | None = None,
        agent_factory: Callable[[int | str], Agent] | None = None,
        intel_factory: Callable[[], IntelResearchEngine] | None = None,
        extractor_factory: Callable[[int | str], GroundedDataExtractor] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client
        self.registry = Registry(self.settings).load()
        self.ledger = FailureLedger(self.settings.state_dir)
        self.cache = Cache(self.settings)
        self._agent_factory = agent_factory
        self._intel_factory = intel_factory
        self._extractor_factory = extractor_factory

    def handle_update(self, update: dict[str, Any]) -> AgentResult | dict[str, Any] | None:
        if not self.client.authorized_update(update):
            return None
        message = update.get("message")
        if not isinstance(message, dict):
            return None
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        text = message.get("text")
        if chat_id is None or not isinstance(text, str):
            return None
        text = text.strip()
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            command = parts[0].split("@", 1)[0].lower()
            if command == "/intel":
                topic = parts[1].strip() if len(parts) == 2 else ""
                return self.handle_intel(topic, chat_id)
            if command == "/extract":
                return self.handle_extract(parts[1] if len(parts) == 2 else "", chat_id)
            return self.handle_command(command, chat_id)
        return self.handle_goal(text, chat_id)

    def handle_command(self, command: str, chat_id: int | str) -> dict[str, Any]:
        if command in {"/start", "/help"}:
            payload = {
                "command": command,
                "text": (
                    "*Ultron cockpit*\n\n"
                    f"*Status:* online · {len(self.registry)} active tool(s)\n"
                    "Send a goal to run the agent. Commands: `/intel <topic>`, `/extract <URL-or-path> --fields '<JSON>' --format json|csv|md`, `/gaps`, `/scavenge`, `/status`."
                ),
            }
        elif command == "/intel":
            payload = {"command": command, "text": "Usage: `/intel <topic>`"}
        elif command == "/gaps":
            gaps = self.ledger.read()
            lines = [f"*Capability gaps:* {len(gaps)}"]
            lines.extend(f"• `{gap.frequency}` — {gap.goal[:160]}" for gap in gaps[:20])
            payload = {"command": command, "text": "\n".join(lines)}
        elif command == "/status":
            stats = self.cache.stats().as_dict()
            payload = {
                "command": command,
                "text": (
                    f"*Tools:* {len(self.registry)}\n"
                    f"*Capability graph nodes:* {len(self.registry.graph())}\n"
                    f"*Cache entries:* {stats['entries']}\n"
                    f"*Cache hit rate:* {stats['hit_rate']:.1%}"
                ),
                "data": {
                    "tools": self.registry.names(),
                    "graph": self.registry.graph(),
                    "cache": stats,
                },
            }
        elif command == "/scavenge":
            if not self.settings.scavenge_enabled:
                payload = {
                    "command": command,
                    "text": "Scavenging is disabled. Set ULTRON_SCAVENGE=1 to enable it.",
                }
            else:
                scavenger = Scavenger(self.settings, max_candidates=20)
                candidates = scavenger.discover()
                report = scavenger.forge_candidates(candidates)
                payload = {
                    "command": command,
                    "text": f"Scavenger found {len(candidates)} candidate(s); forged {len(report['forged'])}.",
                    "data": {"candidates": candidates, "report": report},
                }
        else:
            payload = {"command": command, "text": "Unknown command. Try `/help`."}
        self.client.send_message(chat_id, payload["text"])
        return payload

    def handle_extract(self, arguments: str, chat_id: int | str) -> dict[str, Any]:
        """Run one explicitly-schema'd extraction and deliver its artifact."""
        try:
            tokens = shlex.split(arguments)
            if not tokens:
                raise ExtractionError("usage: /extract <URL-or-path> --fields '{\"name\":\"string\"}' --format json|csv|md")
            source = tokens[0]
            fields_arg: str | None = None
            output_format = "json"
            index = 1
            while index < len(tokens):
                token = tokens[index]
                if token in {"--fields", "-f"}:
                    index += 1
                    if index >= len(tokens):
                        raise ExtractionError("--fields requires a JSON object")
                    fields_arg = tokens[index]
                elif token.startswith("--fields="):
                    fields_arg = token.split("=", 1)[1]
                elif token == "--format":
                    index += 1
                    if index >= len(tokens):
                        raise ExtractionError("--format requires json, csv, or md")
                    output_format = tokens[index]
                elif token.startswith("--format="):
                    output_format = token.split("=", 1)[1]
                else:
                    raise ExtractionError(f"unknown /extract option: {token}")
                index += 1
            if fields_arg is None:
                raise ExtractionError("--fields is required")
            fields = json.loads(fields_arg)
            extractor = (
                self._extractor_factory(chat_id)
                if self._extractor_factory is not None
                else GroundedDataExtractor(
                    self.settings,
                    policy_gate=PolicyGate(
                        self.settings,
                        prompter=TelegramPrompter(self.client, chat_id),
                        run_id=f"telegram-extract-{chat_id}",
                    ),
                    interactive=True,
                )
            )
            result = extractor.extract(source, fields, output_format=output_format)
            rendered = result.render(output_format)
            self.client.send_message(chat_id, rendered[:3800])
            if result.ok:
                artifact = result.write_artifact(
                    self.settings.state_dir / "extract" / f"extract-{int(time.time())}.{output_format}",
                    output_format,
                )
                self.client.send_document(chat_id, artifact, caption="Grounded extraction artifact")
            return {"command": "/extract", "source": source, "result": result.as_dict()}
        except (ExtractionError, json.JSONDecodeError, ValueError) as exc:
            payload = {"command": "/extract", "error": str(exc)}
            self.client.send_message(chat_id, f"Extraction refused: {exc}")
            return payload

    def handle_intel(self, topic: str, chat_id: int | str) -> dict[str, Any]:
        if not topic.strip():
            return self.handle_command("/intel", chat_id)
        self.client.send_message(chat_id, f"Researching public sources for *{topic}*…")
        self.client.send_message(chat_id, "Ingesting RSS, search results, and article text…")
        engine = (
            self._intel_factory()
            if self._intel_factory is not None
            else IntelResearchEngine(self.settings)
        )
        brief = engine.research(
            topic, depth="deep", output=self.settings.state_dir / "intel" / "brief.md"
        )
        self.client.send_message(
            chat_id,
            f"Triangulating claims across {len(brief.sources)} source(s) and {len(brief.verified_facts)} verified fact(s)…",
        )
        self.client.send_message(chat_id, brief.markdown[:3800])
        if brief.markdown_path is not None:
            self.client.send_document(chat_id, brief.markdown_path, caption=f"Intel brief: {topic}")
        return {
            "command": "/intel",
            "topic": topic,
            "sources": len(brief.sources),
            "verified_claims": len(brief.verified_facts),
            "markdown": brief.markdown,
            "path": str(brief.markdown_path) if brief.markdown_path else None,
        }

    def handle_goal(self, goal: str, chat_id: int | str) -> AgentResult:
        self.client.send_message(chat_id, "Planning…")
        agent = self._make_agent(chat_id)
        result = agent.run(goal)
        for step in result.steps:
            self.client.send_message(chat_id, f"Running tool: {step.tool}…")
            self.client.send_message(chat_id, "Verifying with Breaker…")
        final_text = result.answer or "No grounded response was produced."
        self.client.send_message(chat_id, final_text)
        for artifact in self._artifacts(result):
            self.client.send_document(chat_id, artifact)
        return result

    def run_polling(self, *, timeout: int = 20, max_cycles: int | None = None) -> None:
        offset: int | None = None
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            cycles += 1
            updates = self.client.get_updates(offset=offset, timeout=timeout)
            if self.client.last_update_offset is not None:
                offset = max(offset or 0, self.client.last_update_offset)
            for update in updates:
                if "update_id" in update:
                    offset = int(update["update_id"]) + 1
                self.handle_update(update)

    def _make_agent(self, chat_id: int | str) -> Agent:
        if self._agent_factory is not None:
            return self._agent_factory(chat_id)
        prompter: Prompter = TelegramPrompter(self.client, chat_id)
        gate = PolicyGate(self.settings, prompter=prompter, run_id=f"telegram-{chat_id}")
        return Agent(settings=self.settings, gate=gate, interactive=True)

    def _artifacts(self, result: AgentResult) -> list[Path]:
        roots = [self.settings.repo_root.resolve(), self.settings.state_dir.resolve()]
        found: list[Path] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
            elif isinstance(value, str):
                path = Path(value).expanduser()
                if path.suffix.lower() not in {".md", ".csv", ".json"}:
                    return
                try:
                    resolved = path.resolve()
                    if (
                        resolved.is_file()
                        and any(_within(resolved, root) for root in roots)
                        and resolved not in found
                    ):
                        found.append(resolved)
                except OSError:
                    return

        for step in result.steps:
            visit(step.result)
        return found


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _update_user_id(update: dict[str, Any]) -> Any:
    callback = update.get("callback_query")
    if isinstance(callback, dict):
        sender = callback.get("from") or {}
        if isinstance(sender, dict):
            return sender.get("id")
    message = update.get("message") or update.get("edited_message") or {}
    sender = message.get("from") if isinstance(message, dict) else {}
    return sender.get("id") if isinstance(sender, dict) else None
