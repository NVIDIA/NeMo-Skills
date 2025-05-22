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

from __future__ import annotations

import html
import logging
import re
from typing import List

import gradio as gr

from nemo_skills.inference.chat_interface.core import CodeExecStatus, ModelLoader
from nemo_skills.inference.chat_interface.chat_service import ChatService, AppContext

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Styling helpers
# ------------------------------------------------------------------
_CUSTOM_CSS = """
#input-row {align-items: center;}
#send-btn, #cancel-btn {
  width: 48px !important;
  height: 48px !important;
  min-width: 48px !important;
  padding: 0 !important;
  font-size: 24px !important;
  line-height: 24px !important;
}
#msg-box textarea {
  resize: vertical;
}
"""


def _format_output(text: str) -> str:
    """Format special tool_call blocks as code for nicer display."""
    if not text:
        return ""

    parts = re.split(r"(<tool_call>.*?</tool_call>|```output.*?```)", text, flags=re.DOTALL)
    processed: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("<tool_call>") and part.endswith("</tool_call>"):
            processed.append(f"```python{part[len('<tool_call>'):-len('</tool_call>')]}```")
        elif part.startswith("```output") and part.endswith("```"):
            processed.append(part)
        else:
            processed.append(html.escape(part))
    return "".join(processed)


class ChatUI:
    """Gradio front-end wired to an `AppContext`."""

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.latest_code_status = CodeExecStatus.NOT_REQUESTED
        # Flag toggled by the "Cancel generation" button to interrupt streaming.
        self.cancel_requested: bool = False
        self.turns_for_prompt: list[dict] = []

        with gr.Blocks(title="AI Chat Interface", theme=gr.themes.Soft(), css=_CUSTOM_CSS) as self.demo:
            # chat panel needs to exist before server panel (server panel references it in outputs list)
            self._build_chat_panel()

        self.chat_group.visible = True
        # Toggle is available only when the backend supports *both* modes.  For single-
        # mode models ("cot" or "tir") we force the checkbox to the correct state and
        # disable user interaction.
        mode_cap = ctx.cfg.supported_modes
        can_toggle = mode_cap == "both"
        self.code_exec_checkbox.interactive = can_toggle
        if not can_toggle:
            # Force checkbox value based on capabilities.
            self.code_exec_checkbox.value = (mode_cap == "tir")
            note = (
                "Model only supports code execution mode." if mode_cap == "tir" else "Model does not support code execution mode."
            )
            self.subtitle_md.value = f"Status: {note}"

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------
    def _build_chat_panel(self):
        with gr.Column(visible=False) as self.chat_group:
            gr.Markdown("## Chat")
            self.subtitle_md = gr.Markdown("")
            # Sandbox banner (hidden by default)
            self.banner_html = gr.HTML(value="", visible=False)
            with gr.Row():
                self.max_tokens = gr.Slider(50, 20000, value=4000, step=50, label="Max new tokens")
                self.temperature = gr.Slider(0.0, 1.2, value=0.0, step=0.05, label="Temperature")
            self.code_exec_checkbox = gr.Checkbox(label="Enable code execution", value=self.ctx.cfg.initial_code_execution_state)
            self.chatbot = gr.Chatbot(height=450, show_label=False, bubble_full_width=False)
            with gr.Row(elem_id="input-row", equal_height=True):
                self.msg_tb = gr.Textbox(show_label=False, placeholder="Type your message...", lines=3, scale=8, elem_id="msg-box")
                # Send (arrow) and Cancel (square) buttons mimic a typical chat UI.
                self.send_btn = gr.Button("➤", variant="primary", elem_id="send-btn", scale=0)
                # Square stop icon (■). Hidden until a generation is in progress.
                self.cancel_btn = gr.Button("■", variant="secondary", visible=False, interactive=False, elem_id="cancel-btn", scale=0)
            self.clear_btn = gr.Button("Clear chat")

            # Bind events
            self.code_exec_checkbox.change(
                self.on_toggle_code_exec,
                inputs=[self.code_exec_checkbox],
                outputs=[self.subtitle_md, self.code_exec_checkbox, self.banner_html],
            )
            self.send_btn.click(
                self.handle_chat_submit,
                inputs=[self.msg_tb, self.max_tokens, self.temperature],
                outputs=[self.chatbot, self.banner_html, self.send_btn, self.cancel_btn],
            ).then(lambda: gr.update(value=""), None, [self.msg_tb])
            self.msg_tb.submit(
                self.handle_chat_submit,
                inputs=[self.msg_tb, self.max_tokens, self.temperature],
                outputs=[self.chatbot, self.banner_html, self.send_btn, self.cancel_btn],
            ).then(lambda: gr.update(value=""), None, [self.msg_tb])
            self.clear_btn.click(self.on_clear_chat, None, [self.chatbot, self.msg_tb])
            # Bind cancel event – returns updates for both send and cancel buttons.
            self.cancel_btn.click(self.on_cancel, None, [self.send_btn, self.cancel_btn])

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------
    def on_toggle_code_exec(self, checkbox_val: bool):
        # Respect declared capabilities – disallow switching if model is single-mode.
        mode_cap = self.ctx.cfg.supported_modes
        if mode_cap != "both":
            # Revert to the allowed state and show explanatory message.
            allowed_val = (mode_cap == "tir")
            msg = (
                "Model only supports code execution mode." if mode_cap == "tir" else "Model does not support code execution mode."
            )
            self.latest_code_status = self.ctx.loader.get_code_execution_status(allowed_val)
            return (
                f"{msg}",
                gr.update(value=allowed_val),
                self._banner_from_code_status(self.latest_code_status),
            )

        # If we reach here, the toggle is allowed.
        self.latest_code_status = self.ctx.loader.get_code_execution_status(checkbox_val)
        status_txt = {
            CodeExecStatus.ENABLED: "Python interpreter ENABLED.",
            CodeExecStatus.NOT_REQUESTED: "Python interpreter DISABLED.",
            CodeExecStatus.DISABLED: "Interpreter unavailable.",
        }[self.latest_code_status]
        # If sandbox/model unavailable, force checkbox back to OFF; otherwise leave unchanged
        checkbox_update = (
            gr.update(value=False)
            if self.latest_code_status == CodeExecStatus.DISABLED
            else gr.update()
        )

        return (
            f"Status: {status_txt}",
            checkbox_update,
            self._banner_from_code_status(self.latest_code_status),
        )

    def on_cancel(self):
        """Callback for the *Cancel generation* button.

        Sets an instance flag that the streaming loop checks to stop early and
        immediately disables the button in the UI.
        """
        # Request cancellation – the running generator will notice soon.
        self.cancel_requested = True
        # Immediately swap buttons in UI.
        return (
            gr.update(visible=True, interactive=True),   # send button visible again
            gr.update(visible=False, interactive=False),  # cancel button hidden
        )
    
    def on_clear_chat(self):
        """Clear chat history both in UI and internal state."""
        self.turns_for_prompt = []
        return None, ""

    def handle_chat_submit(self, user_msg: str, max_tokens: int, temperature: float):
        if not user_msg.strip():
            yield (
                [(user_msg, "Please enter a question.")],
                gr.update(value="", visible=False),
                gr.update(visible=True, interactive=True),
                gr.update(visible=False, interactive=False),
            )
            return

        # If user just toggled, refresh code_status.
        self.latest_code_status = self.ctx.loader.get_code_execution_status(self.code_exec_checkbox.value)
        if self.latest_code_status == CodeExecStatus.ENABLED and not self.ctx.loader.code_llm:
            self.latest_code_status = self.ctx.loader.get_code_execution_status(True)

        # Reset cancellation flag and enable the button for this generation.
        self.cancel_requested = False

        if not self.ctx.cfg.support_multiturn:
            # if not multiturn, clear history
            self.turns_for_prompt = []

        chat_key = self.ctx.cfg.chat_input_key
        # Prepare the current user turn
        current_turn: dict = {chat_key: user_msg}
        if self.latest_code_status == CodeExecStatus.ENABLED:
            current_turn["total_code_executions"] = self.ctx.cfg.max_code_executions

        self.turns_for_prompt.append(current_turn)

        # Initial UI update – show user question with empty assistant bubble.
        display_history: List[tuple[str, str]] = [
            (t[chat_key], _format_output(t.get("assistant", ""))) for t in self.turns_for_prompt
        ]

        bot_response_so_far = ""
        yield (
            display_history,
            gr.update(value="", visible=False),
            gr.update(visible=False, interactive=False),  # hide send
            gr.update(visible=True, interactive=True),    # show cancel
        )

        try:
            stream = self.ctx.chat.stream_chat(
                self.turns_for_prompt,
                max_tokens,
                temperature,
                self.latest_code_status,
            )
            for chunk in stream:
                if self.cancel_requested:
                    break
                bot_response_so_far += chunk

                # safety check, should not happen
                if not display_history:
                    display_history = [(user_msg, "")]

                display_history[-1] = (user_msg, _format_output(bot_response_so_far))

                yield (
                    display_history,
                    gr.update(value=self.banner_html.value, visible=self.banner_html.visible),
                    gr.update(visible=False),
                    gr.update(),
                )

            # Finalise assistant response for this turn and update history display.
            current_turn["assistant"] = bot_response_so_far
            if display_history:
                display_history[-1] = (user_msg, _format_output(bot_response_so_far))

            yield (
                display_history,
                gr.update(value=self.banner_html.value, visible=self.banner_html.visible),
                gr.update(visible=True, interactive=True),   # show send again
                gr.update(visible=False, interactive=False), # hide cancel
            )
        except Exception as e:
            # On error reset last turn to avoid broken history.
            if self.turns_for_prompt and self.turns_for_prompt[-1] is current_turn:
                self.turns_for_prompt.pop()

            logger.exception("Chat failed")

            yield (
                [(user_msg, f"Error: {e}")],
                self._banner_from_code_status(CodeExecStatus.DISABLED),
                gr.update(visible=True, interactive=True),
                gr.update(visible=False, interactive=False),
            )

    def launch(self):
        return self.demo 

    def _banner_from_code_status(self, code_status: CodeExecStatus):
        sandbox_down = self.ctx.loader.get_code_execution_status(True) != CodeExecStatus.ENABLED  # check if sandbox ok
        if sandbox_down and code_status == CodeExecStatus.ENABLED:
            html_content = (
                "<div style='background-color:#ffcccc;padding:10px;border:2px solid #cc0000;"
                "border-radius:6px;color:#cc0000;font-weight:bold;text-align:center;'>"
                "⚠️ Sandbox is not accessible. Code execution is disabled."
                "</div>"
            )
            return gr.update(value=html_content, visible=True)
        # otherwise hide
        return gr.update(value="", visible=False)
