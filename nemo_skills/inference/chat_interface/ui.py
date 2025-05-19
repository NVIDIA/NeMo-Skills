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

        with gr.Blocks(title="AI Chat Interface", theme=gr.themes.Soft()) as self.demo:
            # chat panel needs to exist before server panel (server panel references it in outputs list)
            self._build_chat_panel()
            self._build_server_config_panel()

        # Show chat by default if launch_mode == "direct"
        if ctx.cfg.launch_mode == "direct":
            self.server_group.visible = False
            self.chat_group.visible = True
            self.code_exec_checkbox.interactive = bool(ctx.loader.code_llm)

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------
    def _build_server_config_panel(self):
        cfg = self.ctx.cfg
        with gr.Column(visible=(cfg.launch_mode == "manual")) as self.server_group:
            gr.Markdown("## Server & Model Configuration")
            self.host_tb = gr.Textbox(label="Host", value=cfg.host)
            self.server_dd = gr.Dropdown(["vllm", "sglang", "openai"], value=cfg.server_type, label="Server Type")
            self.ssh_server_tb = gr.Textbox(label="SSH server (optional)")
            self.ssh_key_tb = gr.Textbox(label="SSH key path (optional)")
            self.connect_btn = gr.Button("Connect", variant="primary")
            self.server_status = gr.Markdown("")

            self.connect_btn.click(
                self.on_connect,
                inputs=[self.host_tb, self.server_dd, self.ssh_server_tb, self.ssh_key_tb],
                outputs=[
                    self.server_status,
                    self.server_group,
                    self.chat_group,
                    self.code_exec_checkbox,
                ],
            )

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
            self.chatbot = gr.Chatbot(height=450, bubble_full_width=False)
            with gr.Row():
                self.msg_tb = gr.Textbox(label="Your question", lines=3, scale=4)
                self.submit_btn = gr.Button("Send", variant="primary")
            self.clear_btn = gr.Button("Clear chat")

            # Bind events
            self.code_exec_checkbox.change(
                self.on_toggle_code_exec,
                inputs=[self.code_exec_checkbox],
                outputs=[self.subtitle_md, self.code_exec_checkbox, self.banner_html],
            )
            self.submit_btn.click(
                self.handle_chat_submit,
                inputs=[self.msg_tb, self.max_tokens, self.temperature],
                outputs=[self.chatbot, self.banner_html],
            ).then(lambda: gr.update(value=""), None, [self.msg_tb])
            self.msg_tb.submit(
                self.handle_chat_submit,
                inputs=[self.msg_tb, self.max_tokens, self.temperature],
                outputs=[self.chatbot, self.banner_html],
            ).then(lambda: gr.update(value=""), None, [self.msg_tb])
            self.clear_btn.click(lambda: (None, ""), None, [self.chatbot, self.msg_tb])

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------
    def on_connect(self, host: str, server_type: str, ssh_server: str, ssh_key: str):
        # Update runtime config before loading
        self.ctx.cfg.host = host or self.ctx.cfg.host
        self.ctx.cfg.server_type = server_type or self.ctx.cfg.server_type
        self.ctx.cfg.ssh_server = ssh_server or None
        self.ctx.cfg.ssh_key_path = ssh_key or None

        # Replace loader & chat service with new connection details
        self.ctx.loader = ModelLoader(self.ctx.cfg)
        self.ctx.chat = ChatService(self.ctx.loader, self.ctx.prompts)

        ok, err_generic = self.ctx.loader.load_generic()
        if not ok:
            return f"<span style='color:red'>Generic model error: {err_generic}</span>"
        code_ok, code_err = self.ctx.loader.load_code_and_sandbox()
        status_lines = ["Generic model ✔" if ok else f"Generic model ❌ ({err_generic})"]
        status_lines.append("Code model ✔" if code_ok else f"Code model ⚠ ({code_err})")
        return (
            "<br/>".join(status_lines),
            gr.update(visible=False),  # hide server page
            gr.update(visible=True),   # show chat page
            gr.update(interactive=code_ok),
        )

    def on_toggle_code_exec(self, checkbox_val: bool):
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

    def handle_chat_submit(self, user_msg: str, max_tokens: int, temperature: float):
        if not user_msg.strip():
            yield ([(user_msg, "Please enter a question.")], gr.update(value="", visible=False))
            return

        # If user just toggled, refresh code_status.
        self.latest_code_status = self.ctx.loader.get_code_execution_status(self.code_exec_checkbox.value)
        if self.latest_code_status == CodeExecStatus.ENABLED and not self.ctx.loader.code_llm:
            # Lazy-load code model if needed
            self.ctx.ensure_code_ready()
            self.latest_code_status = self.ctx.loader.get_code_execution_status(True)

        history: List = []
        bot_response_so_far = ""
        yield (history, gr.update(value="", visible=False))

        try:
            stream = self.ctx.chat.stream_chat(
                user_msg, tokens_to_generate=max_tokens, temperature=temperature, status=self.latest_code_status
            )
            for chunk in stream:
                bot_response_so_far += chunk
                history = [(user_msg, _format_output(bot_response_so_far))]
                yield history, gr.update(value=self.banner_html.value, visible=self.banner_html.visible)
        except Exception as e:
            logger.exception("Chat failed")
            yield [(user_msg, f"Error: {e}")], self._banner_from_code_status(CodeExecStatus.DISABLED)

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