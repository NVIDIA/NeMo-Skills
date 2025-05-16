import argparse
import time
from typing import Callable
from requests.exceptions import ConnectionError
from openai import APIConnectionError

import gradio as gr

from nemo_skills.inference.server.model import get_model
from nemo_skills.inference.server.code_execution_model import get_code_execution_model
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.prompt.utils import get_prompt, Prompt


def parse_args() -> argparse.Namespace:
    """Return CLI arguments for the chat interface."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="localhost",
        help="Hostname where both the model server and the execution sandbox are running.",
    )
    parser.add_argument(
        "--ssh_server",
        default=None,
        help="Server address if model is served remotely or None if model is served locally.",
    )
    parser.add_argument(
        "--ssh_key_path",
        default=None,
        help="Path to the identity file to connect to the `ssh_server`.",
    )
    parser.add_argument(
        "--server_type",
        default="vllm",
        choices=["vllm", "sglang", "trtllm"],
        help="Type of the model server to use.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=["openmath-nemotron", "openmath-nemotron-kaggle"],
        help=("If using one of our recent models, you can specify preset. "
              "This will automatically set `add_remaining_code_executions`, `prompt_config` and `prompt_template`."),
    )
    parser.add_argument(
        "--with_sandbox",
        action="store_true",
        help="Whether to use a model with code execution capabilities.",
    )
    parser.add_argument(
        "--max_code_executions",
        type=int,
        default=8,
        help="Maximum number of Python code blocks that can be executed per generation.",
    )
    parser.add_argument(
        "--add_remaining_code_executions",
        dest="add_remaining_code_executions",
        action="store_true",
        help="Append the number of remaining executions after each code output block.",
    )
    parser.add_argument(
        "--prompt_config",
        default="openmath/tir",
        help="Identifier or path of the prompt config to load.",
    )
    parser.add_argument(
        "--prompt_template",
        default="openmath-instruct",
        help="Identifier or path of the prompt template to load.",
    )

    args = parser.parse_args()
    
    # Apply preset configurations
    if args.preset == "openmath-nemotron":
        args.add_remaining_code_executions = True
        # Default to code reasoning if sandbox is available, otherwise use generic math
        args.prompt_config = "openmath/tir" if args.with_sandbox else "generic/math"
        args.prompt_template = "openmath-instruct"
    elif args.preset == "openmath-nemotron-kaggle":
        args.add_remaining_code_executions = False
        args.prompt_config = "generic/math"
        args.prompt_template = "openmath-instruct"
    
    return args


def _format_tool_calls(text: str) -> str:
    """Convert <tool_call> tags into Markdown code fences for nicer display."""

    return text.replace("<tool_call>", "```python").replace("</tool_call>", "```")


def _build_chat_fn(llm, prompt: Prompt, max_code_executions: int | None = None) -> Callable:
    """Return a Gradio-compatible callback that streams model output."""

    extra_params = prompt.get_code_execution_args()

    def chat_fn(user_input: str, tokens_to_generate: int):
        """Single-turn chat callback streaming tokens one by one."""

        # Clear chat history at the start of every call (single-turn behaviour)
        chat_history = []

        # Build the final prompt string
        prompt_kwargs = {"problem": user_input}
        if max_code_executions is not None:
            prompt_kwargs["total_code_executions"] = max_code_executions
            
        prompt_filled = prompt.fill(prompt_kwargs)

        chat_history.append((user_input, ""))  # show the user message instantly
        yield chat_history

        stream_iter = llm.generate(
            [prompt_filled],
            tokens_to_generate=tokens_to_generate,
            temperature=0.0,
            stream=True,
            stop_phrases=prompt.stop_phrases or [],
            **extra_params,
        )

        bot_response_so_far = ""
        for delta in stream_iter[0]:
            print(delta["generation"], end="")
            bot_response_so_far += delta["generation"]
            chat_history[-1] = (user_input, _format_tool_calls(bot_response_so_far))
            yield chat_history

    return chat_fn


def _get_llm_with_retry(args: argparse.Namespace, wait=30):
    """Attempt to obtain a model, retrying until the server is reachable."""

    while True:
        try:
            if args.with_sandbox:
                sandbox = get_sandbox(host=args.host)
                return get_code_execution_model(
                    server_type=args.server_type,
                    host=args.host,
                    sandbox=sandbox,
                    code_execution={
                        "max_code_executions": args.max_code_executions,
                        "add_remaining_code_executions": args.add_remaining_code_executions,
                    },
                )
            else:
                return get_model(
                    server_type=args.server_type,
                    host=args.host,
                )
        except (ConnectionError, APIConnectionError):
            print(
                "[launch_chat_interface] Could not connect to servers. "
                f"Retrying in {wait} seconds…",
                flush=True,
            )
            time.sleep(wait)


def create_demo(args: argparse.Namespace) -> gr.Blocks:
    """Instantiate model, prompt, and wire everything into a Gradio app."""

    llm = _get_llm_with_retry(args)

    prompt = get_prompt(
        args.prompt_config,
        args.prompt_template,
    )

    max_code_executions = args.max_code_executions if args.with_sandbox else None
    chat_fn = _build_chat_fn(llm, prompt, max_code_executions)

    ui_title = "AI Chat"
    # ----------------------- UI definition ----------------------------------
    with gr.Blocks(theme=gr.themes.Soft(), title=ui_title) as demo:
        ui_subtitle = "Ask a question. " + ("The model can use a Python interpreter." if args.with_sandbox else "")
        
        # If we're using a preset, add it to the title
        if args.preset:
            ui_title += f" ({args.preset})"
            if args.preset == "openmath-nemotron" and args.with_sandbox:
                using_code_reasoning = args.prompt_config == "openmath/tir"
                ui_subtitle += f" Code reasoning is {'enabled' if using_code_reasoning else 'disabled'}."
                
        title_md = gr.Markdown(f"# {ui_title}")
        subtitle_md = gr.Markdown(ui_subtitle)

        chatbot = gr.Chatbot(label="Conversation", height=500)

        with gr.Row():
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here and press Enter...",
                scale=4,
                lines=2,
            )
            submit_btn = gr.Button("Submit", variant="primary", scale=1)

        # Add code reasoning toggle checkbox if using the openmath-nemotron preset with sandbox
        code_reasoning_checkbox = None
        if args.preset == "openmath-nemotron" and args.with_sandbox:
            with gr.Row():
                code_reasoning_checkbox = gr.Checkbox(
                    value=args.prompt_config == "openmath/tir",
                    label="Enable Code Reasoning",
                    info="Use code execution for mathematical reasoning"
                )

        with gr.Row():
            tokens_slider = gr.Slider(
                minimum=1,
                maximum=20000,
                value=16000,
                step=100,
                label="Max new tokens",
                scale=3,
            )
            clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1)

        examples = [
            "What is the sum of the first 5000 prime numbers?",
            "Solve the equation x^13 - 5x^2 + 6 = 0.",
        ]
            
        gr.Examples(
            examples=examples,
            inputs=msg,
            label="Example problems",
        )

        def _clear_chat():
            return None, ""
        
        def _update_model_config(enable_code):
            """Update the model and prompt configuration based on code reasoning toggle."""
            nonlocal llm, prompt, chat_fn, max_code_executions
            
            # Update prompt configuration based on code reasoning setting
            new_prompt_config = "openmath/tir" if enable_code else "generic/math"
            
            # Only update if the configuration has changed
            if args.prompt_config != new_prompt_config:
                args.prompt_config = new_prompt_config
                
                # Get a new prompt with the updated configuration
                prompt = get_prompt(
                    args.prompt_config,
                    args.prompt_template,
                )
                
                # Rebuild the chat function with the new prompt
                chat_fn = _build_chat_fn(llm, prompt, max_code_executions)
                
                # Update the event handlers
                submit_btn.click(chat_fn, inputs=[msg, tokens_slider], outputs=[chatbot])
                msg.submit(chat_fn, inputs=[msg, tokens_slider], outputs=[chatbot])
                
                # Update subtitle to reflect the new code reasoning setting
                new_subtitle = "Ask a question. "
                if args.with_sandbox:
                    new_subtitle += "The model can use a Python interpreter. "
                if args.preset == "openmath-nemotron" and args.with_sandbox:
                    new_subtitle += f"Code reasoning is {'enabled' if enable_code else 'disabled'}."
                
                return new_subtitle
            
            # Return current subtitle if nothing changed
            return subtitle_md.value

        # Event wiring -------------------------------------------------------
        submit_btn.click(chat_fn, inputs=[msg, tokens_slider], outputs=[chatbot])
        msg.submit(chat_fn, inputs=[msg, tokens_slider], outputs=[chatbot])
        clear_btn.click(_clear_chat, inputs=[], outputs=[chatbot, msg])
        
        # Add event handler for code reasoning toggle if it exists
        if code_reasoning_checkbox is not None:
            code_reasoning_checkbox.change(
                _update_model_config, 
                inputs=[code_reasoning_checkbox],
                outputs=[subtitle_md]
            )

    return demo


if __name__ == "__main__":
    cli_args = parse_args()
    gradio_app = create_demo(cli_args)
    gradio_app.queue().launch()
