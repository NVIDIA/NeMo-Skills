import argparse
from typing import Callable

import gradio as gr

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
        "--server_type",
        default="sglang",
        choices=["vllm", "sglang", "trtllm"],
        help="Type of the model server to use.",
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


    return parser.parse_args()


def _format_tool_calls(text: str) -> str:
    """Convert <tool_call> tags into Markdown code fences for nicer display."""

    return text.replace("<tool_call>", "```python").replace("</tool_call>", "```")


def _build_chat_fn(llm, prompt: Prompt, max_code_executions: int) -> Callable:
    """Return a Gradio-compatible callback that streams model output."""

    extra_params = prompt.get_code_execution_args()

    def chat_fn(user_input: str, tokens_to_generate: int):
        """Single-turn chat callback streaming tokens one by one."""

        # Clear chat history at the start of every call (single-turn behaviour)
        chat_history = []

        # Build the final prompt string
        prompt_filled = prompt.fill(
            {"problem": user_input, "total_code_executions": max_code_executions}
        )

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


def create_demo(args: argparse.Namespace) -> gr.Blocks:
    """Instantiate sandbox, model, prompt, and wire everything into a Gradio app."""

    sandbox = get_sandbox(host=args.host)

    llm = get_code_execution_model(
        server_type=args.server_type,
        host=args.host,
        sandbox=sandbox,
        code_execution={
            "max_code_executions": args.max_code_executions,
            "add_remaining_code_executions": args.add_remaining_code_executions,
        },
    )

    prompt = get_prompt(
        args.prompt_config,
        args.prompt_template,
    )

    chat_fn = _build_chat_fn(llm, prompt, args.max_code_executions)

    # ----------------------- UI definition ----------------------------------
    with gr.Blocks(theme=gr.themes.Soft(), title="Math Reasoning Chat") as demo:
        gr.Markdown("# Math Reasoning Chat with Code Execution")
        gr.Markdown(
            "Ask a math question. The model can use a Python interpreter to find the solution."
        )

        chatbot = gr.Chatbot(label="Conversation", height=500)

        with gr.Row():
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here and press Enter...",
                scale=4,
                lines=2,
            )
            submit_btn = gr.Button("Submit", variant="primary", scale=1)

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

        gr.Examples(
            examples=[
                "What is the sum of the first 5000 prime numbers?",
                "Solve the equation x^3 - 5x^2 + 6x = 0.",
            ],
            inputs=msg,
            label="Example problems",
        )

        def _clear_chat():
            return None, ""

        # Event wiring -------------------------------------------------------
        submit_btn.click(chat_fn, inputs=[msg, tokens_slider], outputs=[chatbot])
        msg.submit(chat_fn, inputs=[msg, tokens_slider], outputs=[chatbot])
        clear_btn.click(_clear_chat, inputs=[], outputs=[chatbot, msg])

    return demo


if __name__ == "__main__":
    cli_args = parse_args()
    gradio_app = create_demo(cli_args)
    gradio_app.queue().launch()
