import os

import gradio as gr

from inference import run


def respond(message, history):
    msg_history = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                msg_history.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_text, assistant_text = item
            if user_text:
                msg_history.append({"role": "user", "content": user_text})
            if assistant_text:
                msg_history.append({"role": "assistant", "content": assistant_text})
    output = run(message, msg_history)
    return output


def main():
    demo = gr.ChatInterface(
        fn=respond,
        title="Pocket-Agent",
        description="Offline tool-calling demo",
    )
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    demo.launch(
        share=share,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
