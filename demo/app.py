"""
EVAFRILL-Mo 3B — Gradio Chat Demo
Hybrid Mamba-2 + Transformer custom LLM
"""

import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import gradio as gr
from tokenizers import Tokenizer
from model.transformer import LLM

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints/3b_dpo/checkpoint-slerp")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer/korean_sp/tokenizer.json")

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
EOS_TOKEN_ID = 2  # tokenizer.token_to_id("</s>")

# ---------------------------------------------------------------------------
# Load model and tokenizer at startup
# ---------------------------------------------------------------------------
print(f"[INFO] Loading tokenizer from {TOKENIZER_PATH}")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

print(f"[INFO] Loading model from {CHECKPOINT_PATH}")
model = LLM.from_pretrained(CHECKPOINT_PATH)
model = model.to(DEVICE, DTYPE)
model.eval()
print("[INFO] Model ready.")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def build_prompt(history: list[tuple[str, str]], message: str) -> str:
    """Concatenate all turns using the chat template."""
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"<|user|>\n{user_msg}\n<|assistant|>\n{bot_msg}</s>"
    prompt += f"<|user|>\n{message}\n<|assistant|>\n"
    return prompt


def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """Penalise tokens that already appear in the sequence."""
    if penalty == 1.0:
        return logits
    # Gather scores for tokens present in the sequence
    score = torch.gather(logits, 1, input_ids)
    # If score > 0, divide; if score < 0, multiply
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, input_ids, score)
    return logits


@torch.inference_mode()
def generate(message: str, history: list[tuple[str, str]], temperature: float, rep_penalty: float, max_tokens: int):
    """Token-by-token generation with streaming via yield."""
    prompt = build_prompt(history, message)

    input_ids = tokenizer.encode(prompt).ids
    ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    generated_text = ""

    for _ in range(max_tokens):
        logits, _ = model(ids)
        logits = logits[:, -1, :].float()  # (1, vocab_size)

        # Apply repetition penalty over the entire context so far
        logits = apply_repetition_penalty(logits, ids, rep_penalty)

        if temperature == 0.0:
            # Greedy decoding
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)
        else:
            # Temperature sampling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        token_id = next_id[0, 0].item()

        if token_id == EOS_TOKEN_ID:
            break

        # Decode and accumulate
        token_str = tokenizer.decode([token_id])
        generated_text += token_str

        # Extend the running context
        ids = torch.cat([ids, next_id], dim=1)

        yield generated_text


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
TITLE = "EVAFRILL-Mo 3B — Hybrid Mamba-2 + Transformer"
DESCRIPTION = (
    "**EVAFRILL-Mo 3B** is a custom Korean-centric language model "
    "combining Mamba-2 selective state-space layers with Transformer attention blocks. "
    "Trained with DDP/FSDP on NVIDIA B200 GPUs and refined via DPO alignment. "
    "Adjust the sliders to control generation quality and length."
)

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=520,
                show_copy_button=True,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="메시지를 입력하세요...",
                    label="",
                    lines=2,
                    scale=8,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear conversation")

        with gr.Column(scale=1):
            temperature = gr.Slider(
                minimum=0.0, maximum=1.5, value=0.7, step=0.05,
                label="Temperature",
                info="0 = greedy, higher = more creative",
            )
            rep_penalty = gr.Slider(
                minimum=1.0, maximum=2.0, value=1.2, step=0.05,
                label="Repetition Penalty",
                info="Penalises repeated tokens",
            )
            max_tokens = gr.Slider(
                minimum=64, maximum=512, value=256, step=32,
                label="Max New Tokens",
            )

    # -----------------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------------
    def user_submit(user_message: str, chat_history: list):
        """Append user message immediately; generation fills assistant slot."""
        if not user_message.strip():
            return "", chat_history
        chat_history = chat_history + [[user_message, ""]]
        return "", chat_history

    def bot_respond(chat_history: list, temperature: float, rep_penalty: float, max_tokens: int):
        """Stream assistant tokens into the last history slot."""
        if not chat_history:
            return chat_history
        user_message = chat_history[-1][0]
        # History up to (but not including) the current turn
        prior_history = chat_history[:-1]

        for partial in generate(user_message, prior_history, temperature, rep_penalty, int(max_tokens)):
            chat_history[-1][1] = partial
            yield chat_history

    send_btn.click(
        fn=user_submit,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
        queue=False,
    ).then(
        fn=bot_respond,
        inputs=[chatbot, temperature, rep_penalty, max_tokens],
        outputs=chatbot,
    )

    msg_box.submit(
        fn=user_submit,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
        queue=False,
    ).then(
        fn=bot_respond,
        inputs=[chatbot, temperature, rep_penalty, max_tokens],
        outputs=chatbot,
    )

    clear_btn.click(lambda: [], outputs=chatbot)


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
