import time
import gradio as gr
import spaces
import torch

@spaces.GPU(duration=10)
def gpu_test():
    t = time.time()
    x = torch.randn(4096, 4096, device="cuda")
    for _ in range(20):
        x = x @ x
        x = x / x.norm()
    torch.cuda.synchronize()
    return f"GPU: {torch.cuda.get_device_name(0)} | took {time.time()-t:.2f}s"

with gr.Blocks() as demo:
    gr.Markdown("# dribble3d: GPU test")
    btn = gr.Button("Run GPU test")
    out = gr.Textbox(label="Result")
    btn.click(gpu_test, outputs=out)

demo.launch()