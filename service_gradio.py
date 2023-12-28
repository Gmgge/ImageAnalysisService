import gradio as gr
import os
from core.analysis_pipeline.analysis_pipeline import AnalysisPipeline
from core.analysis_pipeline.analysis_pipeline_args import task_candidate_list
from utils.log_init import logger
from utils.web_utils import is_port_usable
from conf.service_args import project_root
from utils.image_utils import read_image_file

# 检查端口
gradio_port = 19192
assert is_port_usable(gradio_port), "ImageAnalysisService 分析 gradio 端口被占用"

# 构建分析模块
logger.info("初始分析模块")
image_analysis_pipeline = AnalysisPipeline()
logger.info("分析模块初始化成功")


def call_analysis(image_data):
    return image_analysis_pipeline.analysis_image(image_data, ["seal_rec"])


if __name__ == "__main__":
    with gr.Blocks() as gradio_demo:
        with gr.Tab("ImageAnalysisService 印章识别"):
            with gr.Row():
                image = gr.Image(label="图像文件", image_mode="RGB")
                output = gr.JSON(label="识别结果")
            btn = gr.Button("分析")
            btn.click(fn=call_analysis, inputs=[image], outputs=output)
            gr.Examples(
                examples=os.path.join(project_root, "tests/core/seal/image"),
                inputs=[image],
                outputs=output,
                fn=call_analysis

            )
    gradio_demo.launch(show_error=True, server_name="localhost", server_port=gradio_port)
