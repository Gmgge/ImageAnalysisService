import gradio as gr
import os
from core.analysis_pipeline.analysis_pipeline import AnalysisPipeline
from core.analysis_pipeline.analysis_pipeline_args import task_candidate_list
from utils.image_utils import read_image_file, check_image_file
from utils.log_init import logger
from utils.web_utils import is_port_usable
from conf.service_args import project_root, service_config

# 检查端口
gradio_port = 19192
assert is_port_usable(gradio_port), "ImageAnalysisService 分析 gradio 端口被占用"

# 构建分析模块
logger.info("初始分析模块")
image_analysis_pipeline = AnalysisPipeline()
logger.info("分析模块初始化成功")

# 构建任务名列表
analysis_tasks = list(service_config["ModuleSwitch"].keys())
analysis_tasks.remove("analysis_pipeline")


def call_analysis(image_data, task_name_):
    return image_analysis_pipeline.analysis_image(image_data, [task_name_])


def call_image_dir_analysis(image_dir_, task_name_):
    analysis_res = ""
    images = [os.path.join(image_dir_, one_image) for one_image in os.listdir(image_dir_)]
    for image_path in images:
        if not check_image_file(image_path):
            continue
        image_data = read_image_file(image_path)
        res_info = image_analysis_pipeline.analysis_image(image_data, [task_name_])
        task_res = res_info["data"][task_name_]
        analysis_res += f"图像:{os.path.basename(image_path)}, 分析结果:{task_res} \n \n"
        yield analysis_res


if __name__ == "__main__":
    with gr.Blocks() as gradio_demo:
        with gr.Tab("图像分析"):
            with gr.Row():
                task_name = gr.Dropdown(analysis_tasks,
                                        value="ocr",
                                        label="任务类型",
                                        info="请选择分析任务类型")
                image = gr.Image(label="图像文件", image_mode="RGB")
                output = gr.JSON(label="识别结果")
            btn = gr.Button("分析")
            btn.click(fn=call_analysis, inputs=[image, task_name], outputs=output)
            gr.Examples(
                examples=os.path.join(project_root, "tests/core/seal/image"),
                inputs=[image],
                outputs=output,
                fn=call_analysis
            )
        with gr.Tab("本地批量分析"):
            with gr.Row():
                task_name = gr.Dropdown(analysis_tasks,
                                        value="ocr",
                                        label="任务类型",
                                        info="请选择分析任务类型")
                image_dir = gr.Text(label="本地图像文件夹", info="请注意 该分析仅支持本地批量分析，远程请从接口调用")
                output = gr.Textbox(label="识别结果")
            btn = gr.Button("分析")
            btn.click(fn=call_image_dir_analysis, inputs=[image_dir, task_name], outputs=output)

    gradio_demo.launch(show_error=True, server_name="0.0.0.0", server_port=gradio_port)
