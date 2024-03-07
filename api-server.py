from fastapi import FastAPI, BackgroundTasks, UploadFile, File
# from fastapi.responses import FileResponse
# from subprocess import Popen, PIPE, STDOUTpip
import os
from dataclasses import dataclass, field
from typing import Optional, Generic, TypeVar, Dict
import json
import logging
import requests
#from trl_0226 import run, ScriptArguments
from ppo_training import main, ScriptArguments
import traceback
from loguru import logger
logger = logging.getLogger(__name__)

    

app = FastAPI()

# 预设一个变量用于存储任务状态
# 在实际应用中可能需要一个更复杂的机制，比如数据库或者缓存系统来跟踪
task_is_running = False


async def write_ppo_train(train: UploadFile = File(...), train_path="./data/ppo-train.json", output_path="./data/finetune/ppo_train_formate.jsonl"):
# def write_ppo_train(train, train_path="./data/ppo-train.json", output_path="./data/finetune/ppo_train_formate.jsonl"):
    # 保存训练集和评估集的 JSON 文件
    with open(train_path, "wb") as f:
        f.write(await train.read())

    print("train_path11********")
    converted_data_str = ""
    with open(train_path, "r", encoding="utf-8") as f:
        data = f.read()
        # print(f"data{data}")
        for item in json.loads(data):
            # print(f'item{item}')
            converted_data = {"conversations": []}
            converted_data["conversations"].append({"from": "human", "value": f"{item['instruction']} {item['input']}"})
            converted_data["conversations"].append({"from": "gpt", "value": f"{item['output']}"})
            # converted_data_str += f'{converted_data}\n'
            # 保证输出的是双引号包含的json字符串，而不是单引号包含
            # 将 ensure_ascii 设置为 False，以保留非 ASCII 字符的原始形式
            converted_data_str += f'{json.dumps(converted_data, ensure_ascii=False)}\n'



        # print(converted_data_str
        with open(output_path, "w", encoding="utf-8") as train_file:
            train_file.write(converted_data_str)

    
@app.post("/ppo-tune-model")
#async def ppo_tune_model(background_tasks: BackgroundTasks, data: UptorchloadFile = File(...)):
async def ppo_tune_model(background_tasks: BackgroundTasks, train: UploadFile = File(...), eval: UploadFile = File(...), data: UploadFile = File(...)):
    global task_is_running

    # 记录每次请求，无论任务是否需要执行
    print("接收到模型调优请求")

    # 防重逻辑
    if task_is_running:
        # 如果任务正在运行，返回相应的信息
        return {"message": "任务已在执行，不会重复运行"}

    try:
        # 定义文件夹路径和文件名
        folder_path = "upload/ppo/"
        # 如果文件夹不存在，则创建
        os.makedirs(folder_path, exist_ok=True)

        # 保存训练集和评估集的 JSON 文件
        # with open(f"{folder_path}/train.json", "wb") as f:
        #    f.write(await train.read())
        print("wrfdasfdasfdaite_ppo_train********")
        await write_ppo_train(train)
        print("write_ppo_train********")
        with open(f"./data/ppo-eval.json", "wb") as f:
            f.write(await eval.read())
            print("write_ppo_train********")
        with open(f"./data/ppo-data.json", "wb") as f:
            f.write(await data.read())
        request = ScriptArguments(
            model_type="bloom",
            model_name_or_path="D:/TPHY/bigscience/bloomz-560m",
            reward_model_name_or_path="D:/TPHY/bigscience/bloomz-560m",
            torch_dtype="float16",
            device_map="auto",
            train_file_dir="./data/finetune",
            validation_file_dir="./data/finetune",
            batch_size=8,
            max_source_length=256,
            max_target_length=256,
            max_train_samples=1000,
            use_peft=True,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
            do_train=True,
            max_steps=100,
            learning_rate=1e-5,
            save_steps=50,
            output_dir="outputs-rl-bloom-v1",
            early_stopping=True,
            target_kl=0.1,
            reward_baseline=0.0
        )

        print("fdsafsdafsadfasfsda********")
        with open(f"./data/ppo-data.json", "r") as f:
            content = f.read()
            try:
                data_list = json.loads(content)  # 解析为列表
                if len(data_list) > 0:
                    # 创建RequestParams实例
#                    request = ScriptArguments(**data_list[0]['dict'])
                    request.id = data_list[0]['id']

                    # 假设 self.dict 由 TrainRequest 的初始化方法或某些赋值语句填充
#                    request.dict = json.loads(data_list.get('dict', {}))
                    train_request_dict = data_list[0]['dict']

                    # 根据dict内的值修改对应的值
#                    request.ppo_epochs = train_request_dict["num_train_epochs"]
#                    request.batch_size = train_request_dict["per_device_train_batch_size"]
#                    request.learning_rate = train_request_dict["learning_rate"]
#                    request.lr_scheduler_type = train_request_dict["lr_scheduler_type"]
#                    request.warmup_steps = train_request_dict["warmup_steps"]
#                    request.gradient_accumulation_steps = train_request_dict["gradient_accumulation_steps"]
#
#                    request.per_device_eval_batch_size = train_request_dict["per_device_eval_batch_size"]
#                    request.logging_steps = train_request_dict["logging_steps"]
                    # 检查并赋值 ppo_epochs
                    # if "num_train_epochs" in train_request_dict:
                    #     request.ppo_epochs = train_request_dict["num_train_epochs"]
                    #
                    # # 检查并赋值 batch_size
                    # if "per_device_train_batch_size" in train_request_dict:
                    #     request.batch_size = train_request_dict["per_device_train_batch_size"]
                    #
                    # # 检查并赋值 learning_rate
                    # if "learning_rate" in train_request_dict:
                    #     request.learning_rate = train_request_dict["learning_rate"]
                    #
                    # # 检查并赋值 lr_scheduler_type
                    # if "lr_scheduler_type" in train_request_dict:
                    #     request.lr_scheduler_type = train_request_dict["lr_scheduler_type"]
                    #
                    # # 检查并赋值 warmup_steps
                    # if "warmup_steps" in train_request_dict:
                    #     request.warmup_steps = train_request_dict["warmup_steps"]
                    #
                    # # 检查并赋值 gradient_accumulation_steps
                    # if "gradient_accumulation_steps" in train_request_dict:
                    #     request.gradient_accumulation_steps = train_request_dict["gradient_accumulation_steps"]
                    #
                    # # 检查并赋值 per_device_eval_batch_size
                    # if "per_device_eval_batch_size" in train_request_dict:
                    #     request.per_device_eval_batch_size = train_request_dict["per_device_eval_batch_size"]
                    #
                    # # 检查并赋值 logging_steps
                    # if "logging_steps" in train_request_dict:
                    #     request.logging_steps = train_request_dict["logging_steps"]

                    print("request.dataset_name: ", request.dataset_name)
                    print(f"request: {request}")
                else:
                    raise ValueError("Empty list in JSON data")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print("Invalid JSON format or missing keys:", str(e))
                # 返回错误消息
                print("Invalid JSON format or missing keys")
        print(f"request: {request}")

        # 函数定义
        def train_model_in_background(request: ScriptArguments):
            try:
                # 开始训练
                logger.info("*** starting training ***")
                # url = f"http://182.92.74.187:8123/prod-api/fine/tuning/record/status/{request.id}/1"
                # response = requests.get(url)
                # data = {"fineTuningRecordId": request.id, "message": "微调开始", "logLevel": "2"}
                # response = requests.post("http://182.92.74.187:8123/prod-api/log/insertTuningLog", json=data)
                # if response.status_code != 200:
                #     logger.error("微调开始记录日志请求失败")

                # ***** 这里编写训练逻辑代码 *****
                global task_is_running
                task_is_running = True

                print("train_model_in_background.request = ", request)
                main(request)

                task_is_running = False
                # ***** 这里编写训练逻辑代码 *****

                # 记录日志
                # data = {"fineTuningRecordId": request.id, "message": "训练结束", "logLevel": "2"}
                # response = requests.post("http://182.92.74.187:8123/prod-api/log/insertTuningLog", json=data)
                print("训练结束")
                # if response.status_code != 200:
                #     logger.error("记录日志请求失败")
                # # 更新记录状态
                # url = f"http://182.92.74.187:8123/prod-api/fine/tuning/record/status/{request.id}/7"
                # response = requests.get(url)
                # print(response)
                # if response.status_code != 200:
                #     logger.error("更新记录状态请求失败")

            except Exception as e:
                task_is_running = False
                logger.error("An error occurred during training.", exc_info=True)
        
        # 提交后台任务
        background_tasks.add_task(train_model_in_background, request)
        # 返回响应
        return {"message": "模型调优启动，正在后台执行"}
    except Exception as e:
        traceback.print_exc()
        print(f"error: {str(e)}")
        logger.error("微调失败")
        # 返回错误消息
        return {"error": str(e)}

# 在本模块直接运行时启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api-server:app", host="0.0.0.0", port=6006)