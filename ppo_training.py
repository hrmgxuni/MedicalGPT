# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train a model from SFT using PPO
"""

import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BloomForCausalLM,
    AutoModelForCausalLM,
    AutoModel,
    LlamaTokenizer,
    LlamaForCausalLM,
    BloomTokenizerFast,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed

from supervised_finetuning import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}

import gpu_util

gpu_logger = gpu_util.GPUConsoleLogger()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    # Model arguments
    model_type: str = field(
        default="bloom",
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization."}
    )
    reward_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "The reward model name"})
    reward_model_device: Optional[str] = field(default="cuda:0", metadata={"help": "The reward model device"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The template name."})
    batch_size: Optional[int] = field(default=8, metadata={"help": "Batch size"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "PPO minibatch size"})
    max_source_length: Optional[int] = field(default=256, metadata={"help": "Max length of prompt input text"})
    max_target_length: Optional[int] = field(default=256, metadata={"help": "Max length of output text"})
    min_target_length: Optional[int] = field(default=4, metadata={"help": "Min length of output text"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    # Training arguments
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default=None)
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    # PPO arguments
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the validation set."})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "Whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "The kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0, metadata={"help": "Baseline value that is subtracted from the reward"},
    )
    init_kl_coef: Optional[float] = field(
        default=0.2, metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    learning_rate: Optional[float] = field(default=1.5e-5, metadata={"help": "Learning rate"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    save_steps: Optional[int] = field(default=50, metadata={"help": "X steps to save the model"})
    output_dir: Optional[str] = field(default="outputs-rl", metadata={"help": "The output directory"})
    seed: Optional[int] = field(default=0, metadata={"help": "Seed"})
    max_steps: Optional[int] = field(default=200, metadata={"help": "Number of steps to train"})
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": "Report to wandb or tensorboard"})

    def __post_init__(self):
        if self.model_type is None:
            raise ValueError("You must specify a valid model_type to run training.")
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")
        if self.reward_model_name_or_path is None:
            raise ValueError("You must specify a valid reward_model_name_or_path to run training.")
        if self.max_source_length < 60:
            raise ValueError("You must specify a valid max_source_length >= 60 to run training")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_reward_model_output(reward_model, reward_tokenizer, question, answer, device):
    """
    Get the reward score for a given question and answer pair.
    """
    inputs = reward_tokenizer(question, answer, return_tensors='pt').to(device)
    score = reward_model(**inputs).logits[0].cpu().detach()

    return score


def calculate_rewards(reward_score_outputs, reward_baseline=0):
    """
    Calculate the reward for a given score output.
    :param reward_score_outputs: 
    :param reward_baseline: 
    :return: 
    """
    rewards = []
    for score in reward_score_outputs:
        if isinstance(score, torch.Tensor) and score.numel() == 1:
            reward_value = score.item() - reward_baseline
            rewards.append(torch.tensor(reward_value))
        else:
            # Use the average of the tensor elements as `score` is multiple elements
            reward_value = torch.mean(score).item() - reward_baseline
            rewards.append(torch.tensor(reward_value))
    return rewards




def main(args):
    # 1 从命令行解析参数
    # parser = HfArgumentParser(ScriptArguments)
    # args = parser.parse_args_into_dataclasses()[0]
    # 初始化ScriptArguments对象

    logger.info(f"Parse args: {args}")

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.model_type == 'bloom':
        args.use_fast_tokenizer = True

    gpu_logger.use("加载 tokenizer 前")
    # 2 加载tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "trust_remote_code": args.trust_remote_code,
    }
    gpu_logger.use("加载 tokenizer 后")

    tokenizer_name_or_path = args.tokenizer_name_or_path
    # 如果tokenizer_name_or_path为空，则将采用模型的分词器
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = args.model_name_or_path
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0  # set as the <unk> token

    # Load model
    peft_config = None
    if args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=args.target_modules,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    else:
        logger.info("Fine-tuning method: Full parameters training")
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    config = config_class.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        peft_config=peft_config if args.use_peft else None,
    )
    gpu_logger.use("加载 model")

    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    print_trainable_parameters(model)
    # Load reward model
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.reward_model_device if args.reward_model_device is not None else default_device
    reward_config = config_class.from_pretrained(
        args.reward_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name_or_path,
        config=reward_config,
        load_in_8bit=args.load_in_8bit,
        trust_remote_code=args.trust_remote_code,
    )
    gpu_logger.use("加载 reward_model")

    reward_model.to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_name_or_path, **tokenizer_kwargs
    )
    gpu_logger.use("加载 reward_tokenizer")

    # Get datasets
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    else:
        data_files = {}
        if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
            train_data_files = glob(f'{args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
            eval_data_files = glob(f'{args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
        gpu_logger.use("加载 raw_datasets")
    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    prompt_template = get_conv_template(args.template_name)

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        roles = ["human", "gpt"]

        def get_prompt(examples):
            for i, source in enumerate(examples['conversations']):
                if len(source) < 2:
                    continue  # 如果少于2条消息，则跳过到下一个例子
                # 获取对话中第一条消息的角色
                data_role = source[0].get("from", "")
                # 检查角色是否不在指定的角色列表中，或者是否不是预期的角色
                if data_role not in roles or data_role != roles[0]:
                    # 如果第一条消息不是预期的角色，则跳过第一条消息
                    source = source[1:]
                # 跳过第一条消息后，检查对话长度是否仍小于2
                if len(source) < 2:
                    continue  # 如果仍少于2条消息，则跳过到下一个例子
                messages = []
                # 遍历对话中的每条消息
                for j, sentence in enumerate(source):
                    # 获取消息的角色
                    data_role = sentence.get("from", "")
                    # 检查角色是否不在指定的角色列表中
                    if data_role not in roles:
                        # 记录警告消息并忽略此消息
                        logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                        break  # 停止处理此例子
                    # 检查消息的角色是否与当前对话轮次的预期角色匹配
                    if data_role == roles[j % 2]:
                        # 将消息内容添加到消息列表中
                        messages.append(sentence["value"])
                # 检查消息数量是否少于2或是否为奇数条消息
                if len(messages) < 2 or len(messages) % 2 != 0:
                    continue  # 如果消息数量不是偶数或少于2，则跳过到下一个例子
                # 将消息列表转换为成对的元素
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
                # 返回生成的提示
                yield prompt_template.get_prompt(history_messages)

        for prompt in get_prompt(examples):
            for i in range(len(prompt) // 2):
                source_txt = prompt[2 * i]
                tokenized_question = tokenizer(
                    source_txt, truncation=True, max_length=max_source_length, padding="max_length",
                    return_tensors="pt"
                )
                new_examples["query"].append(source_txt)
                new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    # Preprocess the dataset
    train_dataset = None
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        if args.max_train_samples is not None and args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        tokenized_dataset = train_dataset.shuffle().map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        train_dataset = tokenized_dataset.filter(
            lambda x: len(x['input_ids']) > 0
        )
        gpu_logger.use("if args.do_train: 加载 train_dataset")
        logger.debug(f"Num train_samples: {len(train_dataset)}")

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    output_dir = args.output_dir
    config = PPOConfig(
        steps=args.max_steps,
        model_name=args.model_name_or_path,
        learning_rate=args.learning_rate,
        log_with=args.report_to,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=args.early_stopping,
        target_kl=args.target_kl,
        seed=args.seed,
        init_kl_coef=args.init_kl_coef,
        adap_kl_ctrl=args.adap_kl_ctrl,
        project_kwargs={"logging_dir": output_dir},
    )
    # Set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )
    gpu_logger.use("加载 PPOTrainer")

    # These arguments are passed to the `generate` function of the PPOTrainer
    generation_kwargs = {
        "max_new_tokens": max_target_length,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "top_p": 1.0,
        "do_sample": True,
    }
    gpu_logger.use("加载 generation_kwargs")

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        gpu_logger.use("Train 前")
        total_steps = config.total_ppo_epochs
        for step, batch in tqdm(enumerate(trainer.dataloader)):
            gpu_logger.use(f"Train 中 step={step}")
            if step >= total_steps:
                break
            question_tensors = batch["input_ids"]
            question_tensors = [torch.LongTensor(i).to(device).squeeze(0) for i in question_tensors]
            responses = []
            response_tensors = []
            for q_tensor in question_tensors:
                response_tensor = trainer.generate(
                    q_tensor,
                    return_prompt=False,
                    **generation_kwargs,
                )
                r = tokenizer.batch_decode(response_tensor, skip_special_tokens=True)[0]
                responses.append(r)
                response_tensors.append(response_tensor.squeeze(0))
            batch["response"] = responses

            # Compute reward score
            score_outputs = [
                get_reward_model_output(reward_model, reward_tokenizer, q, r, device) for q, r in
                zip(batch["query"], batch["response"])
            ]
            rewards = calculate_rewards(score_outputs, args.reward_baseline)

            # Run PPO step
            try:
                stats = trainer.step(question_tensors, response_tensors, rewards)
                trainer.log_stats(stats, batch, rewards)
                logger.debug(f"Step {step}/{total_steps}: reward score:{score_outputs}")
            except ValueError as e:
                logger.warning(f"Failed to log stats for step {step}, because of {e}")

            if step and step % args.save_steps == 0:
                save_dir = os.path.join(output_dir, f"checkpoint-{step}")
                trainer.save_pretrained(save_dir)

        gpu_logger.use(f"Train 后")
        # Save final model
        trainer.save_pretrained(output_dir)

def main2():
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
    main(request)

if __name__ == "__main__":
    main2()
