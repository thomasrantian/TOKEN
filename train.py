# pylint: skip-file
import datetime
import os
import tempfile
from typing import List, Optional, Tuple

import fire
import numpy as np
import transformers
from peft import get_peft_model_state_dict  # noqa: E402
from transformers import logging  # noqa: F402

import wandb
from utils.model_utils import load_llama_tokenizer, load_model
from utils.training_utils import (
    DEFAULT_EVAL_ITEMS,
    decode_generation_seqeunces,
    get_train_val_data,
    log_txt_as_img,
)


class TrainerWithGeneration(transformers.Seq2SeqTrainer):
    """
    Custom Trainer class for sequence-to-sequence model with additional functionalities.
    Inherits from transformers.Seq2SeqTrainer.
    """

    def __init__(self, *args, **kwargs):
        self.vqa = kwargs.pop("vqa", False)
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs["data_collator"].tokenizer

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Overrided method to perform evaluation loop with custom eval and logging.
        """

        # ensure prediction loss is set to False
        prediction_loss_only = False

        # call parent class method to get the evaluation outputs
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # Perform additional operations based on evaluation output
        all_pred_tokens = (
            eval_output.predictions if self.vqa else eval_output.predictions[:, 77:]
        )  # remove the prompt for easier comparison
        all_pred = decode_generation_seqeunces(self.tokenizer, all_pred_tokens)
        all_label = decode_generation_seqeunces(self.tokenizer, eval_output.label_ids)
        all_sample_token_index = dataloader.dataset['plan_query']
        #print("all_pred", all_pred)
        #print("all_label", all_label)

        if self.args.process_index != 0:
            return eval_output

        # Log the predictions
        if wandb.run is None:
            self.log({"i": None})  # dummy log to initialize wandb
        images = log_txt_as_img((512, 512), [all_pred[0], all_label[0]])
        wandb.log({"val_logits": wandb.Image(np.concatenate(images, axis=1))})
        wandb.log(
            {
                "val_results": wandb.Table(
                    columns=["sample_token", "pred", "label"],
                    data=[list(pair) for pair in zip(all_sample_token_index, all_pred, all_label)],
                )
            }
        )

        # Here we manuuly save some modules for customized use
        save_dir = os.path.join(self.args.output_dir, "customized_modules")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Count the number of saved checkpoints
        checkpoint_count = len(os.listdir(save_dir))
        # Use the number of saved checkpoints as the index of the current checkpoint
        save_path = os.path.join(save_dir, f"checkpoint-{checkpoint_count}")
        # Save the model
        self.model.save_pretrained(save_path)
        return eval_output


def train(
    # model/data params
    base_model: str = 'linhvu/decapoda-research-llama-7b-hf',  # the only required argument
    data_path: str = "TOKEN_sample_reasoning_train_data.pkl",
    # training hyperparams
    batch_size: int = 32,
    micro_batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 3e-4,
    val_set_size: int = 32,
    eval_steps: int = 5,
    # lora hyperparams
    lora_r: int = 128,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
    group_by_length: bool = False,
    # wandb params
    wandb_project: str = "TOKEN",
    wandb_run_name: str = "Overfitting test",
    wandb_watch: str = "false",  # options: false | gradients | all
    wandb_log_model: str = "true",  # options: false | true
    resume_from_checkpoint: str = "",
    augment_times: int = 0,
    output_dir: Optional[str] = 'trained_models/',
    vqa: bool = True,
    eval_items: List[str] = ['vqa'],
    mode: str = "train",
    load_pre_prompt_dataset: bool = False,
    val_data_path: str = "TOKEN_sample_reasoning_train_data.pkl",
    feature_mode: str = "object_level_token_visual_only_without_map",
    adapter_mode: str = "direct_inject",
    use_ego_state_info: bool = True,
    root_data_path: str = "sample_train_data/",
    freeze_LORA=False,
):
    #if output_dir is None:
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # creat a folder inside output_dir and name it with the current timestamp
    output_dir = os.path.join(output_dir, f"offline-run-{current_timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    if mode == "eval":
        transformers.set_seed(42)

    # set DDP flags
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print("Training Alpaca-LoRA model with params:")
        for k in [
            "base_model",
            "data_path",
            "output_dir",
            "batch_size",
            "micro_batch_size",
            "num_epochs",
            "learning_rate",
            "val_set_size",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "group_by_length",
            "wandb_project",
            "wandb_run_name",
            "wandb_watch",
            "wandb_log_model",
            "resume_from_checkpoint",
            "mode",
            "eval_items",
        ]:
            print(f"    {k}={eval(k)}")

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    model = load_model(
        base_model=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        resume_from_checkpoint=resume_from_checkpoint,
        feature_mode=feature_mode,
        adapter_mode=adapter_mode,
        use_ego_state_info=use_ego_state_info,
        root_data_path=root_data_path,
        freeze_LORA=freeze_LORA,
    )

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params. VectorLMWithLoRA

    # set and update the data_path
    data_path = os.path.join(root_data_path, data_path)
    val_data_path = os.path.join(root_data_path, val_data_path)

    # Load tokenizer
    tokenizer = load_llama_tokenizer(base_model)

    train_data, val_data = get_train_val_data(
        data_path,
        tokenizer,
        val_data_path=val_data_path,
        val_set_size=val_set_size,
        augment_times=augment_times,
        load_pre_prompt_dataset=load_pre_prompt_dataset,
        vqa=vqa,
        eval_only=mode == "eval",
        eval_items=eval_items,
        #processed_train_data_path="processed_train_data",
    )

    # Initialize trainer
    trainer = TrainerWithGeneration(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.04,
            lr_scheduler_type="cosine",
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=2,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            label_names=[
                "route_descriptors",
                "vehicle_descriptors",
                "pedestrian_descriptors",
                "ego_vehicle_descriptor",
                "user_input_ids",
                "user_attention_mask",
            ],
            prediction_loss_only=False,
            predict_with_generate=True,
            generation_max_length=384,
            generation_config=model.generation_config,
            per_device_eval_batch_size = 16,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        vqa=vqa,
    )

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    logging.set_verbosity_info()
    if mode == "train":
        is_full_checkpoint = os.path.exists(
            os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        )
        trainer.train(resume_from_checkpoint=is_full_checkpoint)
        if local_rank == 0:
            print("🤗🤗🤗🤗🤗Model saved to:", output_dir, "🤗🤗🤗🤗🤗")
            model.save_pretrained(output_dir)
    elif mode == "eval":
        outputs = trainer.evaluate(num_beams = 1)
        print(outputs)


if __name__ == "__main__":
    import time

    st = time.time()
    fire.Fire(train)
    print("Total time:", time.time() - st)
