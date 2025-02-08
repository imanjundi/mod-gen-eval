import logging
import os

import datasets
import numpy as np
import pandas as pd
import wandb
import torch
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from trl import SFTTrainer
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

import data
from args import parse_arguments, save_args
from data import ORIGINAL_MODERATION_COL, GENERATED_MODERATION_COL, \
    filter_by_moderation_types
from moderation_type import ModerationType
from trainer import CustomTrainer


# add main
def main():
    # # data
    # load_saved = True
    # load_saved = False
    #
    # if load_saved:
    #     data = pd.read_csv('moderated_comments_output.csv')
    #     moderated_df = data[data['broadening']==1]
    # else:
    # data_path = Path().home()

    set_seed(42)

    model_args, data_args, training_args, custom_args, debug = parse_arguments()

    logging.basicConfig(level=custom_args.logging_level.upper(),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(module)s - %(funcName)s : %(message)s",
                        datefmt="%m/%d/%y %I:%M:%S %p", )

    # assert path doesn't exist
    if not training_args.overwrite_output_dir:
        assert not os.path.exists(training_args.output_dir)
        # make dir with all parent dirs in the path
        os.makedirs(training_args.output_dir, exist_ok=False)
    else:
        assert os.path.exists(training_args.output_dir)

    moderation_types = custom_args.moderation_types

    if data_args.dataset != 'usermod':
        logging.info('Using preprocessed dataset')
        data_args.data_dir = '.'
    train_df, df = data.read_data_splits(data_args)

    if train_df is not None:
        print(f'dataset size: {train_df.shape[0]}')
    else:
        print('no test dataset')
    if df is not None:
        print(f'dataset size: {df.shape[0]}')
    else:
        print('no test dataset')

    if ModerationType.GENERAL_MODERATION not in moderation_types:
        df = filter_by_moderation_types(df, moderation_types)
    if data_args.batch:
        logging.warning(f'Using batch {data_args.batch}')
        if data_args.dataset == 'annotated':
            df = df[df['batch'] == data_args.batch]
        elif data_args.dataset == 'aggregated_annotated':
            batches = np.array_split(df, 5)
            df = batches[data_args.batch - 1]
    if data_args.sample_size > 0:
        if df is not None:
            df = df.sample(data_args.sample_size, random_state=42)
        if train_df is not None:
            train_df = train_df.sample(data_args.sample_size, random_state=42)

    # model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_id = model_args.model_name_or_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if custom_args.load_peft_model:
        # Load Model with PEFT adapter
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
    elif device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     #  torch_dtype=torch.float16,
                                                     torch_dtype=torch.bfloat16,
                                                     quantization_config=bnb_config,
                                                     attn_implementation="flash_attention_2",
                                                     #  map_location=device
                                                     device_map="auto"
                                                     )

        # Wrap the model with DataParallel if multiple GPUs are available
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs for data parallelism.")
        #     model = torch.nn.DataParallel(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
    # tokenizer.padding_side = 'right'  # to prevent warnings
    # ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)`
    tokenizer.pad_token = tokenizer.eos_token

    custom_trainer = CustomTrainer(device, model, tokenizer, data_args, model_args, training_args, custom_args)

    if training_args.do_train:
        logging.info('processing data...')
        assert moderation_types == [ModerationType.GENERAL_MODERATION], 'only general moderation supported'

        # quick fix with pandas for now
        if custom_args.training_method == 'orpo':
            def map_to_messages(sample, col):
                return custom_trainer.create_messages(
                    comment=sample[data.USER_COMMENT_COL], reply=sample[col],
                    topic=None)

            train_df['chosen'] = train_df.apply(lambda row: map_to_messages(row, 'chosen'), axis=1)
            train_df['rejected'] = train_df.apply(lambda row: map_to_messages(row, 'rejected'), axis=1)
            train_df['prompt'] = train_df.apply(lambda row: row['chosen'][1]['content'], axis=1)

            dataset_name = 'preference'
            # train_cols = [x for x in train_df.columns if x not in ['reply_1', 'reply_2']]
            train_cols = train_df.columns
        else:
            dataset_name = 'regroom'
            train_cols = ['id', 'comment id', data.USER_COMMENT_COL, data.ORIGINAL_MODERATION_COL, 'type']

        dataset = datasets.Dataset.from_pandas(train_df[train_cols], split='train')

        dataset = dataset.shuffle()
        dataset.to_json(f"data/train_dataset_{dataset_name}.json", orient="records")

        logging.info('training...')
        # based on https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
        if custom_args.training_method == 'sft':
            def map_fn(sample):
                return {
                    # removed now till figuring out how to pass 'messages' (not text_field)
                    # 'id': sample['id'],
                    'messages': custom_trainer.create_messages(
                        comment=sample[data.USER_COMMENT_COL], reply=sample[data.ORIGINAL_MODERATION_COL],
                        topic=sample['type'])
                }

            dataset = dataset.map(map_fn,
                                  remove_columns=[data.USER_COMMENT_COL, data.ORIGINAL_MODERATION_COL],
                                  batched=False)

            # LoRA config based on QLoRA paper & Sebastian Raschka experiment
            peft_config = LoraConfig(
                lora_alpha=128,
                lora_dropout=0.05,
                r=256,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )

            training_args.num_train_epochs = 3  # number of training epochs
            training_args.per_device_train_batch_size = 3  # batch size per device during training
            training_args.gradient_accumulation_steps = 2  # number of steps before performing a backward/update pass
            training_args.gradient_checkpointing = True  # use gradient checkpointing to save memory
            training_args.optim = "adamw_torch_fused"  # use fused adamw optimizer
            training_args.logging_steps = 10  # log every 10 steps
            training_args.save_strategy = "epoch"  # save checkpoint every epoch
            training_args.learning_rate = 2e-4  # learning rate based on QLoRA paper
            training_args.bf16 = True  # use bfloat16 precision
            training_args.tf32 = True  # use tf32 precision
            training_args.max_grad_norm = 0.3  # max gradient norm based on QLoRA paper
            training_args.warmup_ratio = 0.03  # warmup ratio based on QLoRA paper
            training_args.lr_scheduler_type = "constant"  # use constant learning rate scheduler
            # training_args.push_to_hub = True  # push model to hub
            # training_args.report_to = "tensorboard"  # report metrics to tensorboard

            training_args.packing = (data_args.sample_size == 0 or data_args.sample_size > 10)
            # which field
            # training_args.dataset_text_field = "messages"

            save_args(data_args, model_args, training_args, custom_args)

            max_seq_length = 3072  # max sequence length for model and packing of the dataset

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                peft_config=peft_config,
                max_seq_length=max_seq_length,
                tokenizer=tokenizer,
                dataset_kwargs={
                    "add_special_tokens": False,  # We template with special tokens
                    "append_concat_token": False,  # No need to add additional separator token
                }
            )
        # based on https://huggingface.co/blog/mlabonne/orpo-llama-3
        elif custom_args.training_method == 'orpo':
            def format_chat_template(row):
                row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
                row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
                return row

            dataset = dataset.map(
                format_chat_template,
                num_proc=os.cpu_count(),
            )

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=(['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
                                if not debug else "all-linear")
            )
            model, tokenizer = setup_chat_format(model, tokenizer)
            model = prepare_model_for_kbit_training(model)

            # ORPO uses very low learning rates compared to traditional SFT or even DPO. This value of 8e-6 comes from the original paper,
            # and roughly corresponds to an SFT learning rate of 1e-5 and a DPO learning rate of 5e-6.
            # I would recommend increasing it around 1e-6 for a real fine-tune.
            # maximum in paper 8e-6
            training_args.learning_rate = 1e-6
            # the $\lambda$ parameter in the paper, with a default value of 0.1. An appendix from the original paper shows how it's been selected with an ablation study.
            training_args.beta = 0.1
            training_args.lr_scheduler_type = "linear"
            # increased from 1024 to 2048
            training_args.max_length = 2048
            training_args.max_prompt_length = 512
            training_args.per_device_train_batch_size = 3
            training_args.per_device_eval_batch_size = 3
            training_args.gradient_accumulation_steps = 2
            training_args.optim = "paged_adamw_8bit"
            # Ideally we would train the model for 3-5 epochs but we'll stick to 1 here
            training_args.num_train_epochs = 1
            training_args.evaluation_strategy = "steps"
            training_args.eval_steps = 0.2
            training_args.logging_steps = 1
            training_args.warmup_steps = 10
            training_args.report_to = ["wandb"]

            trainer = ORPOTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                peft_config=peft_config,
                tokenizer=tokenizer,
            )

        else:
            raise ValueError(f'Unknown training method: {custom_args.training_method}')
        if debug:
            return 1
        trainer.train()
        trainer.save_model()

        # free the memory again
        del model
        del trainer
        torch.cuda.empty_cache()

    # debug first 3 samples
    logging.getLogger().setLevel('DEBUG')
    if training_args.do_predict:
        custom_trainer.predict(df, moderation_types)

    if training_args.do_eval:
        if custom_args.eval_generated_input_dir:
            eval_df = pd.read_csv(f"{custom_args.eval_generated_input_dir}/moderated_comments_output.csv")
            eval_df['id'] = eval_df['id'].astype(str)
            df = df.merge(eval_df[['id', GENERATED_MODERATION_COL]], on='id', how='left')
        custom_trainer.eval(df, GENERATED_MODERATION_COL, 'eval_generated')

    # evaluation (original)
    if custom_args.do_eval_original:
        custom_trainer.eval(df, ORIGINAL_MODERATION_COL, 'eval_original')

    if custom_args.answer_intro_questions:
        custom_trainer.eval_intro_questions()


if __name__ == "__main__":
    main()
