# based on transformers/examples/pytorch/text-classification/run_glue.py
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from pprint import pprint
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments
from trl import SFTConfig, ORPOConfig

from moderation_type import ModerationType


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset to use: regroom, usermod"}
    )
    train_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset to use for training. mostly used for preference dataset"}
    )
    data_dir: Optional[str] = field(
        default=''
    )
    old_samples: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use old samples."
        },
    )
    sample_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "The number of samples to use for training and eval."
        },
    )
    in_domain: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use only in-domain data (has effect only on quality moderation)."
        },
    )
    comment_parents_num: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of comment parents to include."
        },
    )
    use_post_id: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use post id as context for the comment."
        },
    )
    split_rand_state: Optional[int] = field(
        default=42,
        metadata={
            "help": "The random state to split the data according to"
        },
    )
    batch: Optional[int] = field(
        default=None,
        metadata={
            "help": "The batch to use for eval of annotated data"
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    use_system_role: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use system role in chat template."
        },
    )



@dataclass
class CustomArguments:
    moderation_types: Optional[list[ModerationType]] = field(
        default=None,
        metadata={
            "help": "The type of moderation to model."
        }
    )
    use_topic: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use topic as context for the comment."
        },
    )
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum number of new tokens to generate."
        },
    )
    eval_max_new_tokens: Optional[int] = field(
        # default=15,
        default=256,
        metadata={
            "help": "The maximum number of new tokens to generate."
        },
    )
    num_return_sequences: Optional[int] = field(
        default=3,
        metadata={
            "help": "The number of sequences to generate."
        },
    )
    add_explanation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Add explanation to the generated eval"
        },
    )
    use_cot: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use chain of thoughts in prompt."
        },
    )
    use_annotation_instructions: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use detailed instructions from the annotation study."
        },
    )
    use_gen_instructions_for_eval: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use generation prompts for evaluation."
        },
    )
    # Full form works on GPT-4o but not with LLaMA-3 70B so asking only to answer the score question
    use_annotation_form_for_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use annotation form for evaluation."
        },
    )
    use_annotation_intro_answers_for_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use annotation intro answers for evaluation."
        },
    )
    answer_intro_questions: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use annotation intro answers for evaluation."
        },
    )
    training_method: Optional[str] = field(
        default=None,
        # default='sft',
        # default='orpo',
        metadata={
            "help": "The training method to use: sft, orpo"
        },
    )
    load_peft_model: Optional[bool] = field(
        default=False,
        metadata={
            "help": "The model id is for a PEFT model."
        },
    )
    use_pipeline: Optional[bool] = field(
        default=True
    )
    do_eval_original: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Eval on original data."
        },
    )
    output_dir_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "Output path prefix"}
    )
    eval_generated_input_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Input path of the data to evaluate"}
    )
    save_batch_size: Optional[int] = field(
        default=10
    )
    project_name: Optional[str] = field(
        default=None,
        metadata={"help": "Project name in wandb"}
    )

    logging_level: Optional[str] = field(
        # default='info',
        default='debug',
        metadata={"help": "Logging level"}
    )

    def __post_init__(self):
        self.moderation_types = [ModerationType(x) for x in self.moderation_types]



def parse_arguments() -> tuple[ModelArguments, DataTrainingArguments, SFTConfig, CustomArguments, bool]:
    if debug := 'DEBUG' in os.environ:
        print('running in debug mode')
        sys.argv += ['--data_dir', str(Path.home() / 'data')]
    if 'orpo' in sys.argv:
        config_class = ORPOConfig
    elif 'sft' in sys.argv:
        config_class = SFTConfig
    else:
        config_class = TrainingArguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, config_class, CustomArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, custom_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    if data_args.dataset == 'usermod':
        print('adjusting args for usermod')
        training_args.moderation_types = [ModerationType.GENERAL_MODERATION]
        training_args.use_topic = False
        data_args.data_dir = '/mount/projekte'

    if custom_args.training_method in ['orpo']:
        data_args.train_dataset = 'preference'

    for x in (model_args, data_args, training_args, debug):
        pprint(x)
    save_args(data_args, model_args, training_args, custom_args)

    # args compatibility checks
    assert (custom_args.eval_generated_input_dir is not None) + training_args.do_predict + custom_args.use_annotation_intro_answers_for_eval in (0, 1), \
        ('only one of args can be used, no combo of: '
         'custom_args.eval_generated_input_dir, training_args.do_predict, custom_args.use_annotation_intro_answers_for_eval')
    assert not (data_args.dataset == 'annotated' and training_args.do_eval), 'only do_eval_original is supported for annotated dataset'

    return model_args, data_args, training_args, custom_args, debug


def save_args(data_args, model_args, training_args, custom_args):
    with open(f"{training_args.output_dir}/config.json", "w") as f:
        all_configs = {name: asdict(x) for name, x in
                       zip(["model_args", "data_args", "training_args", "custom_args"],
                           [model_args, data_args, training_args, custom_args])}
        json.dump(all_configs, f, indent=2)
