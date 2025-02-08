import json
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

import data
from args import CustomArguments
from data import GENERATED_MODERATION_COL
from eval.evaluator import EvaluationInstructionCreator
from moderation_type import ModerationType

logger = logging.getLogger(__name__)

USER_COMMENT_COL = 'comment parent content'


class CustomTrainer:
    def __init__(self, device, model, tokenizer, data_args, model_args, training_args, custom_args: CustomArguments):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.pipline = pipeline('text-generation', model=model, tokenizer=tokenizer)
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.custom_args = custom_args
        self.eval_instruction_creator = EvaluationInstructionCreator()
        self.moderator_persona = open('prompts/moderator_persona.txt').read()
        self.annotation_instructions = open('prompts/annotation_instructions.txt').read()
        self.annotation_form = open('prompts/annotation_form.txt').read()
        self.cots = {x.value: open(f'prompts/{x.value}_cot.txt').read() if custom_args.use_cot else ''
                     for x in ModerationType if x != ModerationType.SOCIAL_MODERATION}
        logger.info(f'system_instruction:\n{self.moderator_persona}')
        logger.info(f'user_instruction:\n{self.annotation_instructions}')

    def create_instructions(self, task_instruction, comment, moderation_type, topic):
        instruction = moderation_type.get_description()
        instructions = [instruction]
        if self.custom_args.use_annotation_instructions:
            instructions.append(self.annotation_instructions)
        if self.custom_args.use_cot:
            instructions.append(self.cots[moderation_type.value])
        instructions += [
            (f'Given the following user comment on the topic "{topic}", '
             if (self.custom_args.use_topic and topic) else '') +
            f'{task_instruction} {moderation_type.get_title()} of the following User Comment:\n\n']
        instruction = '\n\n'.join(instructions)
        user_instruction = f"{instruction} {comment}"
        return user_instruction

    def create_messages(self, comment, reply=None, moderation_type=ModerationType.GENERAL_MODERATION, topic=None):
        user_instruction = self.create_instructions(
            f'Generate a short moderator comment (around 60 words, maximum 80 words) as a reply that aims at',
            comment, moderation_type, topic)

        system_instruction = self.moderator_persona

        messages = self.create_system_user_messages(system_instruction, user_instruction)

        if reply:
            messages.append({"role": "assistant", "content": reply})

        logging.debug(f'system_instruction:\n{system_instruction}')
        logging.debug(f'user_instruction:\n{user_instruction}')
        logging.debug(f'input:\n{comment}')

        return messages

    def create_system_user_messages(self, system_instruction, user_instruction):
        if self.model_args.use_system_role:
            messages = [
                {"role": "user",
                 "content": user_instruction}
            ]
            if system_instruction:
                messages.insert(0, {"role": "system", "content": system_instruction})
        else:
            # add persona, cot
            # https://web.archive.org/web/20231030013339/https://docs.mistral.ai/usage/guardrailing/#appendix
            messages = [
                {"role": "user",
                 "content": (system_instruction or '') + "\n" + user_instruction}
            ]
        return messages

    def get_instruction_output(self, comment, moderation_type, topic):
        messages = self.create_messages(comment=comment, moderation_type=moderation_type, topic=topic)
        outputs = self.get_messages_output(messages,
                                           max_new_tokens=self.custom_args.max_new_tokens)
        return outputs, messages

    def get_messages_output(self, messages, max_new_tokens=256):
        if self.custom_args.use_pipeline:
            # if 'llama' in self.model_args.model_name_or_path:
            # based on https://huggingface.co/blog/llama3#how-to-prompt-llama-3
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            try:
                outputs = self.pipline(
                    messages,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators,
                    # use top-p sampling
                    do_sample=True,
                    num_return_sequences=self.custom_args.num_return_sequences,
                    # default sampling parameters (temperature and top_p) taken from the original meta codebase
                    temperature=0.6,
                    top_p=0.9,
                )
                outputs = [x["generated_text"][-1]["content"] for x in outputs]
            # mostly for local use as it does not resolve issues of GPU memory
            except Exception as e:
                logging.exception(f'Ignoring Exception and continuing: {e}')
                torch.cuda.empty_cache()
                outputs = None
        else:
            encodeds = self.tokenizer.apply_chat_template(
                messages,
                # no mention in https://docs.mistral.ai/platform/guardrailing/
                # add_generation_prompt=True,
                return_tensors="pt",
                # suppress warning
                # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id/71397707#71397707
                # pad_token_id = tokenizer.eos_token_id,
                # error: Asking to pad but the tokenizer does not have a padding token
                # padding='longest', truncation=True, max_length=1000
            )
            logging.debug(self.tokenizer.batch_decode(encodeds))
            model_inputs = encodeds.to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                                    num_return_sequences=self.custom_args.num_return_sequences)
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            logging.debug(decoded)
            # Delete the tensor
            del generated_ids
            # Free up GPU memory
            torch.cuda.empty_cache()
            # does not work for debug model
            outputs = [x.split('[/INST]')[-1] for x in decoded]
        logging.debug(f'output:\n{outputs[0] if outputs else None}')
        return outputs

    def get_eval_instruction(self, moderation_type, comment, reply, topic):
        return self.eval_instruction_creator.create_instruction(moderation_type, comment, reply, topic=topic)

    def predict(self, df, moderation_types):
        output_filename = 'moderated_comments_output'
        # Ensure the directory exists
        file_path = f'{self.training_args.output_dir}/{output_filename}.jsonl'

        # Open the file once
        with open(file_path, 'a') as file:
            for i, (index, row) in enumerate(tqdm(df.iterrows(), total=df.shape[0])):
                for moderation_type in moderation_types:
                    comment = row[USER_COMMENT_COL]

                    topic = row["type"] if 'type' in df.columns else None
                    outputs, messages = self.get_instruction_output(
                        comment=comment,
                        moderation_type=moderation_type,
                        topic=topic)
                    if outputs:
                        for k, x in enumerate(outputs):
                            df.at[index, f'{GENERATED_MODERATION_COL}' + ('' if k == 0 else f'_{k}')] = x

                    if i < 3:
                        logging.getLogger().setLevel(self.custom_args.logging_level.upper())

                    # Append to the file
                    append_dict_to_jsonl(file, {
                        'id': df.at[index, 'id'],
                        'comment': comment,
                        'topic': topic,
                        'moderation_type': moderation_type.value,
                        'messages': messages,
                        'outputs': outputs
                    })
                if i % self.custom_args.save_batch_size == 0:
                    # Save the updated dataframe
                    df[[data.ID] + [col for col in df.columns if col.startswith(GENERATED_MODERATION_COL)]].to_csv(
                        f"{self.training_args.output_dir}/{output_filename}.csv", index=False)
                    file.flush()
        # Save the updated dataframe
        df[[data.ID] + [col for col in df.columns if col.startswith(GENERATED_MODERATION_COL)]].to_csv(
            f"{self.training_args.output_dir}/{output_filename}.csv", index=False)

    def eval(self, df, col, output_filename):
        # extra cols to save based on the dataset
        extra_cols = ['user_id', 'index', 'score', 'dataset', 'batch'] if self.data_args.dataset == 'annotated' else []

        if self.custom_args.use_annotation_intro_answers_for_eval:
            intro_questions = json.loads(Path('data/annotated/intro_questions.json').read_text())
            intro_answers_df = pd.read_csv('data/annotated/intro_answers.csv')

        file_path = f'{self.training_args.output_dir}/{output_filename}.jsonl'
        # Open the file once
        with open(file_path, 'a') as file:
            for i, (index, row) in enumerate(tqdm(df.iterrows(), total=df.shape[0])):
                comment = row[USER_COMMENT_COL]
                reply = df.at[index, col]
                if pd.isna(reply) or pd.isna(comment):
                    continue
                for moderation_type in self.custom_args.moderation_types:
                    if not reply or not comment:
                        continue

                    if self.custom_args.use_annotation_intro_answers_for_eval:
                        system_instruction = ['You are a user who has those answers to the following questions:']
                        for k, question in intro_questions.items():
                            answer = intro_answers_df.loc[intro_answers_df['user_id'] == row['user_id'], k].values[0]
                            system_instruction.append(f'{question}\n{answer}')
                        system_instruction = '\n\n'.join(system_instruction)
                    else:
                        system_instruction = None

                    topic = df.at[index, 'type'] if 'type' in df.columns else None
                    if self.custom_args.use_gen_instructions_for_eval:
                        instruction = ['\n'.join([line for line in self.moderator_persona.strip().split('\n') if line.startswith('-')])]
                        instruction += [self.create_instructions(
                            f'Evaluate the Reply Comment as a reply that aims at',
                            comment, moderation_type, topic)]
                        instruction += ['Reply Comment:\n\n' + reply]
                        instruction += ['Evaluate the Reply Comment by choosing from '
                                        '(very poor, poor, acceptable, good, very good)']
                        instruction = '\n\n'.join(instruction)
                    elif self.custom_args.use_annotation_form_for_eval:
                        instruction = self.annotation_form.format(comment=comment, reply=reply)
                    elif self.custom_args.use_annotation_instructions:
                        instruction = [self.annotation_instructions]
                        if self.custom_args.use_topic and topic:
                            instruction += [f'Topic: {topic}']
                        instruction += ['*User Comment*:\n' + comment]
                        instruction += ['*Reply Comment*:\n' + reply]
                        instruction += ['Based on the *User Comment*, evaluate the *Reply Comment* by choosing from '
                                        '(very poor, poor, acceptable, good, very good)']
                        instruction = '\n\n'.join(instruction)
                    else:
                        system_instruction = 'You are a helpful assistant who replies with a number between 0 and 4'
                        instruction = self.eval_instruction_creator.create_instruction(
                            moderation_type, comment, reply, topic)

                    if self.custom_args.add_explanation:
                        instruction += f' Please provide an explanation for your evaluation.'

                    messages = self.create_system_user_messages(system_instruction=system_instruction,
                                                                user_instruction=instruction)
                    outputs = self.get_messages_output(
                        messages,
                        max_new_tokens=self.custom_args.eval_max_new_tokens)

                    if outputs:
                        for k, x in enumerate(outputs):
                            df.at[index, f'eval_{col}_{moderation_type}' + ('' if k == 0 else f'_{k}')] = x

                    # Append to the file
                    append_dict_to_jsonl(file, {
                        'id': df.at[index, 'id'],
                        'comment': comment,
                        'reply': reply,
                        'topic': topic,
                        'moderation_type': moderation_type.value,
                        'messages': messages,
                        'outputs': outputs
                    })

                if i % self.custom_args.save_batch_size == 0:
                    # Save
                    df[
                        ['id'] + extra_cols + [col for col in df.columns if col.startswith(f'eval_')]].to_csv(
                        f'{self.training_args.output_dir}/{output_filename}.csv', index=False)
                    file.flush()
        # Save
        df[
            ['id'] + extra_cols + [col for col in df.columns if col.startswith(f'eval_')]].to_csv(
            f'{self.training_args.output_dir}/{output_filename}.csv', index=False)

    def eval_intro_questions(self):
        output_filename = 'intro_answers'
        intro_questions = json.loads(Path('data/annotated/intro_questions.json').read_text())
        file_path = f'{self.training_args.output_dir}/{output_filename}.jsonl'
        # Open the file once
        with open(file_path, 'a') as file:
            for k, v in intro_questions.items():
                for i in range(5):
                    print(f'Question: {v}')

                    messages = self.create_system_user_messages(system_instruction=None,
                                                                user_instruction=v)
                    outputs = self.get_messages_output(
                        messages,
                        max_new_tokens=self.custom_args.eval_max_new_tokens)

                    # Append to the file
                    append_dict_to_jsonl(file, {
                        'question': v,
                        'messages': messages,
                        'outputs': outputs
                    })


def append_dict_to_jsonl(file, dictionary):
    json_line = json.dumps(dictionary)
    file.write(json_line + '\n')
