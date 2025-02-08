import os

from moderation_type import ModerationType


class EvaluationInstructionCreator:
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = dir_path + '/prompts/moderation_function_eval_full.txt'
        prompt_beginning = open(file.replace('full', 'beginning')).read()
        self.instruction_template = open(file).read().replace('{beginning}', prompt_beginning)

    def create_instruction(self, moderation_type: ModerationType, comment, reply, topic):
        return self.instruction_template.format(function=moderation_type.get_title(),
                                                description=moderation_type.get_description(),
                                                comment=comment,
                                                reply=reply,
                                                topic=f'\ntopic: {topic}\n' or '')
