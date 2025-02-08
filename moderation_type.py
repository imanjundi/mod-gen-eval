from transformers.utils import ExplicitEnum

MODERATOR_ACTION_DESCRIPTION = {'Social Functions':
                                    ['Welcoming',
                                     'Encouragement; appreciation of comment',
                                     'Thanking users for participating'],
                                'Resolving Site Use Issues':
                                    ['Resolving technical difficulties',
                                     'Providing information about the goals/rules of moderation',
                                     'Providing information about role of CeRI'],
                                'Organizing Discussion':
                                    ['Directing user to another issue post more relevant to his/her expressed interest'],
                                'Policing': [
                                    'Redact and quarantine for inappropriate language or content',
                                    'Maintaining/encouraging civil deliberative discourse'],
                                'Keeping Discussion on Target': [
                                    'Explaining why comment is beyond agency authority or competence or outside scope of current rule',
                                    'Indicating irrelevant, off point comments'],
                                'Improving Comment Quality':
                                    ['Providing substantive information about the proposed rule',
                                     'Correcting misstatements or clarifying what the agency is looking for',
                                     # 'Pointing to relevant information in primary documents or other data',
                                     'Pointing out characteristics of effective commenting',
                                     'Asking users to provide more information, factual details, or data to support their statements',
                                     'Asking users to make or consider possible solutions/alternative approaches'],
                                'Broadening Discussion':
                                    ['Encouraging users to consider and engage comments of other users',
                                     'Posing a question to the community at large that encourages other users to respond']}


class ModerationType(ExplicitEnum):
    GENERAL_MODERATION = "moderation"
    SOCIAL_MODERATION = "social"
    BROADENING_MODERATION = "broadening"
    QUALITY_MODERATION = "quality"

    def get_title(self):
        return {
            'moderation': 'Moderation',
            'social': 'Social Functions',
            'quality': 'Improving Comment Quality',
            'broadening': 'Broadening Discussion'}[self.value]

    def get_subfunctions(self):
        if self == ModerationType.GENERAL_MODERATION:
            return ', '.join([x.get_subfunctions() for x in
                              [ModerationType.SOCIAL_MODERATION, ModerationType.BROADENING_MODERATION,
                               ModerationType.QUALITY_MODERATION]])
        return ', '.join(MODERATOR_ACTION_DESCRIPTION[self.get_title()])

    def get_description(self):
        return (f'{self.get_title()} contains: ' +
                self.get_subfunctions()) + '.'
