import html
import logging
from pathlib import Path

import ftfy
import pandas as pd
from tqdm import tqdm

from args import DataTrainingArguments
from moderation_type import ModerationType

REGROOM_PROCESSED_DATA_PATH = 'data/regulation_room'

THRESHOLD = 0.5

columns_mapping = {
    'Social Functions (Welcome, Ataboy, and thanks)? Yes/No': 'social',
    'Site Use Issues? Yes/No': 'site',
    'Organizing Discussion - direct to another post? Yes/No ': 'organizing',
    'Policing (Civility, relevance, wrong venue)? Yes/No': 'policing',
    'Out of bounds for agency? Yes/No': 'target',
    'Improving individual comment quality (user- specific)? Yes/No': 'quality',
    'Broadening discussion (between users or bring other users in)? Yes/No': 'broadening'
}
moderator_actions = list(columns_mapping.values())


def read_dataset_with_all_annotations(path: str):
    path = Path(path) / 'regulationroom'
    # read annotations
    df1 = pd.read_excel(path / 'original_data/08_03_2011/EOBR Comment Quality Coding All 8_3_11.xls')
    print(f"number of comments in EOBR {df1['comment ID'].nunique()}")
    df2 = pd.read_excel(path / 'original_data/10_04_2011/APR Moderator Comments Combined 10_4_11.xlsx')
    print(f"number of comments in APR {df2['comment_ID'].nunique()}")

    # read comments
    comments_df1 = pd.read_excel(path / 'Comment_Data_from_CeRI_4_3_2017.xlsx',
                                 sheet_name='Electronic On-Board Recorders')
    comments_df2 = pd.read_excel(path / 'Comment_Data_from_CeRI_4_3_2017.xlsx',
                                 sheet_name='Airline Passenger Rights')

    # map comment ids to text
    df1['comment parent content'] = df1['comment parent'].map(comments_df1.set_index('COMMENT ID')['COMMENT'])
    df2['comment_parent_content'] = df2['comment_parent'].map(comments_df2.set_index('COMMENT ID')['COMMENT'])

    if [x for x in df1['comment ID'].to_list() if x in df2['comment_ID'].to_list()]:
        print('duplicates in IDs!!! use in combination with type!')

    # join datasets
    df2.rename(columns=lambda x: x.replace('_', ' '), inplace=True)
    df1['type'] = 'EOBR'
    df2['type'] = 'APR'
    df = pd.concat([df1, df2], ignore_index=True, sort=False)
    print(f"number of comments in merged dataset {df.groupby(['comment ID', 'type']).ngroups}")

    # aggregate annotations
    df.rename(columns=columns_mapping, inplace=True)
    df[moderator_actions] = df[moderator_actions].replace({'Yes': 1, 'No': 0})
    return df


def read_dataset_with_annotations(path: str):
    df = read_dataset_with_all_annotations(path)
    df = df.groupby(['comment ID', 'type']).agg(
        {**{x: 'first' for x in df.columns.to_list() if x not in moderator_actions},
         **{x: 'mean' for x in moderator_actions}})
    print(f"number of comments after aggregation {df.shape[0]}")

    for x in moderator_actions:
        df[x + ' label'] = df[x].map(lambda x: 1 if x > THRESHOLD else 0)

    return df


def read_dataset_complete(data_args: DataTrainingArguments):
    xls = pd.ExcelFile(Path(data_args.data_dir) / 'regulationroom/Comment_Data_from_CeRI_4_3_2017.xlsx')
    dfs = [xls.parse(x).assign(type=x) for x in xls.sheet_names[1:]]
    for df in dfs:
        # one row has nan for user
        df.dropna(subset=['USER LOGIN'], inplace=True)
        tqdm.pandas()

        df['MODERATOR'] = df['USER LOGIN'].map(lambda x: 'moderator' in x.lower()
                                                         # with 'MODERATOR ACTION' = reply
                                                         or x in ['Nathan Koskella', 'CJ Kim', 'jdb386',
                                                                  'Krystal Anderson', 'mjn3', 'Andres Castillo'])
        df['MODERATED'] = df['COMMENT ID'].map(lambda x: df[df['COMMENT PARENT'] == x]['MODERATOR'].any())

        # add comment parents to the dataset
        df['COMMENT PARENT 1'] = df['COMMENT PARENT']
        for i in range(1, data_args.comment_parents_num + 1):
            df[f'COMMENT PARENT {i} CONTENT'] = df[f'COMMENT PARENT {i}'].map(df.set_index('COMMENT ID')['COMMENT'])
            df[f'COMMENT PARENT {i} CONTENT'].fillna('', inplace=True)
            df[f'COMMENT PARENT {i} USER LOGIN'] = df[f'COMMENT PARENT {i}'].map(
                df.set_index('COMMENT ID')['USER LOGIN'])
            df[f'COMMENT PARENT {i + 1}'] = df[f'COMMENT PARENT {i}'].map(df.set_index('COMMENT ID')['COMMENT PARENT'])

    df = pd.concat(dfs, ignore_index=True, sort=False)
    # drop the many Unnamed columns with all nan values
    df.dropna(axis=1, how='all', inplace=True)

    return df


def read_dataset_complete_with_annotations(data_args: DataTrainingArguments):
    complete_df = read_dataset_complete(data_args)
    annotations_df = read_dataset_with_annotations(data_args.data_dir)
    annotations_df = (annotations_df.reset_index(drop=True)
                      .drop(columns=['coder'])
                      .rename(columns={'comment ID': 'COMMENT ID'}))

    annotations_df['type'] = annotations_df['type'].map(
        {'EOBR': 'Electronic On-Board Recorders', 'APR': 'Airline Passenger Rights'})
    df = pd.merge(complete_df, annotations_df, how='outer', on=['COMMENT ID', 'type'], indicator=True)

    mask = df['_merge'] == 'right_only'
    # loc doesn't work with multiple cols
    df.loc[mask, 'COMMENT'] = df.loc[mask, 'comment content']
    df.loc[mask, 'COMMENT PARENT 1 CONTENT'] = df.loc[mask, 'comment parent content']

    return df


ID = 'id'
USER_COMMENT_COL = 'comment parent content'
ORIGINAL_MODERATION_COL = 'comment content'
GENERATED_MODERATION_COL = 'generated content'


def read_dataset_complete_with_annotations_clean(data_args: DataTrainingArguments):
    df = read_dataset_complete_with_annotations(data_args)

    df.drop(columns=['DATE', 'DATE GMT',
                     # regulationroom original annotations.ipynb: same minor space differences
                     'comment parent content', 'comment content'
                     ], inplace=True)
    df.rename(columns={'COMMENT PARENT 1 CONTENT': 'comment parent content', 'COMMENT': 'comment content',
                       'COMMENT ID': 'comment id', 'MODERATOR': 'moderator'},
              inplace=True)

    df = df[~(df['comment content'].isna()) | (df['comment content'] == '')]
    df = df[~(df['comment parent content'].isna()) | (df['comment parent content'] == '')]

    df['comment content'] = df['comment content'].apply(html.unescape)
    df['comment content'] = df['comment content'].apply(ftfy.fix_text)
    # df['comment content'] = df['comment content'].str.replace('_x000D_', '\n')
    # df['comment content'] = df['comment content'].apply(lambda x: x.encode('windows-1252').decode('utf-8'))
    #
    df['comment parent content'] = df['comment parent content'].apply(html.unescape)
    df['comment parent content'] = df['comment parent content'].apply(ftfy.fix_text)
    # df['comment parent content'] = df['comment parent content'].str.replace('_x000D_', '\n')
    # df['comment parent content'] = df['comment parent content'].apply(lambda x: x.encode('windows-1252').decode('utf-8'))

    df.replace({'_x000D_': '\n', '\xa0': ' '}, regex=True, inplace=True)
    return df


TO_SAVE_COLS = [ID, 'comment id', 'type',
                USER_COMMENT_COL, ORIGINAL_MODERATION_COL, 'moderator', 'quality', 'broadening']


def filter_by_moderation_types(df, moderation_types):
    columns = [x for x in moderation_types]
    df = df[df[columns].gt(0.5).any(axis=1)]
    return df


def prep_data(data_args: DataTrainingArguments):
    path = Path(REGROOM_PROCESSED_DATA_PATH)

    path.mkdir(parents=True, exist_ok=True)
    df = read_dataset_complete_with_annotations_clean(data_args)

    # add id: comment id can be duplicated between types
    df = df.reset_index(drop=False).rename(columns={'index': ID})

    # use for annotation
    target_moderation_type_df = filter_by_moderation_types(df,
                                                           [ModerationType.BROADENING_MODERATION,
                                                            ModerationType.QUALITY_MODERATION])

    # use the subset of data that is annotated for moderation as test
    df['test'] = df['_merge'] != 'left_only'
    df[TO_SAVE_COLS + ['test']].to_csv(path / 'all_data.csv', index=False)

    target_moderation_type_df.sample(n=100, random_state=42)[TO_SAVE_COLS].to_csv(
        path / 'annotation_data.csv', index=False)
    df[~df['test'] & (df['moderator'] == 1)].sample(n=10, random_state=42)[TO_SAVE_COLS].to_csv(
        path/'pilot_annotation_data.csv', index=False)
    print(f"saved moderator comments of size ={df.shape[0]}")
    return df


def read_regroom_data(base_path: str = '.', data_file: str = 'all_data.csv'):
    path = Path(base_path) / REGROOM_PROCESSED_DATA_PATH
    csv_path = path / data_file
    df = pd.read_csv(csv_path)
    df = df[df['comment parent content'].notna()]
    return df


def read_usermod_data(data_args: DataTrainingArguments):
    data_path = Path(data_args.data_dir) / 'e-delib/data/usermoderation_dataset/usermoderation_aggregated.csv'
    df = pd.read_csv(data_path, sep='\t')
    return df


def cleanup(df):
    # remove any with urls
    print(f'size before removing urls {df.shape[0]}')
    print((df[['broadening', 'quality']] > 0.5).astype(int).value_counts())
    temp = df[~df['comment content'].str.contains('http', case=False)]
    print(f'size after removing urls {temp.shape[0]}')
    print((df[['broadening', 'quality']] > 0.5).astype(int).value_counts())
    return temp


def read_data(data_args: DataTrainingArguments):
    if data_args.dataset == 'regroom':
        df = read_regroom_data(base_path=data_args.data_dir)

        mod_comments = df[df['moderator'] == 1]['comment content'].unique()
        print(f"number of parent moderator comments {len(mod_comments)}")
        print(f"number of parent moderator comments before {df[(df['comment parent content'].isin(mod_comments))].shape[0]}")
        df = cleanup(df)

        df.loc[~(df['test']) & df['moderator'], 'split'] = 'train'
        nomod_df = df[((df['moderator'] == 0) & ~(df['comment parent content'].isin(mod_comments)))].sample(n=300, random_state=42)
        print(f"number of parent moderator comments after {nomod_df[(nomod_df['comment parent content'].isin(mod_comments))].shape[0]}")
        df.loc[nomod_df.index, 'split'] = 'test_nomod'
        df.loc[df['test'], 'split'] = 'test'
        print(df[['split', 'test', 'moderator']].value_counts())

        data_path = Path(data_args.data_dir) / REGROOM_PROCESSED_DATA_PATH
        df.to_csv(data_path/'all_data_splits.csv', index=False)
    elif data_args.dataset == 'usermod':
        df = read_usermod_data(data_args)
        df = df.rename(columns={'preceding_comment': USER_COMMENT_COL, 'reply': ORIGINAL_MODERATION_COL})
        df = df.sample(n=500, random_state=42)
        print(pd.cut(df['prob_moderation'], bins=[i / 10.0 for i in range(11)],
                     include_lowest=True, right=True).value_counts().sort_index())
    elif data_args.dataset == 'annotated':
        df = pd.read_csv(Path(data_args.data_dir) / 'data/annotated/all_annotated_samples.csv')
    else:
        raise ValueError(f"Unknown dataset {data_args.dataset}")
    df = df.astype({'id': 'str'})
    return df


def read_data_splits(data_args: DataTrainingArguments):
    if data_args.dataset == 'aggregated_annotated':
        test_df = pd.read_csv(Path(data_args.data_dir) / f'data/annotated/preference/aggregated_annotated_samples.csv')
    elif data_args.dataset:
        test_df = read_data(data_args)
    else:
        test_df = None

    if data_args.train_dataset == 'preference':
        train_df = pd.read_csv(Path(data_args.data_dir) / f'data/annotated/preference/preference_merged.csv')
    elif data_args.train_dataset == 'regroom':
        train_df = test_df[test_df['split'] == 'train'].copy()
        test_df = test_df[test_df['split'].str.startswith('test')].copy()
    else:
        train_df = None

    return train_df, test_df


if __name__ == '__main__':
    read_data(DataTrainingArguments(dataset='regroom', data_dir='.'))
    # prep_data(DataTrainingArguments())
