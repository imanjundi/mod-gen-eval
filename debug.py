import os
import shutil
import sys
import baseline

# os.environ['WANDB_MODE'] = 'offline'
# os.environ['WANDB_MODE'] = 'disabled'

os.environ['DEBUG'] = 'True'

# sys.argv += ['--local', 'True']

shutil.rmtree('test', ignore_errors=True)
os.mkdir('test')


sys.argv += (
    # '--model_name_or_path bigscience/bigscience-small-testing'
    '--model_name_or_path sshleifer/tiny-gpt2'
    ' --sample_size 3'
    ' --max_new_tokens 10'
    ' --output_dir test'
    ' --overwrite_output_dir True'
    
    ' --moderation_type moderation'    
    # ' --moderation_type quality broadening'
    
    # ' --use_cot True'
    # ' --use_annotation_instructions True'

    # ' --dataset regroom'
    # ' --use_topic True'
    ' --dataset usermod'
    
    #  persona annotation eval
    # ' --dataset annotated'
    # ' --batch 1'
    # ' --do_eval_original True'
    # ' --use_gen_instructions_for_eval False'
    # ' --use_annotation_intro_answers_for_eval True'
    # ' --use_annotation_instructions True'
    # # does not work well
    # # ' --use_annotation_form_for_eval True'
    
    # train orpo
    # ' --do_train True'
    # # ' --training_method sft'
    # ' --training_method orpo'
    # # ' --do_predict True'
    
    # ' --add_explanation True'
    
    # eval aggregated_annotated_samples
    # ' --dataset aggregated_annotated'
    # ' --do_eval False'
    # ' --do_eval_original True'
    # ' --batch 1'
    
    # predict aggregated_annotated_samples
    ' --dataset aggregated_annotated'
    ' --do_predict True'
    ' --do_eval_original False'
    ' --num_return_sequences 1'
    
    ' --logging_level debug'
).split(' ')
baseline.main()
