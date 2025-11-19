import subprocess
import time

def run_training_tasks(commands):
    log_file = 'training_batch_log.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f" - ：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write('=' * 50 + '\n\n')
    for (task_idx, cmd) in enumerate(commands, 1):
        total_tasks = len(commands)
        task_info = f' {task_idx}/{total_tasks}'
        print(f'\n{task_info}： -> {cmd}')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{task_info} - ：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f'：{cmd}\n')
            f.write('-' * 30 + '\n')
        try:
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=None)
            if result.returncode == 0:
                status = ''
                print(f'{task_info}：！')
            else:
                status = ''
                print(f'{task_info}：！：{result.returncode}，')
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f'{task_info} - ：{status}\n')
                f.write(f'：{result.returncode}\n')
                f.write(f'（stdout）：\n{result.stdout}\n')
                f.write(f'（stderr）：\n{result.stderr}\n')
                f.write(f"{task_info} - ：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write('=' * 50 + '\n')
        except Exception as e:
            error_msg = f'：{str(e)}'
            print(f'{task_info}：{error_msg}')
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f'{task_info} - ：\n')
                f.write(f'：{error_msg}\n')
                f.write(f"{task_info} - ：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write('=' * 50 + '\n')
    final_msg = f'\n {total_tasks} ！：{log_file}'
    print(final_msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f'\n{final_msg}\n')
        f.write(f"：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
if __name__ == '__main__':
    training_commands = ['python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir /results/result0_Aggregation0.5moreclient/8/ --lora_r_client 4 --lora_r_server 16 --cuda_device 7 --num_clients 15     --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.5 --dataset YAHOO_ANS --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir /results/result0_Aggregation0.5moreclient/9/ --lora_r_client 4 --lora_r_server 16 --cuda_device 7 --num_clients 15     --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.5 --dataset QQP --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir /results/result0_Aggregation0.5moreclient/10/ --lora_r_client 4 --lora_r_server 16 --cuda_device 7 --num_clients 15     --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.5 --dataset IMDB --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir /results/result0_Aggregation0.5moreclient/11/ --lora_r_client 4 --lora_r_server 16 --cuda_device 7 --num_clients 15     --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.5 --dataset qnli --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir /results/result0_Aggregation0.5moreclient/12/ --lora_r_client 4 --lora_r_server 16 --cuda_device 7 --num_clients 15     --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.5 --dataset sst2 --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir /results/result0_Aggregation0.5moreclient/13/ --lora_r_client 4 --lora_r_server 16 --cuda_device 7 --num_clients 15     --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.5 --dataset AG_NEWS --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir /results/result0_Aggregation0.5moreclient/14/ --lora_r_client 4 --lora_r_server 16 --cuda_device 7 --num_clients 15     --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.5 --dataset DBPEDIA14 --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"']
    run_training_tasks(training_commands)