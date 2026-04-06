import subprocess
import time

def run_training_tasks(commands):
    log_file = 'training_batch_log.txt'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write('=' * 50 + '\n\n')
    for (task_idx, cmd) in enumerate(commands, 1):
        total_tasks = len(commands)
        task_info = f'Task {task_idx}/{total_tasks}'
        print(f'\n{task_info}: start -> {cmd}')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{task_info} - start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f'Command: {cmd}\n')
            f.write('-' * 30 + '\n')
        try:
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=None)
            if result.returncode == 0:
                status = 'success'
                print(f'{task_info}: completed successfully')
            else:
                status = 'error'
                print(f'{task_info}: failed with return code {result.returncode}')
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f'{task_info} - status: {status}\n')
                f.write(f'Return code: {result.returncode}\n')
                f.write(f'Stdout:\n{result.stdout}\n')
                f.write(f'Stderr:\n{result.stderr}\n')
                f.write(f"{task_info} - end time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write('=' * 50 + '\n')
        except Exception as e:
            error_msg = f'Exception: {str(e)}'
            print(f'{task_info}: {error_msg}')
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f'{task_info} - exception\n')
                f.write(f'Error: {error_msg}\n')
                f.write(f"{task_info} - end time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write('=' * 50 + '\n')
    final_msg = f'\nFinished all {total_tasks} tasks. Log saved to: {log_file}'
    print(final_msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f'\n{final_msg}\n')
        f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
if __name__ == '__main__':
    training_commands = ['python main_fl_nlp_Ablation_Study_new2.py --algorithm FEDIT --optimizer AdamW --lr 0.0001 --use_control False --QR_init True --save_dir ./results/result0_Aggregation0.5/1/ --lora_r_client 4 --lora_r_server 16 --cuda_device 0 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.1 --dataset YAHOO_ANS --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new2.py --algorithm FEDIT --optimizer AdamW --lr 0.0001 --use_control False --QR_init True --save_dir ./results/result0_Aggregation0.5/2/ --lora_r_client 4 --lora_r_server 16 --cuda_device 0 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.1 --dataset QQP --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new2.py --algorithm FEDIT --optimizer AdamW --lr 0.0001 --use_control False --QR_init True --save_dir ./results/result0_Aggregation0.5/3/ --lora_r_client 4 --lora_r_server 16 --cuda_device 0 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.1 --dataset IMDB --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new2.py --algorithm FEDIT --optimizer AdamW --lr 0.0001 --use_control False --QR_init True --save_dir ./results/result0_Aggregation0.5/4/ --lora_r_client 4 --lora_r_server 16 --cuda_device 0 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.1 --dataset qnli --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new2.py --algorithm FEDIT --optimizer AdamW --lr 0.0001 --use_control False --QR_init True --save_dir ./results/result0_Aggregation0.5/5/ --lora_r_client 4 --lora_r_server 16 --cuda_device 0 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.1 --dataset sst2 --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new2.py --algorithm FEDIT --optimizer AdamW --lr 0.0001 --use_control False --QR_init True --save_dir ./results/result0_Aggregation0.5/6/ --lora_r_client 4 --lora_r_server 16 --cuda_device 0 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.1 --dataset AG_NEWS --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new2.py --algorithm FEDIT --optimizer AdamW --lr 0.0001 --use_control False --QR_init True --save_dir ./results/result0_Aggregation0.5/7/ --lora_r_client 4 --lora_r_server 16 --cuda_device 0 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.1 --dataset DBPEDIA14 --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"']
    run_training_tasks(training_commands)
