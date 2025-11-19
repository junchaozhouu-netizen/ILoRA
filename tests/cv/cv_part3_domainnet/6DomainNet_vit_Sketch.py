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
    training_commands = ['python main_fl_domainNet.py --cuda_device 5 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 50 --lr 0.001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FEDIT --dataset domainnet_sketch --save_dir /home/pr3022206045/code/LORA+SAM5_cifar10/DomainNet6/1/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.6 --heterogeneous_rank_clients "2,4,6"', 'python main_fl_domainNet.py --cuda_device 5 --lora_r_client 4 --lora_r_server 12 --model_name vit-base-patch16-224 --num_rounds 50 --lr 0.001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FLORA --dataset domainnet_sketch --save_dir /home/pr3022206045/code/LORA+SAM5_cifar10/DomainNet6/2/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.6 --heterogeneous_rank_clients "2,4,6"', 'python main_fl_domainNet.py --cuda_device 5 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 50 --lr 0.001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm LoRA_FAIR --dataset domainnet_sketch --save_dir /home/pr3022206045/code/LORA+SAM5_cifar10/DomainNet6/3/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.6 --heterogeneous_rank_clients "2,4,6"', 'python main_fl_domainNet.py --cuda_device 5 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 50 --lr 0.001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FFA-LORA --dataset domainnet_sketch --save_dir /home/pr3022206045/code/LORA+SAM5_cifar10/DomainNet6/4/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.6 --heterogeneous_rank_clients "2,4,6"', 'python main_fl_domainNet.py --cuda_device 5 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 50 --lr 0.001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset domainnet_sketch --save_dir /home/pr3022206045/code/LORA+SAM5_cifar10/DomainNet6/5/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.6 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python main_fl_domainNet.py --cuda_device 5 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 50 --lr 0.001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset domainnet_sketch --save_dir /home/pr3022206045/code/LORA+SAM5_cifar10/DomainNet6/6/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control True --alpha 0.6 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1']
    run_training_tasks(training_commands)