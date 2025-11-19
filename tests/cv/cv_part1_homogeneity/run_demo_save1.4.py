import subprocess
import time

def run_training_tasks(commands):
    """
    ，（/、/）

    Args:
        commands (list): ，python
    """
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
    training_commands = ['python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FEDIT --dataset cifar10 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/1/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 12 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FLORA --dataset cifar10 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/2/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm LoRA_FAIR --dataset cifar10 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/3/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FFA-LORA --dataset cifar10 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/4/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar10 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/5/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3 --lora_scale_factor 1', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar10 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/6/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control True --alpha 0.3 --lora_scale_factor 1', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FEDIT --dataset cifar100 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/7/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 12 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FLORA --dataset cifar100 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/8/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm LoRA_FAIR --dataset cifar100 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/9/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FFA-LORA --dataset cifar100 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/10/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar100 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/11/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3 --lora_scale_factor 1', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar100 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/12/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control True --alpha 0.3 --lora_scale_factor 1', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FEDIT --dataset tiny-imagenet-200 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/13/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 12 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FLORA --dataset tiny-imagenet-200 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/14/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm LoRA_FAIR --dataset tiny-imagenet-200 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/15/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FFA-LORA --dataset tiny-imagenet-200 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/16/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset tiny-imagenet-200 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/17/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control False --alpha 0.3 --lora_scale_factor 1', 'python main_fl.py --cuda_device 3 --lora_r_client 4 --lora_r_server 4 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset tiny-imagenet-200 --save_dir /data/zjc/LORA+SAM5_cifar10/result14/18/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank False --use_control True --alpha 0.3 --lora_scale_factor 1']
    run_training_tasks(training_commands)