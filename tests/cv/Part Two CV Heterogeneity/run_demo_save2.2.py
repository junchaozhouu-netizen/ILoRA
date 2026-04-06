import subprocess
import time


def run_training_tasks(commands):
    """
    Run a batch of training commands sequentially and save logs.

    Args:
        commands (list[str]): A list of Python command strings to execute.
    """
    log_file = "training_batch_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Batch training log - start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

    total_tasks = len(commands)

    for task_idx, cmd in enumerate(commands, 1):
        task_info = f"Task {task_idx}/{total_tasks}"
        print(f"\n{task_info}: Running command -> {cmd}")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{task_info} - start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command: {cmd}\n")
            f.write("-" * 30 + "\n")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=None,
            )

            if result.returncode == 0:
                status = "Success"
                print(f"{task_info}: Completed successfully.")
            else:
                status = "Failed"
                print(f"{task_info}: Execution failed with return code {result.returncode}.")

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{task_info} - status: {status}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Standard output:\n{result.stdout}\n")
                f.write(f"Standard error:\n{result.stderr}\n")
                f.write(f"{task_info} - end time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")

        except Exception as e:
            error_msg = f"Exception occurred: {str(e)}"
            print(f"{task_info}: {error_msg}")

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{task_info} - status: Exception\n")
                f.write(f"Error message: {error_msg}\n")
                f.write(f"{task_info} - end time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")

    final_msg = f"\nAll {total_tasks} tasks have finished. Log file: {log_file}"
    print(final_msg)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{final_msg}\n")
        f.write(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    training_commands = ['python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FEDIT --dataset cifar10 --save_dir outputs/result8/1/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6"', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 12 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FLORA --dataset cifar10 --save_dir outputs/result8/2/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6"', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm LoRA_FAIR --dataset cifar10 --save_dir outputs/result8/3/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6"', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FFA-LORA --dataset cifar10 --save_dir outputs/result8/4/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6"', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar10 --save_dir outputs/result8/5/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar10 --save_dir outputs/result8/6/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control True --alpha 0.3 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FEDIT --dataset cifar100 --save_dir outputs/result8/7/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6"', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 12 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FLORA --dataset cifar100 --save_dir outputs/result8/8/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6"', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm LoRA_FAIR --dataset cifar100 --save_dir outputs/result8/9/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6"', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FFA-LORA --dataset cifar100 --save_dir outputs/result8/10/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6"', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar100 --save_dir outputs/result8/11/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar100 --save_dir outputs/result8/12/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control True --alpha 0.3 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FEDIT --dataset tiny-imagenet-200 --save_dir outputs/result8/13/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6"', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 12 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FLORA --dataset tiny-imagenet-200 --save_dir outputs/result8/14/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6"', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm LoRA_FAIR --dataset tiny-imagenet-200 --save_dir outputs/result8/15/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6"', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FFA-LORA --dataset tiny-imagenet-200 --save_dir outputs/result8/16/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6"', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset tiny-imagenet-200 --save_dir outputs/result8/17/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.3 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python main_fl.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 5 --lr 0.01 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset tiny-imagenet-200 --save_dir outputs/result8/18/ --training_mode Homo --optimizer SGD --distribution NON-IID --heterogeneous_rank True --use_control True --alpha 0.3 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1']
    run_training_tasks(training_commands)
