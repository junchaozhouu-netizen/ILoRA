import subprocess
import time


def run_training_tasks(commands):
    log_file = "training_batch_log.txt"
    with open(log_file, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"Batch execution started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_obj.write("=" * 50 + "\n\n")

    total_tasks = len(commands)
    for task_idx, cmd in enumerate(commands, 1):
        task_info = f"Task {task_idx}/{total_tasks}"
        print(f"\n{task_info}: running -> {cmd}")

        with open(log_file, "a", encoding="utf-8") as file_obj:
            file_obj.write(f"\n{task_info} - start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file_obj.write(f"Command: {cmd}\n")
            file_obj.write("-" * 30 + "\n")

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
                status = "success"
                print(f"{task_info}: completed successfully.")
            else:
                status = "failed"
                print(f"{task_info}: failed with return code {result.returncode}.")

            with open(log_file, "a", encoding="utf-8") as file_obj:
                file_obj.write(f"{task_info} - status: {status}\n")
                file_obj.write(f"Return code: {result.returncode}\n")
                file_obj.write(f"stdout:\n{result.stdout}\n")
                file_obj.write(f"stderr:\n{result.stderr}\n")
                file_obj.write(f"{task_info} - end time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                file_obj.write("=" * 50 + "\n")

        except Exception as exc:
            error_msg = f"Execution error: {exc}"
            print(f"{task_info}: {error_msg}")
            with open(log_file, "a", encoding="utf-8") as file_obj:
                file_obj.write(f"{task_info} - status: exception\n")
                file_obj.write(f"Error: {error_msg}\n")
                file_obj.write(f"{task_info} - end time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                file_obj.write("=" * 50 + "\n")

    final_msg = f"\nAll {total_tasks} tasks have finished. Log file: {log_file}"
    print(final_msg)
    with open(log_file, "a", encoding="utf-8") as file_obj:
        file_obj.write(f"\n{final_msg}\n")
        file_obj.write(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    training_commands = ['python Centralized_with_no_LORA.py --cuda_device 2 --data_dir ./data --model_name vit-base-patch16-224 --num_epochs 5 --lr 0.0001 --momentum 0.9 --weight_decay 0.0 --batch_size 128 --save_dir ./results/LORA+SAM5_cifar10/vit-base-patch16-224/adamw_result --dataset cifar10 --optimizer AdamW', 'python Centralized_with_no_LORA.py --cuda_device 2 --data_dir ./data --model_name vit-base-patch16-224 --num_epochs 5 --lr 0.0001 --momentum 0.9 --weight_decay 0.0 --batch_size 128 --save_dir ./results/LORA+SAM5_cifar10/vit-base-patch16-224/adamw_result --dataset cifar100 --optimizer AdamW', 'python Centralized_with_no_LORA.py --cuda_device 2 --data_dir ./data --model_name vit-base-patch16-224 --num_epochs 5 --lr 0.0001 --momentum 0.9 --weight_decay 0.0 --batch_size 128 --save_dir ./results/LORA+SAM5_cifar10/vit-base-patch16-224/adamw_result --dataset tiny-imagenet-200 --optimizer AdamW', 'python Centralized_with_no_LORA.py --cuda_device 3 --data_dir ./data --model_name vit-base-patch16-224 --num_epochs 5 --lr 0.01 --momentum 0.9 --weight_decay 0.0 --batch_size 128 --save_dir ./results/LORA+SAM5_cifar10/vit-base-patch16-224/sgd_result --dataset cifar10 --optimizer SGD', 'python Centralized_with_no_LORA.py --cuda_device 3 --data_dir ./data --model_name vit-base-patch16-224 --num_epochs 5 --lr 0.01 --momentum 0.9 --weight_decay 0.0 --batch_size 128 --save_dir ./results/LORA+SAM5_cifar10/vit-base-patch16-224/sgd_result --dataset cifar100 --optimizer SGD', 'python Centralized_with_no_LORA.py --cuda_device 3 --data_dir ./data --model_name vit-base-patch16-224 --num_epochs 5 --lr 0.01 --momentum 0.9 --weight_decay 0.0 --batch_size 128 --save_dir ./results/LORA+SAM5_cifar10/vit-base-patch16-224/sgd_result --dataset tiny-imagenet-200 --optimizer SGD']
    run_training_tasks(training_commands)
