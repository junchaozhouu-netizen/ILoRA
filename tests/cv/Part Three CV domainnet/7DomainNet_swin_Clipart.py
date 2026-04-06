import subprocess
import time


def run_training_tasks(commands):
    """Run a batch of training commands and save the outputs to a log file.

    Args:
        commands (list): A list of Python command strings.
    """
    log_file = "training_batch_log.txt"

    with open(log_file, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"Batch execution started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_obj.write("=" * 50 + "\n\n")

    total_tasks = len(commands)

    for task_idx, cmd in enumerate(commands, 1):
        task_info = f"Task {task_idx}/{total_tasks}"
        print(f"\n{task_info}: running -> {cmd}")

        with open(log_file, "a", encoding="utf-8") as file_obj:
            file_obj.write(f"\n{task_info} - started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
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
                status = "SUCCESS"
                print(f"{task_info}: completed successfully.")
            else:
                status = "FAILED"
                print(f"{task_info}: failed with return code {result.returncode}.")

            with open(log_file, "a", encoding="utf-8") as file_obj:
                file_obj.write(f"{task_info} - status: {status}\n")
                file_obj.write(f"Return code: {result.returncode}\n")
                file_obj.write(f"Standard output:\n{result.stdout}\n")
                file_obj.write(f"Standard error:\n{result.stderr}\n")
                file_obj.write(f"{task_info} - finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                file_obj.write("=" * 50 + "\n")

        except Exception as exc:
            error_message = f"Exception occurred: {exc}"
            print(f"{task_info}: {error_message}")

            with open(log_file, "a", encoding="utf-8") as file_obj:
                file_obj.write(f"{task_info} - status: ERROR\n")
                file_obj.write(f"Error: {error_message}\n")
                file_obj.write(f"{task_info} - finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                file_obj.write("=" * 50 + "\n")

    final_message = f"\nAll {total_tasks} tasks finished. Log saved to: {log_file}"
    print(final_message)

    with open(log_file, "a", encoding="utf-8") as file_obj:
        file_obj.write(f"\n{final_message}\n")
        file_obj.write(f"Batch execution finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    training_commands = ['python main_fl_domainNet.py --cuda_device 6 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 50 --lr 0.001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FEDIT --dataset domainnet_clipart --save_dir outputs/DomainNet7/1/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.6 --heterogeneous_rank_clients "2,4,6"', 'python main_fl_domainNet.py --cuda_device 6 --lora_r_client 4 --lora_r_server 12 --model_name swin-base-patch4-window7-224 --num_rounds 50 --lr 0.001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FLORA --dataset domainnet_clipart --save_dir outputs/DomainNet7/2/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.6 --heterogeneous_rank_clients "2,4,6"', 'python main_fl_domainNet.py --cuda_device 6 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 50 --lr 0.001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm LoRA_FAIR --dataset domainnet_clipart --save_dir outputs/DomainNet7/3/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.6 --heterogeneous_rank_clients "2,4,6"', 'python main_fl_domainNet.py --cuda_device 6 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 50 --lr 0.001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm FFA-LORA --dataset domainnet_clipart --save_dir outputs/DomainNet7/4/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.6 --heterogeneous_rank_clients "2,4,6"', 'python main_fl_domainNet.py --cuda_device 6 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 50 --lr 0.001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset domainnet_clipart --save_dir outputs/DomainNet7/5/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.6 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python main_fl_domainNet.py --cuda_device 6 --lora_r_client 4 --lora_r_server 6 --model_name swin-base-patch4-window7-224 --num_rounds 50 --lr 0.001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset domainnet_clipart --save_dir outputs/DomainNet7/6/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control True --alpha 0.6 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1']
    run_training_tasks(training_commands)
