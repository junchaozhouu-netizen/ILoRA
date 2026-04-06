import subprocess
import time


def run_training_tasks(commands):
    log_file = "training_batch_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Batch execution started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

    total_tasks = len(commands)
    for task_idx, cmd in enumerate(commands, 1):
        task_info = f"Task {task_idx}/{total_tasks}"
        print(f"\n{task_info}: running -> {cmd}")

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
                status = "success"
                print(f"{task_info}: completed successfully.")
            else:
                status = "failed"
                print(f"{task_info}: failed with return code {result.returncode}.")

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{task_info} - status: {status}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"stdout:\n{result.stdout}\n")
                f.write(f"stderr:\n{result.stderr}\n")
                f.write(f"{task_info} - end time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")

        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print(f"{task_info}: {error_msg}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{task_info} - status: exception\n")
                f.write(f"Error: {error_msg}\n")
                f.write(f"{task_info} - end time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")

    final_msg = f"\nAll {total_tasks} tasks have finished. Log file: {log_file}"
    print(final_msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{final_msg}\n")
        f.write(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    training_commands = ['python demo5.py --algorithm FEDIT --save_dir outputs/result1/7/ --lora_r_client 4 --lora_r_server 6 --cuda_device 1 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.6 --dataset YAHOO_ANS --model_name roberta-base --optimizer AdamW --lr 0.0001 --use_control False --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,4,6"', 'python demo5.py --algorithm FLORA --save_dir outputs/result1/8/ --lora_r_client 4 --lora_r_server 12 --cuda_device 1 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.6 --dataset YAHOO_ANS --model_name roberta-base --optimizer AdamW --lr 0.0001 --use_control False --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,4,6"', 'python demo5.py --algorithm LoRA_FAIR --save_dir outputs/result1/9/ --lora_r_client 4 --lora_r_server 6 --cuda_device 1 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.6 --dataset YAHOO_ANS --model_name roberta-base --optimizer AdamW --lr 0.0001 --use_control False --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,4,6"', 'python demo5.py --algorithm FFA-LORA --save_dir outputs/result1/10/ --lora_r_client 4 --lora_r_server 6 --cuda_device 1 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.6 --dataset YAHOO_ANS --model_name roberta-base --optimizer AdamW --lr 0.0001 --use_control False --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,4,6"', 'python demo5.py --algorithm ILORA --save_dir outputs/result1/11/ --lora_r_client 4 --lora_r_server 6 --cuda_device 1 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.6 --dataset YAHOO_ANS --model_name roberta-base --optimizer AdamW --lr 0.0001 --lora_scale_factor 1 --use_control False --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,4,6"', 'python demo5.py --algorithm ILORA --save_dir outputs/result1/12/ --lora_r_client 4 --lora_r_server 6 --cuda_device 1 --num_clients 3 --batch_size 64 --local_epochs 1 --num_rounds 5 --alpha 0.6 --dataset YAHOO_ANS --model_name roberta-base --optimizer AdamW --lr 0.0001 --lora_scale_factor 1 --use_control True --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,4,6"']
    run_training_tasks(training_commands)
