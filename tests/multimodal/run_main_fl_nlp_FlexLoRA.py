import subprocess
import time


def run_training_tasks(commands):
    log_file = "training_batch_log.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

    total_tasks = len(commands)

    for task_idx, cmd in enumerate(commands, 1):
        task_info = f"Task {task_idx}/{total_tasks}"

        print(f"\n{task_info}: start -> {cmd}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{task_info} - Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
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
                print(f"{task_info}: completed")
            else:
                status = "error"
                print(f"{task_info}: failed with exit code {result.returncode}")

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{task_info} - Status: {status}\n")
                f.write(f"Exit code: {result.returncode}\n")
                f.write(f"Stdout:\n{result.stdout}\n")
                f.write(f"Stderr:\n{result.stderr}\n")
                f.write(f"{task_info} - Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")

        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print(f"{task_info}: {error_msg}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{task_info} - Error\n")
                f.write(f"Message: {error_msg}\n")
                f.write(f"{task_info} - Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")

    final_msg = f"\nAll tasks completed. Log saved to: {log_file}"
    print(final_msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(final_msg + "\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    training_commands = ['python main_fl_qwen25vl_multimodal_all_fix_numpy_indices.py --algorithm FlexLoRA --save_dir ./results/nlp_new3/1/ --cuda_device 3 --alpha 0.5 --dataset scienceqa --lora_r_client 4 --lora_r_server 6 --num_clients 3 --batch_size 1 --local_epochs 1 --num_rounds 5 --optimizer AdamW --lr 0.0001 --use_control False --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,4,6" --model_name \'Qwen2.5-VL-3B-Instruct\' --data_dir \'./multi_data\'', 'python main_fl_qwen25vl_multimodal_all_fix_numpy_indices.py --algorithm FlexLoRA --save_dir ./results/nlp_new3/2/ --cuda_device 3 --alpha 0.5 --dataset hateful_memes --lora_r_client 4 --lora_r_server 6 --num_clients 3 --batch_size 1 --local_epochs 1 --num_rounds 5 --optimizer AdamW --lr 0.0001 --use_control False --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,4,6" --model_name \'Qwen2.5-VL-3B-Instruct\' --data_dir \'./multi_data\'', 'python main_fl_qwen25vl_multimodal_all_fix_numpy_indices_MAMI.py --algorithm FlexLoRA --save_dir ./results/nlp_new3/3/ --cuda_device 3 --alpha 0.5 --dataset mami --lora_r_client 4 --lora_r_server 6 --num_clients 3 --batch_size 1 --local_epochs 1 --num_rounds 5 --optimizer AdamW --lr 0.0001 --use_control False --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,4,6" --model_name \'Qwen2.5-VL-3B-Instruct\' --data_dir \'./multi_data\'']
    run_training_tasks(training_commands)
