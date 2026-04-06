import subprocess
import time


def run_training_tasks(commands):
    log_file = "training_batch_log.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

    total_tasks = len(commands)

    for task_idx, cmd in enumerate(commands, 1):
        task_info = f"Task {task_idx}/{total_tasks}"

        print(f"\n{task_info}: starting -> {cmd}")
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
                print(f"{task_info}: completed successfully")
            else:
                status = "failed"
                print(f"{task_info}: failed with return code {result.returncode}")

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{task_info} - status: {status}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Stdout:\n{result.stdout}\n")
                f.write(f"Stderr:\n{result.stderr}\n")
                f.write(f"{task_info} - end time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")

        except Exception as e:
            error_msg = str(e)
            print(f"{task_info}: exception: {error_msg}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{task_info} - exception\n")
                f.write(f"Error: {error_msg}\n")
                f.write(f"{task_info} - end time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")

    final_msg = f"\nAll {total_tasks} tasks finished. Log saved to: {log_file}"
    print(final_msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{final_msg}\n")
        f.write(f"Finish time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    training_commands = ['python control_drift_on_CV.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar100 --save_dir ./results/result0_CV_cifar100/1/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.4 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python control_drift_on_CV.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar100 --save_dir ./results/result0_CV_cifar100/2/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.5 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python control_drift_on_CV.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar100 --save_dir ./results/result0_CV_cifar100/3/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.6 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python control_drift_on_CV.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar100 --save_dir ./results/result0_CV_cifar100/4/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control False --alpha 0.7 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python control_drift_on_CV.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar100 --save_dir ./results/result0_CV_cifar100/5/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control True --alpha 0.4 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python control_drift_on_CV.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar100 --save_dir ./results/result0_CV_cifar100/6/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control True --alpha 0.5 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python control_drift_on_CV.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar100 --save_dir ./results/result0_CV_cifar100/7/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control True --alpha 0.6 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1', 'python control_drift_on_CV.py --cuda_device 1 --lora_r_client 4 --lora_r_server 6 --model_name vit-base-patch16-224 --num_rounds 5 --lr 0.0001 --local_epochs 1 --batch_size 64 --num_clients 3 --algorithm ILORA --dataset cifar100 --save_dir ./results/result0_CV_cifar100/8/ --training_mode Homo --optimizer AdamW --distribution NON-IID --heterogeneous_rank True --use_control True --alpha 0.7 --heterogeneous_rank_clients "2,4,6" --lora_scale_factor 1']
    run_training_tasks(training_commands)
