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
    training_commands = ['python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir ./results/result0_Aggregation0.7/8/ --lora_r_client 4 --lora_r_server 16 --cuda_device 5 --num_clients 20    --batch_size 64 --local_epochs 1 --num_rounds 20 --alpha 1.6 --dataset YAHOO_ANS --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir ./results/result0_Aggregation0.7/9/ --lora_r_client 4 --lora_r_server 16 --cuda_device 5 --num_clients 20    --batch_size 64 --local_epochs 1 --num_rounds 20 --alpha 1.6 --dataset QQP --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir ./results/result0_Aggregation0.7/10/ --lora_r_client 4 --lora_r_server 16 --cuda_device 5 --num_clients 20    --batch_size 64 --local_epochs 1 --num_rounds 20 --alpha 1.6 --dataset IMDB --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir ./results/result0_Aggregation0.7/11/ --lora_r_client 4 --lora_r_server 16 --cuda_device 5 --num_clients 20    --batch_size 64 --local_epochs 1 --num_rounds 20 --alpha 1.6 --dataset qnli --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir ./results/result0_Aggregation0.7/12/ --lora_r_client 4 --lora_r_server 16 --cuda_device 5 --num_clients 20    --batch_size 64 --local_epochs 1 --num_rounds 20 --alpha 1.6 --dataset sst2 --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir ./results/result0_Aggregation0.7/13/ --lora_r_client 4 --lora_r_server 16 --cuda_device 5 --num_clients 20    --batch_size 64 --local_epochs 1 --num_rounds 20 --alpha 1.6 --dataset AG_NEWS --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"', 'python main_fl_nlp_Ablation_Study_new.py --algorithm ILORA --optimizer AdamW --lr 0.0001 --use_control False --QR_init Auto --save_dir ./results/result0_Aggregation0.7/14/ --lora_r_client 4 --lora_r_server 16 --cuda_device 5 --num_clients 20    --batch_size 64 --local_epochs 1 --num_rounds 20 --alpha 1.6 --dataset DBPEDIA14 --model_name roberta-base --lora_scale_factor 1 --distribution NON-IID --heterogeneous_rank True --training_mode Homo --heterogeneous_rank_clients "2,8,16"']
    run_training_tasks(training_commands)
