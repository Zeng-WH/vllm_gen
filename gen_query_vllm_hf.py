import json
import random
import tqdm
import argparse
import wandb
from huggingface_hub import HfApi
from huggingface_hub import HfApi, upload_file
#from some_llm_library import LLM, SamplingParams  # 请根据实际情况导入LLM和SamplingParams库
from vllm import LLM, SamplingParams


def upload_to_huggingface(output_file, repo_id, token, repo_type='dataset', commit_message="Upload output file"):
    # Upload the file to the specified repository
    upload_file(
        path_or_fileobj=output_file,
        path_in_repo=output_file,  # You can change the path in the repo if needed
        repo_id=repo_id,
        token=token,
        repo_type=repo_type,
        commit_message=commit_message
    )
    print(f"File {output_file} uploaded successfully to {repo_id}.")


def sample_data(system_prompt, example_list, n_shot):
    random.shuffle(example_list)
    user_input = system_prompt + "\n"
    for item in range(n_shot):
        user_input += "#Question#: " + example_list[item]["query"] + "\t\n"
    return user_input

def main(args):
    # wandb.init(project=args.wandb_project, config={
    #     "input_data": args.input_file,
    #     "model_dir": args.model_path,
    #     "sample_num": args.sample_num,
    #     "temperature": args.temperature,
    #     "top_k": args.top_k,
    #     "max_tokens": args.max_tokens,
    #     "output_file": args.output_file,
    #     "tensor_parallel_size": args.tensor_parallel_size,
    # })

    wandb.init(project=args.wandb_project, config={
        "input_data": args.input_file,
        "model_dir": args.model_path,
        "sample_num": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "output_file": args.output_file,
        "tensor_parallel_size": args.tensor_parallel_size,
        "n_shot": args.n_shot,
        "random_syn": args.random_syn,
    })
    with open(args.input_file, "r") as r:
        epoch_3 = json.load(r)
    
    if args.random_syn:
        trn_json = epoch_3
        print("random_syn")
    else:

        trn_json = [item for item in epoch_3 if item["output"].split("\n\n# Answer\n\n")[-1] != item["output0"].split("\n\n# Answer\n\n")[-1]]

    system_prompt = "Construct new example according to the following ones:"
    stop_tokens = ["\t\n#Question#"]
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, stop=stop_tokens)

    user_input = [sample_data(system_prompt, trn_json, args.n_shot) for _ in tqdm.tqdm(range(args.num_samples))]
    
    outputs = llm.generate(user_input, sampling_params)
    output_json = []
    for output in outputs:
        temp_output = output.outputs[0].text
        temp_cum_probs = output.outputs[0].cumulative_logprob
        temp_reason = output.outputs[0].finish_reason
        temp_json = {"gen_query": temp_output, "cum_logprobs": temp_cum_probs, "finish_reason": temp_reason}
        output_json.append(temp_json)

    with open(args.output_file, "w") as w:
        json.dump(output_json, w)
    # 上传文件到Hugging Face
    upload_to_huggingface(args.output_file, args.repo_id, args.hf_token)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data using LLM")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for sampling")
    parser.add_argument("--n_shot", type=int, default=3, help="Number of examples for few-shot learning")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of samples to generate")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--random_syn", action="store_true", help="Enable verbose logging")
    parser.add_argument("--wandb_project", type=str, required=True, help="Wandb project name")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID")  # Hugging Face存储库ID
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token")  # Hugging Face访问令牌
    args = parser.parse_args()
    main(args)
