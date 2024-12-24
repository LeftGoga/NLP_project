# [Language Model Fine-Tuning and Benchmarking Project](https://drive.google.com/file/d/1cEO3HFDxu_s8uLvRDB8TiyOy_dUGWqyi/view?usp=sharing)

![Image 1](./.images/TinyLlama_logo.png) 


This project provides tools and scripts for fine-tuning [TinyLlama-1.1B](https://github.com/jzhang38/TinyLlama) with sitchback technology, evaluating its performance, and analyzing benchmarks. In addition, there is a script for building a custom dataset, as well as logs of fine-tuning and evaluating with an analysis of statistical significance of training acceleration and metrics on benchmarks.

Our experimental results and the research paper can be found at the following link: [SwichBack Tiny Llama](https://drive.google.com/file/d/1cEO3HFDxu_s8uLvRDB8TiyOy_dUGWqyi/view?usp=sharing)

## Main results of acceleration
| Model              | OASST1      | Longform        | Custom RU      | 
|--------------------|-------------|-----------------|----------------|
| nn.Linear          | 72.57 min   | 173.19 min      | 175.46 min     | 
| SwichBackLinear    | 65.50 min   | 155.73 min      | 158.85 min     |
| Acceleration (%)   | 9.76%       | 10.08%          | 9.46%          |


## Dockerfile
To reproduce all results, you can use a Docker image setup to ensure all dependencies for fine-tuning and benchmarking workflows are installed. The base image includes `Python`, `CUDA`, `Triton`, `transformers`, `lm_eval`, and `torch` dependencies.

```bash
docker pull dmitryredkosk/bitsandbytes_transformer:cuda12.5
```

## Benchmarking

In the benchmarking section, you can find the reproducibility of results from the original paper, our results, results from GitHub issues, as well as the script from the official GitHub repository [bitsandbytes-repo](https://github.com/bitsandbytes-foundation/bitsandbytes).

## Fine-tune quick start

This script `sft/script.sh` performs fine-tuning and evaluation iterations for a language model. It runs the fine-tuning process using `sft/script.py`, and evaluates the model on specified tasks.

### General Parameters
 - `BS`: Batch size for training. Default value is 64.

 - `USE_SWICHBACK`: Boolean flag indicating whether to use switchback mode. Default value is false.

 - `CUDA_VISIBLE_DEVICES`: Specifies which GPU devices to use. Default value is 0.

 - `NUM_FTUNES`: Number of fine-tuning iterations to perform. Default value is 1.

 - `MODEL_NAME_OR_PATH`: Path to the pre-trained model. Default value is ../models/TinyLlama-1.1B-intermediate-step-240k-503b.

 - `OUTPUT_MODEL_NAME`: Name of the output model. Default value is 503b.

 - `OUTPUT_DIR`: Directory where fine-tuned models will be saved. Constructed as ../models_iter_ft_swichback_${USE_SWICHBACK}.

 - `LOG_DIR`: Directory for logging fine-tuning and evaluation outputs. Constructed as ./logs_use_switchback_${USE_SWICHBACK}.

 - `EVAL_TASKS`: Comma-separated list of evaluation tasks. Default value is `hellaswag,boolq,swag,winogrande,xwinograd_en`.

 - `EVAL_OUTPUT_DIR`: Directory for evaluation results. Constructed as ./lmeval_res_use_switchback_${USE_SWICHBACK}.

 - `FT_DATASET: Dataset to use for fine-tuning. Default value is oasst1.

### Directories Creation
The script creates the following directories if they do not exist:

- `${LOG_DIR}/ft`: For fine-tuning logs.

- `${LOG_DIR}/eval`: For evaluation logs.

- `${EVAL_OUTPUT_DIR}`: For evaluation results.

### Quick Usage

- Ensure that all dependences are installed and paths are correct in `script.sh`. To execute the script, run:

    ```bash
    bash script.sh
    ```

## switchback_layer

In the `./switchback_layer`, you can find the original `SwitchBackLinear` layer and all additional kernel functions from bitsandbytes, which are used for fine-tuning.

## Statistical significance

To calculate the significance of training speedup, use ./stats/stats.py and provide paths to the logs of nn.Linear and SwitchBackLinear. You also need to specify the significance level as the alpha parameter. 

Example usage:

```bash
python stats.py --file1 "Linear_logs_path" --file2 "SwichBackLinear_logs_path" --output_dir "path_for_results" --alpha 0.05
```



## Hardware and Software Specifications
### Hardware Specifications:

- `GPU Model`: NVIDIA A40
- `Number of GPUs`: 1

### Software Environment:

- `CUDA Version`: 12.0

- `NVIDIA Driver Version`: 525.147.05

- `Operating System`: Ubuntu 22.04

