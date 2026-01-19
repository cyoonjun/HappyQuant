# Preserve then Quantize: Dominant-Subspace Guided Low-Rank Reconstruction

This is code for "Preserve then Quantize: Dominant-Subspace Guided Low-Rank Reconstruction".

## Env Setup

```bash
unzip HappyQuant.zip
cd HappyQuant
conda env create -f environment.yml
conda activate srr
pip install -r requirements.txt
pip install -e .
```

# How to Run SRR in Post-Training Quantization

## 1. Activate Conda Environment

```bash
conda activate srr
```

## 2. Run the PTQ Script

```bash
./experiments/ptq/run.sh
```

By default, this runs PTQ with SRR using `qera-exact` scaling on the LLaMA-2 7B model.

## 3. Optional Configurations

- **Select GPU:** Edit `export CUDA_VISIBLE_DEVICES=0` in `experiments/ptq/quantize_and_eval_iterate.sh` to choose the GPU ID.
- **Enable Zero-Shot Evaluation:** Remove `--disable-lm-eval` from the default settings.

## 4. Check Results

Results are saved automatically to the `./checkpoints` directory.



# How to Run SRR in Quantized Parameter-Efficient Fine-Tuning

## 1. Activate Conda Environment

```bash
conda activate srr
```

## 2. Run the QPEFT Script for task
Navigate to the specific task directory you want to run. For example, to run a GLUE task, use this command:
```bash
cd experiments/qpeft/glue
```

and then execute:
```bash
./srr.sh
```

By default, this runs QPEFT with SRR with 4bit MXINT quantization.


## 3. Optional Configurations

- **GLUE:** Adjust task_list in `srr.sh` and learning_rate_list in `adapt_and_glue_train.sh`
- **GSM8K & SlimPajama:** Select model and quant_bits in `srr.sh` and modify learning_rate_list in corresponding training script (`adapt_and_gsm8k_train.sh` or `adapt_and_clm_train.sh`)


## 4. Check Results

All scales are automatically saved in the `./storage` directory.
Experiment results are saved in the `./checkpoints` directory.

