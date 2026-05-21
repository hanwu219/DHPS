# graph_v2_charls_ipf.py Usage Notes

`graph_v2_charls_ipf.py` is the main CHARLS data synthesis experiment script. It uses a 200-record real-data sample as the source of conditional constraints and an IPF-generated demographic base as the population to complete. Through LangGraph, it connects structure discovery, target distribution estimation, profile grouping, LLM planning, LLM validation and generation, and second-fill steps, finally writing the synthetic data CSV.

## Inputs and Outputs

The script currently targets remote server paths and does not read local `D:\...` paths by default.

Input files:

```text
/root/v2/CHARLS_processed_2020.csv
/root/v2/CHARLS_ipf.csv
```

Output files:

```text
/root/v2/synthetic_data.csv
/root/v2/second_fill_log.txt
```

`synthetic_data.csv` will be overwritten. `second_fill_log.txt` records the rules and row indices used in the second-fill stage.

## Data Fields

Demographic features D:

```python
["age_bin", "gender", "income_bin", "family_size", "marry", "edu", "health_status"]
```

High-level features H:

```python
["hospital", "exercise", "ins", "satlife", "social_need"]
```

Real sample processing workflow:

1. Read `CHARLS_processed_2020.csv`.
2. Delete `row_id` and `iwy`.
3. Keep samples with `income_total >= 0`.
4. Bin `age` into `age_bin`: `60-`, `60-64`, `65-69`, `70-74`, `75-79`, `80+`.
5. Bin `income_total` into tertiles as `income_bin`: `Low`, `Medium`, `High`.
6. Delete the original continuous columns `age` and `income_total`.
7. Keep `family_size <= 10`.
8. Drop missing values.
9. Use `random_state=24` to sample 200 records as the training/constraint sample.

IPF base processing workflow:

1. Read `CHARLS_ipf.csv`.
2. Keep the D columns.
3. Add H columns and initialize them as missing values.
4. Complete the H columns step by step in the subsequent workflow.

## Workflow

LangGraph node order:

```text
initialize_state
  -> llm_proposal
  -> bootstrap_target
  -> profiling
  -> planner
  -> generator
  -> planner loop
  -> second_fill
  -> END
```

### 1. initialize_state

Initializes the data, field lists, and graph state.

### 2. llm_proposal

Calls `deepseek-reasoner` concurrently for 10 runs, asking the LLM to propose potential D-H and H-H dependency edges.

Filtering rules:

```text
TOTAL_RUNS = 10
VOTE_THRESHOLD = 6
NMI_THRESHOLD = 0.02
```

Each candidate edge is also evaluated with NMI on the 200 real samples. Only edges that meet the threshold and receive enough votes enter the subsequent workflow.

### 3. bootstrap_target

Runs 1000 bootstrap iterations on the 200 real samples to estimate:

- The marginal distribution of each H feature.
- The joint distributions of validated dependency edges.

Then calls `deepseek-reasoner` to perform a reality check on low-probability joint combinations and generate `corrections` for use in second-fill.

### 4. profiling

Based on the validated D-H dependency edges, selects related D columns to group the IPF base and generate `profiles`.

If no related D columns are found, it falls back to grouping by all D columns.

### 5. planner

Dynamically scans the H-column gaps in the current synthetic data and selects the most urgent target:

```text
urgency = gap / available_slots
```

Each round plans at most:

```text
batch_size = min(max_gap, 300)
```

During planning, it:

- Checks H-H joint distribution conflicts.
- Checks D-H joint distribution conflicts.
- Calls `deepseek-reasoner` to allocate generation quotas among candidate profiles.

If there are no remaining gaps, it sets `finished=True` and enters `second_fill`.

### 6. generator

Calls `deepseek-chat` to semantically validate the planned tasks.

Concurrency limit:

```text
asyncio.Semaphore(10)
```

Tasks that pass validation are written to `df_large` in batches. Failure records are written to `failed_attempts` for reference by the next planner round.

### 7. second_fill

Uses the LLM-corrected low-probability reasonable combinations from `bootstrap_target` to second-fill rows that still contain missing values.

Final output:

```text
/root/v2/synthetic_data.csv
```

## Running

Make sure the current environment contains the required dependencies and that `.env` is configured with the API keys required by DeepSeek/LangChain.

Recommended execution in the `LLM` environment:

```bash
conda activate LLM
cd /root/v2
python graph_v2_charls_ipf.py
```

Or, on Windows, call the equivalent remote environment through conda:

```powershell
conda run -n LLM python graph_v2_charls_ipf.py
```

## Dependencies

Main dependencies include:

```text
python-dotenv
langgraph
langchain
langchain-core
langchain-community
langchain-openai
langchain-deepseek
pandas
numpy
scipy
scikit-learn
ipfn
aiohttp
aiofiles
faiss-cpu
```

If model or API-related errors occur at runtime, check first:

1. Whether `.env` is loaded correctly.
2. Whether the DeepSeek API key is available.
3. Whether `langchain_deepseek` is installed.
4. Whether the server can access the model API.

## Important Parameters

| Parameter | Current Value | Location/Meaning |
|---|---:|---|
| Small sample size | 200 | `df_real.sample(n=200, random_state=24)` |
| LLM structure proposal runs | 10 | `TOTAL_RUNS` |
| LLM vote threshold | 6 | `VOTE_THRESHOLD` |
| NMI threshold | 0.02 | `NMI_THRESHOLD` |
| Bootstrap iterations | 1000 | `n_iterations` |
| Single-round planning cap | 300 | `batch_size = min(max_gap, 300)` |
| Generation validation concurrency | 10 | `asyncio.Semaphore(10)` |
| LangGraph recursion limit | 10000 | `config={"recursion_limit": 10000}` |

## Notes

- Script paths are currently hard-coded as `/root/v2/...`; update input and output paths before migrating to another directory.
- The program calls the LLM many times, so runtime and cost depend on sample size, planning rounds, and API response speed.
- `synthetic_data.csv` will be overwritten. Back up old results before running.
- If the planner loops for a long time, first check the low-probability filtering threshold, LLM JSON return format, and `ignored_features` logs.
- Some Chinese comments in the code may display as garbled text in certain terminals. This does not affect Python execution.

This repository is for peer-review purposes only. All rights reserved.