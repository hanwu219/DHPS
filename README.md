# Graph-Guided Synthetic CHARLS Data Generation and Evaluation

This repository contains a graph-guided pipeline for generating and evaluating
synthetic CHARLS-style elderly population records. The core generation workflow
uses a small real-data reference sample, dependency discovery, bootstrap target
estimation, profile-based allocation, and LLM-assisted semantic validation to
complete high-level health and social variables over an IPF demographic base.

The repository also includes an evaluation suite for distributional fidelity,
dependency preservation, utility, privacy risk, ablation analysis, and structural
constraint violations.

## Repository Structure

```text
.
|-- graph_v2_charls_ipf.py          Main graph-guided IPF generation pipeline
|-- graph_v2_charls_ipf_a.py        Ablation variant
|-- graph_v2_charls_ipf_b.py        Ablation variant
|-- graph_v2_charls_ipf_b+c.py      Ablation variant
|-- graph_v2_charls_ipf_c.py        Ablation variant
|-- graph_v2_charls_great.py        GREAT-based generation script
|-- graph_v2_charls_tabpfn.py       TabPFN-related generation script
|-- graphstate.json                 Example or saved graph state
|-- input.jsonl                     Batch input file for LLM requests
|-- output_deepseek.jsonl           Batch output file from LLM requests
|-- second_fill_log.txt             Log from the second-fill stage
|-- analysis code/                  Evaluation and plotting scripts
`-- great migration directory       GREAT migration-related files
```

## Data Inputs

The main generation script is currently configured for server-style paths:

```text
/root/v2/CHARLS_processed_2020.csv
/root/v2/CHARLS_ipf.csv
```

Most evaluation scripts under `analysis code/` use Windows paths by default:

```text
D:\LLM generate data\data\CHARLS_processed_2020.csv
D:\LLM generate data\data\synthetic_data_v12.csv
D:\LLM generate data\data\synthetic_data_v16.csv
D:\LLM generate data\data\synthetic_data_v17.csv
D:\LLM generate data\data\synthetic_data_v17b.csv
```

Update these paths or pass command-line arguments when running the scripts in a
different environment.

## Real Data Preprocessing

Across the evaluation scripts, the real CHARLS data is processed using the same
standard procedure:

1. Read `CHARLS_processed_2020.csv`.
2. Drop `iwy` and `row_id` when present.
3. Keep records with `income_total >= 0`.
4. Bin `age` into:

   ```text
   60-, 60-64, 65-69, 70-74, 75-79, 80+
   ```

5. Bin `income_total` into tertiles:

   ```text
   Low, Medium, High
   ```

6. Drop the original continuous columns `age` and `income_total`.
7. Keep records with `family_size <= 10`.
8. Drop missing rows for the standard real-data reference.

The main generation pipeline samples 200 real records as a small reference set:

```python
df = df_real.sample(n=200, random_state=24)
```

## Feature Groups

The graph-guided pipeline separates variables into demographic features `D` and
high-level target features `H`.

Demographic features:

```python
["age_bin", "gender", "income_bin", "family_size", "marry", "edu", "health_status"]
```

High-level target features:

```python
["hospital", "exercise", "ins", "satlife", "social_need"]
```

The IPF base contains the demographic features and is progressively completed
with the high-level target features.

## Generation Workflow

The main pipeline in `graph_v2_charls_ipf.py` is implemented as a LangGraph
workflow:

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

### 1. `initialize_state`

Loads the real CHARLS reference data, prepares the IPF base, defines `D` and `H`
feature groups, and initializes the graph state.

### 2. `llm_proposal`

Uses `deepseek-reasoner` to propose plausible dependency edges between `D` and
`H`, and between high-level features. Multiple LLM runs are aggregated by voting.
Each candidate edge is also checked using normalized mutual information.

Default settings:

```text
TOTAL_RUNS = 10
VOTE_THRESHOLD = 6
NMI_THRESHOLD = 0.02
```

Only edges that pass both the voting criterion and the NMI threshold are used in
later stages.

### 3. `bootstrap_target`

Runs bootstrap estimation over the 200-record reference sample. The procedure
estimates:

- Marginal distributions for each high-level feature.
- Joint distributions for validated dependency edges.

The generated target object has the following structure:

```python
{
    "marginals": {
        "hospital": {...},
        "exercise": {...},
        ...
    },
    "joints": {
        "age_bin|hospital": {...},
        "age_bin|ins": {...},
        ...
    }
}
```

Low-probability joint combinations are optionally reviewed by the LLM as a
reality check. Approved corrections are later used by the second-fill stage.

### 4. `profiling`

Uses validated dependencies to group the IPF base into demographic profiles. If
no relevant dependency is found for a target feature, the pipeline falls back to
all demographic variables.

### 5. `planner`

Identifies the most urgent missing high-level feature and category by comparing
the current synthetic population against the bootstrap target distribution.

The urgency score is:

```text
urgency = gap / available_slots
```

The planner allocates at most:

```text
batch_size = min(max_gap, 300)
```

per planning round. It checks both `D-H` and `H-H` joint constraints before
assigning rows to candidate profiles.

### 6. `generator`

Uses `deepseek-chat` to validate whether the planned fills are semantically
reasonable. Validated fills are written back to the synthetic population. Failed
tasks are recorded and used by later planning rounds.

Default concurrency:

```text
asyncio.Semaphore(10)
```

### 7. `second_fill`

Uses LLM-approved low-probability corrections from `bootstrap_target` to fill
remaining missing values. The final generated dataset is written to:

```text
/root/v2/synthetic_data.csv
```

## Evaluation Scripts

The `analysis code/` directory contains a compact evaluation suite.

| Script | Purpose | Main Output |
|---|---|---|
| `01_mutual_information_analysis.py` | Pairwise mutual information comparison between real and one synthetic dataset | MI heatmaps and distribution comparison |
| `02_tsne_visualization.py` | t-SNE visualization of real and synthetic manifold coverage | t-SNE scatter plot |
| `03_multi_model_mutual_information_analysis.py` | Mutual information comparison across multiple synthetic generators | Multi-model MI heatmap and metrics |
| `04_adversarial_verification.py` | Real-vs-synthetic discriminator test | LR/DT AUC and PMSE table |
| `05_tstr_evaluation.py` | Train-on-synthetic, test-on-real utility evaluation | TSTR summary table |
| `06_privacy_risk_analysis.py` | DCR, membership inference, and attribute inference analysis | Privacy risk tables |
| `07_ablation_global_metrics.py` | Ablation metrics based on MI fidelity and discriminator utility | MAE, RMSE, correlation, discriminator metrics |
| `08_violation_frequency_distribution.py` | Near-zero and zero constraint violation distribution | Histogram with fitted density curves |
| `09_ablation_jsd_case_plot.py` | Ablation JSD heatmap and specific-case structural collapse analysis | Figure with heatmap and grouped bars |

Each script can be run independently. Most scripts print tabular results and
save figures into a subdirectory under `analysis code/`.

## Running the Main Pipeline

Install the required dependencies and configure API credentials in `.env`.
At minimum, the generation pipeline requires access to the DeepSeek models used
by LangChain.

Example server-side execution:

```bash
conda activate LLM
cd /root/v2
python graph_v2_charls_ipf.py
```

Example Windows execution:

```powershell
conda run -n LLM python graph_v2_charls_ipf.py
```

## Running Evaluation Scripts

Examples:

```powershell
python "analysis code/03_multi_model_mutual_information_analysis.py"
python "analysis code/06_privacy_risk_analysis.py"
python "analysis code/08_violation_frequency_distribution.py"
python "analysis code/09_ablation_jsd_case_plot.py"
```

Most scripts expose command-line arguments for input paths, output directories,
sample sizes, random seeds, and evaluation thresholds:

```powershell
python "analysis code/08_violation_frequency_distribution.py" --help
```

## Dependencies

Core dependencies include:

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
matplotlib
ipfn
aiohttp
aiofiles
faiss-cpu
```

Install any missing packages in the target environment before running the graph
pipeline or evaluation scripts.

## Key Parameters

| Parameter | Default | Location or Meaning |
|---|---:|---|
| Reference sample size | 200 | `df_real.sample(n=200, random_state=24)` |
| LLM proposal runs | 10 | `TOTAL_RUNS` |
| LLM vote threshold | 6 | `VOTE_THRESHOLD` |
| NMI threshold | 0.02 | `NMI_THRESHOLD` |
| Bootstrap iterations | 1000 | `n_iterations` in `bootstrap_target` |
| Planning batch cap | 300 | `batch_size = min(max_gap, 300)` |
| Generator concurrency | 10 | `asyncio.Semaphore(10)` |
| LangGraph recursion limit | 10000 | `config={"recursion_limit": 10000}` |

## Reproducibility Notes

- The real reference sample is fixed with `random_state=24`.
- The original bootstrap step in `graph_v2_charls_ipf.py` uses pandas sampling
  without an explicit bootstrap seed, so repeated runs may produce minor
  numerical differences.
- The analysis scripts generally expose `--random-state` or equivalent
  arguments where stochastic procedures are used.
- Generated CSV files may be overwritten by the pipeline. Back up important
  outputs before rerunning generation scripts.

## Troubleshooting

If the generation pipeline fails, check the following:

1. `.env` is loaded correctly.
2. The DeepSeek API key is valid and has sufficient quota.
3. `langchain_deepseek` and related LangChain packages are installed.
4. The runtime machine can access the model API.
5. Input CSV paths match the current operating system and directory layout.
6. LLM responses are valid JSON where JSON is expected.

If an evaluation script fails, first verify that all required input CSV files
exist and contain the expected columns listed in `FEATURE_ORDER`.

## Important Notes

- The main generation script and the evaluation scripts currently use different
  default path conventions (`/root/v2/...` versus `D:\...`). Adjust paths before
  moving between server and Windows environments.
- LLM calls can be time-consuming and may incur API costs.
- Some legacy Chinese comments in older scripts may display incorrectly in
  terminals with a different encoding. This does not necessarily affect Python
  execution.
- Evaluation results depend on the specific synthetic CSV files used. Always
  report the input file names and random seeds together with metric values.
