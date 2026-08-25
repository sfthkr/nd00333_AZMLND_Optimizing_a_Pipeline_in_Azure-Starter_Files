# Optimizing an ML Pipeline in Azure — Project Summary

> **Note on tooling:** This project was implemented using the **Azure ML Python SDK v2** (`azure-ai-ml`, `azure-identity`) rather than the older SDK v1 (`azureml-core`, `azureml-train.hyperdrive`, etc.) that the original walkthrough is based on. As a result, several commands differ from the "classic" SDK v1 syntax — most notably, hyperparameter tuning is done through a `command` job wrapped in `.sweep()` instead of a separate `HyperDriveConfig`/`HyperDriveRun` object, and AutoML is configured via the `azure.ai.ml.automl` module and submitted through `ml_client.jobs.create_or_update()` instead of `Experiment.submit()`. Functionally the two approaches are equivalent; only the API surface differs.

## Table of Contents
- [Overview](#overview)
- [Summary](#summary)
- [Scikit-learn Pipeline](#scikit-learn-pipeline)
- [AutoML](#automl)
- [Pipeline Comparison](#pipeline-comparison)
- [Future Work](#future-work)
- [Proof of Cluster Clean-up](#proof-of-cluster-clean-up)
- [Citation](#citation)
- [References](#references)

---

## Overview

This project was completed as part of the Udacity Azure ML Nanodegree. The goal was to build and tune an Azure Machine Learning pipeline using the **Python SDK v2** together with a hand-written Scikit-learn Logistic Regression model, tuning its hyperparameters via a **sweep job** (the SDK v2 equivalent of HyperDrive). Separately, **Azure AutoML** was applied to the same dataset in order to find a strong model automatically, so the two approaches could be compared side by side.

The overall workflow followed four stages:

1. Connect to the Azure ML workspace via `MLClient`, provision a compute cluster, and build a conda environment for the training script.
2. Define a `command` job running `train.py` and wrap it in `.sweep()` to search for the best hyperparameters for the logistic regression model.
3. Load the same dataset and run an `automl.classification` job to independently discover an optimized model.
4. Compare the outcomes of both approaches and document the findings (this summary).

---

## Summary

The dataset consists of marketing records tied to phone-based direct marketing campaigns run by a Portuguese bank (`bankmarketing_train.csv`, 32,950 rows × 21 columns). The prediction task is binary classification: will a given client subscribe to a term deposit (target column `y`)?

The best-performing **HyperDrive-equivalent sweep run** was **`ashy_sail_69kn9r3g69_12`**, with hyperparameters `C=0.1` and `max_iter=100`, achieving an accuracy of **0.9174506828528073**. Out of 16 total trials, 14 logged an accuracy metric; this run had the highest value among them.

The best-performing **AutoML run** was `coral_stamp_3fvbjkfk`, a **VotingEnsemble** achieving an accuracy of **0.9184825**, edging out the sweep's best Logistic Regression model.

---

## Scikit-learn Pipeline

**Pipeline architecture: workspace, compute, environment, data, hyperparameter search, and the classification model**

**Workspace connection**

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential()
)
```

The workspace, subscription, and resource group are resolved automatically from the local `config.json` using `DefaultAzureCredential` — this replaces the SDK v1 pattern of instantiating a `Workspace` object directly.

**Compute cluster**

```python
from azure.ai.ml.entities import AmlCompute

cpu_cluster_name = "lab-cluster-compute-notebook"

try:
    compute_target = ml_client.compute.get(cpu_cluster_name)
except Exception:
    compute_target = AmlCompute(
        name=cpu_cluster_name,
        type="amlcompute",
        size="Standard_D2_v2",
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=120,
    )
    compute_target = ml_client.compute.begin_create_or_update(compute_target).result()
```

A `Standard_D2_v2` cluster was provisioned, scaling between 0 and 4 nodes, and scaling down after 120 seconds of idle time — this keeps cost down between runs while still allowing parallel trials during the sweep.

**Environment**

```python
from azure.ai.ml.entities import Environment

sklearn_env = Environment(
    name="udacity-sklearn-env",
    description="Scikit-learn environment for Udacity project",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="conda.yml"
)
sklearn_env = ml_client.environments.create_or_update(sklearn_env)
```

The conda environment (Python 3.8) pins `scikit-learn`, `pandas`, `numpy`, `mlflow<3`, and `azureml-mlflow`, built on top of the standard `openmpi4.1.0-ubuntu20.04` base image.

**Command job + sweep (parameter sampler)**

```python
from azure.ai.ml import command
from azure.ai.ml.sweep import Choice, BanditPolicy

job = command(
    code="./",
    command="python train.py --C ${{inputs.C}} --max_iter ${{inputs.max_iter}}",
    environment=f"{sklearn_env.name}:{sklearn_env.version}",
    compute=cpu_cluster_name,
    inputs={"C": 1.0, "max_iter": 100}
)

sweep_job = job(
    C=Choice(values=[0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 200, 500, 1000]),
    max_iter=Choice(values=[50, 100, 200, 300])
).sweep(
    compute=cpu_cluster_name,
    sampling_algorithm="random",
    primary_metric="Accuracy",
    goal="maximize"
)
```

In SDK v2, hyperparameter tuning starts from a base `command` job, which is then turned into a sweep by calling `.sweep()` on it with a chosen `sampling_algorithm`. This replaces SDK v1's standalone `RandomParameterSampling` object. `C` (regularization strength) and `max_iter` (maximum solver iterations) were both searched over discrete values using `Choice`, with `sampling_algorithm="random"` — the SDK v2 equivalent of `RandomParameterSampling`. Random sampling was chosen for speed and its compatibility with early termination of low-performing runs; `GridParameterSampling` or `BayesianParameterSampling` would be reasonable alternatives if compute budget were less of a concern.

**Early stopping policy**

```python
sweep_job.early_termination = BanditPolicy(
    evaluation_interval=2,
    slack_factor=0.1
)
```

Same `BanditPolicy` as SDK v1, just assigned to `sweep_job.early_termination` instead of passed into a `HyperDriveConfig`. `evaluation_interval=2` checks the policy every 2 logged intervals; `slack_factor=0.1` terminates any run whose primary metric falls outside 10% of the best-performing run so far. This lets top runs finish undisturbed while cutting off clearly underperforming ones early.

**Run limits**

```python
sweep_job.set_limits(
    max_total_trials=16,
    max_concurrent_trials=4,
    timeout=3600
)
```

The sweep was capped at 16 total trials, up to 4 running concurrently, with an overall timeout of one hour (3600 seconds) — another SDK v2-specific construct (`set_limits()`) that consolidates settings that were spread across multiple parameters in `HyperDriveConfig` in SDK v1.

**Best run**

Metrics were pulled back via MLflow (`workspace.mlflow_tracking_uri`) rather than the SDK v1 `Run` object model, since SDK v2 leans on MLflow as the primary tracking interface. Out of 16 trials, 14 completed with a logged accuracy metric. The best was:

| Field | Value |
|---|---|
| Run name | `ashy_sail_69kn9r3g69_12` |
| Accuracy | 0.9174506828528073 |
| C | 0.1 |
| max_iter | 100 |

## AutoML

**Model and hyperparameter configuration produced by AutoML**

```python
from azure.ai.ml import automl, Input
from azure.ai.ml.constants import AssetTypes

training_data = Input(
    type=AssetTypes.URI_FILE,
    path="./bankmarketing_train.csv"
)

automl_job = automl.classification(
    compute=cpu_cluster_name,
    experiment_name="udacity-automl",
    training_data=training_data,
    target_column_name="y",
    primary_metric="accuracy",
    n_cross_validations=5
)

automl_job.set_limits(timeout_minutes=30)
```

In SDK v2, AutoML runs are built with `automl.classification(...)` and submitted the same way as any other job, via `ml_client.jobs.create_or_update(automl_job)` — this replaces the SDK v1 `AutoMLConfig` + `Experiment.submit()` pattern.

- **`compute`** — reuses the same `lab-cluster-compute-notebook` cluster used for the sweep job.
- **`training_data`** — a local `Input` of type `URI_FILE` pointing at `bankmarketing_train.csv`, uploaded automatically at submission time.
- **`target_column_name='y'`** — the label column to predict.
- **`primary_metric='accuracy'`** — accuracy was selected as the metric AutoML optimizes for.
- **`n_cross_validations=5`** — 5-fold cross-validation; metrics are averaged across folds to reduce the risk of overfitting to a single split.
- **`timeout_minutes=30`** (set via `set_limits()`) — the current Udacity requirement is a 30-minute cap, up from the 15 minutes used in the original SDK v1 walkthrough.

**Results**

The best run, `coral_stamp_3fvbjkfk`, selected a `VotingEnsemble` as its final model, reaching an accuracy of **0.9184825** and an `AUC_weighted` of **0.9486233**. A `VotingEnsemble` combines the predictions of several previously-trained AutoML child models, weighting each one's contribution to the final vote. In this run the ensemble combined nine child models across five algorithm types — `XGBoostClassifier` (4 models, combined weight ≈ 0.571), `LightGBM` (2 models, combined weight ≈ 0.214), `LogisticRegression` (weight ≈ 0.071), `SGD` (weight ≈ 0.071), and `RandomForest` (weight ≈ 0.071) — with `XGBoostClassifier` contributing the largest share of the vote.

---

## Pipeline Comparison

**How the two models stack up, and why**

| Sweep / HyperDrive-equivalent Model |                                |
| ------------------------------------ | ------------------------------ |
| Run name                             | ashy_sail_69kn9r3g69_12        |
| Accuracy                             | 0.9174506828528073             |
| C                                    | 0.1                             |
| max_iter                             | 100                             |

| AutoML Model  |                |
| -------------- | -------------- |
| Run ID         | coral_stamp_3fvbjkfk |
| Accuracy       | 0.9184825 |
| AUC_weighted   | 0.9486233 |
| Algorithm      | VotingEnsemble |

The AutoML `VotingEnsemble` model achieved a slightly higher accuracy (0.9184825) compared to the hyperparameter-tuned Logistic Regression model (0.9174507), representing a marginal improvement of approximately 0.1 percentage points. This modest difference suggests that while ensemble methods can capture more complex, non-linear patterns in the data by combining multiple algorithms (`XGBoostClassifier`, `LightGBM`, `LogisticRegression`, `SGD`, and `RandomForest`), the logistic regression model with optimized hyperparameters performs nearly as well on its own.

The two approaches also come with different trade-offs beyond raw accuracy. AutoML's `VotingEnsemble` required no manual algorithm selection or feature engineering and automatically searched across many model types, but the resulting model is a black-box combination of nine underlying pipelines, making it harder to interpret and slower to score at inference time. The HyperDrive-tuned Logistic Regression, by contrast, offers a single, transparent, well-understood model — its coefficients can be inspected directly, and it is cheaper and faster to deploy and run predictions with — but it required the model family to be chosen manually up front and only its two hyperparameters (`C` and `max_iter`) were tuned. In practice, AutoML is a good fit when squeezing out the last bit of performance matters and compute budget allows for it, while the tuned Logistic Regression is a strong choice when interpretability, faster inference, or simpler deployment are priorities.

---

## Future Work

**Possible improvements for future iterations**

**Class imbalance.** The dataset is heavily skewed toward one class (most clients do not subscribe to a term deposit). This is a well-known problem in classification: a model can appear highly accurate simply by favoring the majority class, while performing poorly on the minority class — meaning accuracy alone can be a misleading measure of quality.

Common approaches to mitigating imbalance include:

1. Choosing a different evaluation metric — e.g., AUC_weighted, which handles imbalance better.
2. Switching to a different algorithm.
3. Under-sampling the majority class.
4. Over-sampling the minority class.
5. Using a dedicated library such as `imbalanced-learn`.

Addressing the imbalance would likely be one of the biggest levers for improving the model in a future iteration.

**Cross-validation folds.** Increasing `n_cross_validations` further would likely improve robustness, since more folds mean training on more varied subsets of the data — but at the cost of longer training time and higher compute cost, so there's a trade-off to manage against the `timeout_minutes` limit.

**Sweep budget.** Raising `max_total_trials` and/or `max_concurrent_trials` in the sweep job could surface better hyperparameter combinations, again at the cost of compute time.

---

## Proof of Cluster Clean-up

See the last cell of the notebook.


---

## Citation

Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, Elsevier, 62:22–31.

---

## References

- Udacity Nanodegree course material
- Microsoft Docs: Azure ML Python SDK v2 overview
- Microsoft Docs: Tutorial on building a classification model with Azure Automated ML
- UCI Bank Marketing dataset (via Kaggle)
- Microsoft Docs: Creating and running ML pipelines with the Azure ML SDK
- Microsoft Docs: Configuring data splits and cross-validation in Azure AutoML
- Microsoft Docs: Handling imbalanced data in automated ML
- Analytics Vidhya: 10 techniques for handling class imbalance in ML

---

*Adapted from the original project structure by dimikara, updated to reflect this personal implementation using Azure ML SDK v2. Original reference repository: [Optimizing-an-ML-Pipeline-in-Azure](https://github.com/dimikara/Optimizing-an-ML-Pipeline-in-Azure)*
