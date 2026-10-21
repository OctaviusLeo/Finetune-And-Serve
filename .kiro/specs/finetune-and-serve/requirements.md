# Requirements Document

## Introduction

Fine-Tune and Serve is a project that fine-tunes a small open-weight language model on **one narrow, measurable task** on a single consumer GPU (an NVIDIA RTX 3080), proves the fine-tuned model beats prompting the base model alone, and then quantizes and serves it through an OpenAI-compatible endpoint with measured latency and throughput. The project is portfolio evidence for ML-engineer roles, which ask for training, evaluation, model optimization, and deployment — not just calling a hosted API. Every reported number must therefore be reproducible from a script in this repository and recorded with its generating command, seed, and full commit hash, and every third-party dependency, model, and dataset must be proven on the target machine and license-checked.

Before any full training run, a short dry run must prove that the chosen model size, fine-tuning method (for example LoRA or QLoRA on 4-bit base weights), and sequence length fit in the GPU's VRAM, and the measured peak memory must be recorded. The quality case is made against baselines measured on the same test set and metric: the base model zero-shot, the base model few-shot, and — within an owner-set spend cap — one hosted model few-shot. The fine-tuning itself is parameter-efficient, uses fixed seeds, records loss curves and config and commit hash per run, includes a small hyperparameter sweep with a written rationale, and runs at least three seeds for the final configuration. Evaluation reports the test-set metric with 95% confidence intervals and an error analysis of at least 50 grouped failures. Serving loads or merges the adapters, quantizes the model, exposes an OpenAI-compatible endpoint, and is load-tested for P50/P95 latency, tokens per second, and maximum concurrent requests, with quality compared before and after quantization. The project ships a model card and a README whose results table and plots are generated.

The work is organized into milestones M0 through M5, each ending in an automated verification gate (a repository command whose exit status decides pass or fail). M0 proves the GPU environment and the VRAM dry run. M1 delivers the task definition, the data pipeline with fixed overlap-checked splits, and the baselines. M2 delivers the parameter-efficient fine-tuning runs and the hyperparameter sweep. M3 delivers the evaluation with confidence intervals and the grouped error analysis. M4 delivers quantization, the OpenAI-compatible serving endpoint, and the load test. M5 delivers the model card and the README.

This document defines the requirements only. Model selection, dataset selection, prompt wording, fine-tuning library choice, quantization method, and other implementation details are deferred to the design phase.

## Glossary

- **System**: The complete Fine-Tune and Serve project, comprising the environment proof, the data pipeline, the baselines, the fine-tuning runs, the evaluation harness, the quantization and serving stack, and the documentation.
- **Task**: The one narrow, measurable task the model is fine-tuned to perform (for example text-to-SQL or structured JSON extraction).
- **Task_Metric**: The single automatic metric that scores Task outputs (for example execution accuracy for text-to-SQL, or field-level F1 for JSON extraction), expressed in a documented numeric range.
- **Dataset**: The public, license-clear dataset used for the Task.
- **Split**: One of the three fixed, disjoint subsets of the Dataset: the Train_Split, the Validation_Split, and the Test_Split.
- **Train_Split**: The fixed subset used to fine-tune the model.
- **Validation_Split**: The fixed subset used for the hyperparameter sweep and model selection.
- **Test_Split**: The fixed subset, never used for training or model selection, on which all reported quality numbers are measured.
- **Base_Model**: The small open-weight language model that is fine-tuned, used unmodified for the base-model baselines.
- **Finetuned_Model**: The Base_Model after parameter-efficient fine-tuning on the Train_Split.
- **Adapter**: The parameter-efficient fine-tuning weights (for example LoRA weights) produced by a fine-tuning Run and applied to or merged into the Base_Model.
- **PEFT**: Parameter-efficient fine-tuning (for example LoRA or QLoRA on 4-bit base weights) that trains the Adapter while leaving most Base_Model weights frozen.
- **Run**: A single fine-tuning execution with a fixed seed that produces an Adapter, a loss curve, a recorded config, and a recorded commit hash.
- **Sweep**: The small hyperparameter sweep over a documented set of configurations, with a written rationale, that selects the final configuration on the Validation_Split.
- **Final_Configuration**: The hyperparameter configuration selected by the Sweep and used for the at-least-three-seed final runs.
- **Baseline**: A measured reference result on the Test_Split and Task_Metric that the Finetuned_Model is compared against: the Base_Model zero-shot, the Base_Model few-shot, and the Hosted_Model few-shot.
- **Hosted_Model**: The one hosted API model evaluated few-shot as a Baseline, used under the Spend_Cap.
- **Spend_Cap**: The owner-set spend limit, expressed in United States dollars, for the Hosted_Model.
- **Dry_Run**: The short fine-tuning dry run at M0 that proves the chosen model size, PEFT method, and sequence length fit in VRAM and records peak GPU memory.
- **VRAM**: The GPU memory of the NVIDIA RTX 3080, in mebibytes, as reported by `nvidia-smi`.
- **Quantized_Model**: The served model after quantization (for example 4-bit or 8-bit) of the merged or adapter-applied Finetuned_Model.
- **Serving_Endpoint**: The OpenAI-compatible endpoint that serves the Quantized_Model.
- **Load_Test**: The test that drives concurrent requests at the Serving_Endpoint and measures latency, throughput, and maximum concurrency.
- **Error_Analysis**: The analysis of at least 50 Finetuned_Model failures on the Test_Split, grouped by failure type.
- **Confidence_Interval**: A 95% confidence interval for a reported Task_Metric value, computed from at least 1,000 bootstrap resamples.
- **Model_Card**: The document recording intended use, data, metrics, limitations, and the licenses of the Base_Model and the Dataset.
- **Measured_Value**: A number produced by a script in this repository, recorded with its generating command, seed, and full 40-character commit hash.
- **Results_Table**: The generated file containing the measured Baseline and Finetuned_Model results, including Confidence_Intervals.
- **ENVIRONMENT_Doc**: The file `docs/ENVIRONMENT.md` recording exact tool, model, and dependency versions proven on the machine.
- **THIRD_PARTY_Doc**: The file `THIRD_PARTY.md` recording each third-party package, model, and dataset and its license.
- **CI_Pipeline**: The GitHub Actions continuous-integration pipeline.
- **Smoke_Evaluation**: The tiny evaluation that runs on CPU in CI to exercise the data-processing and metric code paths end to end.

## Requirements

### Requirement 0: Governing non-functional requirements

**User Story:** As the repository owner, I want every result reproducible and every dependency proven and license-checked, so that I can defend every line of the project in an interview.

#### Acceptance Criteria

1. WHERE a number is reported in the README, in `docs/`, in the Model_Card, or in the Results_Table, THE System SHALL produce that number from a script in this repository and SHALL record, in the same document and within the same section as the number, the generating command, the seed, and the full 40-character commit hash.
2. WHERE a System component uses randomness, THE System SHALL expose a configurable seed through a documented configuration parameter.
3. WHEN two runs of a System component execute with the same seed, the same configuration, and the same model, THE System SHALL produce numeric results that are equal to within an absolute tolerance of 1e-6 for floating-point values and exactly equal for integer and categorical values, excluding timestamps, absolute file paths, hardware-nondeterministic GPU-kernel variation that the System documents, and values a model provider documents as nondeterministic.
4. IF a Run or evaluation fails or misses a stated target, THEN THE System SHALL report, in the run output, the actual result and the identified cause, SHALL terminate with a non-zero exit status, and SHALL leave the stated target and the test unchanged.
5. THE System SHALL record, in the ENVIRONMENT_Doc, the exact version string of every tool, library, API, and model it relies on, including the Base_Model name and version and, where used, the Hosted_Model name and version.
6. WHEN a dependency, tool, or model version is recorded in the ENVIRONMENT_Doc, THE System SHALL include, alongside that version, at least one of the following verifiable artifacts obtained on the target machine: a package-manager query result showing that exact version, a documentation reference for that exact version, or an automated test that exercises the dependency and passes against that exact version.
7. WHERE a third-party package, model, or dataset is added to the repository, THE System SHALL record, in the THIRD_PARTY_Doc, its name, its version, and its license identifier.
8. WHERE Dataset content or a reused module is derived from an external source, THE System SHALL record the source and its license in the THIRD_PARTY_Doc.
9. THE System SHALL read every secret, including any Hosted_Model API key, from an environment variable and SHALL list every secret-bearing file in `.gitignore` so that no such file is tracked in version control.
10. THE System SHALL pin the exact version of every dependency it declares.
11. WHERE a module contains more than one public function or exceeds 50 source lines, THE System SHALL provide unit tests achieving at least 80 percent statement coverage for that module, and THE README SHALL state the exact single command that runs those tests.
12. THE System SHALL remain a private repository without a public license, published package, or public release, and SHALL NOT publish the Finetuned_Model weights or Adapter, until the repository owner directs otherwise.
    - **Owner decision, 2026-10-21:** the repository was released publicly under the MIT License. The repository and license parts of this requirement are superseded; the Finetuned_Model weights and Adapter remain unpublished.
13. IF a required secret environment variable is unset or empty when a System component that depends on it starts, THEN THE System SHALL terminate before issuing any request that requires that secret, SHALL emit an error indicating which environment variable is missing, and SHALL exit with a non-zero exit status.

### Requirement 1: GPU environment and VRAM dry run (M0)

**User Story:** As the repository owner, I want a validated GPU environment and a proven VRAM budget, so that I know the chosen model, method, and sequence length fit before I commit to a full training run.

#### Acceptance Criteria

1. WHEN the environment check runs, THE System SHALL record in the ENVIRONMENT_Doc the output of `nvidia-smi`, including the reported total VRAM in mebibytes of the NVIDIA RTX 3080.
2. WHEN the environment check runs, THE System SHALL verify that the installed Python version is 3.11 or newer, and IF it is not, THEN THE System SHALL report the failure, record in the ENVIRONMENT_Doc the prerequisite that failed, and exit with a non-zero exit status.
3. WHEN the Dry_Run runs, THE System SHALL execute a short parameter-efficient fine-tuning step on the chosen Base_Model at the chosen PEFT method and sequence length, and SHALL measure and record in the ENVIRONMENT_Doc the peak GPU memory in mebibytes used during that step.
4. THE System SHALL select a Base_Model, PEFT method, and sequence length whose recorded Dry_Run peak GPU memory does not exceed the total VRAM reported by `nvidia-smi`.
5. IF the Dry_Run peak GPU memory exceeds the total VRAM reported by `nvidia-smi`, THEN THE System SHALL report the measured peak and the VRAM limit, record the failing condition in the ENVIRONMENT_Doc, and exit with a non-zero exit status.
6. THE System SHALL record in the ENVIRONMENT_Doc the Base_Model name, version, quantization used for the Dry_Run, and license, and the exact version of the fine-tuning library.
7. WHERE a Hosted_Model is used, THE System SHALL record the Spend_Cap in United States dollars in the ENVIRONMENT_Doc and SHALL read the Hosted_Model API key from an environment variable.
8. WHEN the Dry_Run completes successfully, THE System SHALL record in the ENVIRONMENT_Doc the chosen PEFT method, the sequence length, the per-device batch size, and the generating command, seed, and full 40-character commit hash of the Dry_Run.

### Requirement 2: Task and data pipeline (M1)

**User Story:** As the repository owner, I want one narrow task with a license-clear dataset, fixed splits, and an automatic metric, so that quality is measured objectively and reproducibly.

#### Acceptance Criteria

1. THE System SHALL define exactly one Task and exactly one Task_Metric that scores Task outputs automatically, and SHALL document the Task_Metric's numeric range.
2. THE System SHALL assemble the Dataset from a public, license-clear dataset and SHALL record the Dataset source and license in the THIRD_PARTY_Doc.
3. THE System SHALL partition the Dataset into a fixed Train_Split, Validation_Split, and Test_Split whose membership is identical across repeated runs with the same seed and configuration.
4. WHEN the splits are produced, THE System SHALL verify that the Train_Split, Validation_Split, and Test_Split are pairwise disjoint, and IF any example appears in more than one Split, THEN THE System SHALL report every overlapping example identifier and the total count of overlapping examples and exit with a non-zero exit status.
5. WHEN the data pipeline processes a raw Dataset example into a training or evaluation example, THE System SHALL apply a documented, deterministic transformation such that the same raw example and seed produce the identical processed example.
6. IF a raw Dataset example is malformed or missing a field the Task requires, THEN THE System SHALL exclude that example, report its identifier and the reason, and record the total count of excluded examples.
7. THE System SHALL record the example count of each Split as a Measured_Value.

### Requirement 3: Baselines (M1)

**User Story:** As the repository owner, I want baselines measured on the same test set and metric, so that I can show whether fine-tuning beats prompting alone.

#### Acceptance Criteria

1. WHEN the baselines are evaluated, THE System SHALL evaluate the Base_Model zero-shot, the Base_Model few-shot, and one Hosted_Model few-shot, each on the Test_Split and scored by the Task_Metric.
2. WHEN a baseline is reported, THE System SHALL report its Task_Metric value as a Measured_Value within the documented metric range, together with its 95% Confidence_Interval computed from at least 1,000 bootstrap resamples.
3. WHERE the Hosted_Model baseline runs, THE System SHALL record its token counts and per-token prices and SHALL compute and report the total spend of that baseline in United States dollars, and THE System SHALL NOT exceed the Spend_Cap.
4. IF running the Hosted_Model baseline would cause recorded spend to exceed the Spend_Cap, THEN THE System SHALL stop before issuing the request that would exceed the cap, report the spend reached and the cap, and exit with a non-zero exit status.
5. WHEN the few-shot baselines run, THE System SHALL use a fixed, documented set of few-shot exemplars drawn only from the Train_Split, and SHALL NOT draw any exemplar from the Validation_Split or Test_Split.
6. THE System SHALL record the exact prompt template used for each baseline so that the baseline is reproducible from the recorded template and seed.

### Requirement 4: Parameter-efficient fine-tuning and sweep (M2)

**User Story:** As the repository owner, I want parameter-efficient fine-tuning with fixed seeds, recorded runs, and a small sweep, so that the training is reproducible and the final configuration is justified.

#### Acceptance Criteria

1. WHEN a Run executes, THE System SHALL fine-tune the Base_Model with a PEFT method that leaves most Base_Model weights frozen and produces an Adapter.
2. WHEN a Run executes, THE System SHALL use a fixed seed and SHALL record, for that Run, its loss curve, its full configuration, its seed, and the full 40-character commit hash.
3. WHEN the Sweep runs, THE System SHALL evaluate a documented set of at least three hyperparameter configurations on the Validation_Split and SHALL record a written rationale for the configurations chosen for the Sweep.
4. WHEN the Sweep completes, THE System SHALL select the Final_Configuration using the Validation_Split Task_Metric and SHALL record which configuration was selected and the Validation_Split metric value that justified the selection.
5. WHEN the final training is performed, THE System SHALL execute at least three Runs of the Final_Configuration, each with a distinct fixed seed, and SHALL record the Adapter, loss curve, config, seed, and commit hash of each.
6. WHEN two Runs execute with the same seed and the same configuration on the same machine, THE System SHALL produce training loss values that are equal within the tolerance of Requirement 0.3, excluding the documented hardware-nondeterministic GPU-kernel variation.
7. THE System SHALL select the Final_Configuration and the number of training steps using only the Train_Split and Validation_Split and SHALL NOT use the Test_Split for any training or model-selection decision.

### Requirement 5: Evaluation with confidence intervals and error analysis (M3)

**User Story:** As the repository owner, I want the test-set metric with confidence intervals and a grouped error analysis, so that I can report quality honestly and explain where the model fails.

#### Acceptance Criteria

1. WHEN the evaluation runs, THE System SHALL report the Finetuned_Model's Task_Metric on the Test_Split as a Measured_Value within the documented metric range, together with its 95% Confidence_Interval computed from at least 1,000 bootstrap resamples.
2. WHEN the evaluation reports the Finetuned_Model result and the baselines, THE System SHALL report them on the same Test_Split and the same Task_Metric so that the comparison is like-for-like.
3. WHEN the Finetuned_Model is reported with at least three seeds, THE System SHALL report the mean Test_Split Task_Metric across those seeds and a measure of its spread across seeds.
4. WHEN the Error_Analysis runs, THE System SHALL select at least 50 Test_Split examples on which the Finetuned_Model failed according to the Task_Metric and SHALL assign each selected failure to a documented failure-type group.
5. WHEN the Error_Analysis is reported, THE System SHALL report the count of failures in each failure-type group and SHALL ensure the group counts sum to the number of analyzed failures.
6. THE evaluation SHALL be invocable from a single non-interactive command.
7. WHEN the evaluation runs twice on the Test_Split with the same seed and the same model, THE System SHALL produce Task_Metric values equal within the tolerance of Requirement 0.3.
8. IF a Test_Split example required by the Task_Metric is missing or malformed, THEN THE System SHALL report that example as not scored with its identified cause and SHALL continue scoring the remaining examples.

### Requirement 6: Quantization and serving (M4)

**User Story:** As a user, I want the fine-tuned model quantized and served through an OpenAI-compatible endpoint, so that I can query it like any other OpenAI-compatible model.

#### Acceptance Criteria

1. WHEN the serving model is prepared, THE System SHALL load or merge the Adapter into the Base_Model and SHALL quantize the resulting model, recording the quantization method and bit width.
2. WHEN the Serving_Endpoint is started, THE System SHALL expose an OpenAI-compatible endpoint that accepts a well-formed completion or chat request and returns a generated completion.
3. WHEN the Quantized_Model is evaluated on the Test_Split with the Task_Metric, THE System SHALL report its Task_Metric and 95% Confidence_Interval alongside the pre-quantization Finetuned_Model's Task_Metric so that the quality change from quantization is measured.
4. THE System SHALL verify that the Quantized_Model's recorded peak GPU memory while serving does not exceed the total VRAM reported by `nvidia-smi`.
5. IF a required secret environment variable is unset or empty when the Serving_Endpoint starts, THEN THE System SHALL fail to start and SHALL report, in the start-up output, which required secret was missing.
6. WHEN the documented single start command is run, THE System SHALL start the Serving_Endpoint and reach a state in which it accepts requests.

### Requirement 7: Load test (M4)

**User Story:** As the repository owner, I want the served model load-tested, so that I can report latency, throughput, and maximum concurrency on my GPU.

#### Acceptance Criteria

1. WHEN the Load_Test runs, THE System SHALL drive requests at the Serving_Endpoint and report P50 latency in milliseconds and P95 latency in milliseconds as Measured_Values, with P50 less than or equal to P95.
2. WHEN the Load_Test runs, THE System SHALL report the throughput in generated tokens per second as a Measured_Value.
3. WHEN the Load_Test runs, THE System SHALL report the maximum number of concurrent requests sustained on the NVIDIA RTX 3080, defined by a documented sustaining criterion.
4. WHEN the Load_Test reports a result, THE System SHALL record the Measured_Value with its generating command, seed, concurrency level, and full 40-character commit hash.
5. IF the Serving_Endpoint returns an error or times out for a request during the Load_Test, THEN THE System SHALL count that request as failed, report the count of failed requests, and exclude failed requests from the latency and throughput statistics while reporting the failed count alongside them.
6. THE Load_Test SHALL be invocable from a single non-interactive command.

### Requirement 8: Model card (M5)

**User Story:** As a reader, I want a model card, so that I understand the model's intended use, data, metrics, limitations, and licensing.

#### Acceptance Criteria

1. THE Model_Card SHALL state the Finetuned_Model's intended use and at least one documented out-of-scope or discouraged use.
2. THE Model_Card SHALL identify the Dataset, its source, and the Splits used for training, validation, and testing.
3. WHERE the Model_Card reports a metric value, THE System SHALL source that value from a file generated by a repository script and SHALL report it with its Confidence_Interval.
4. THE Model_Card SHALL state at least one documented limitation of the Finetuned_Model supported by the Error_Analysis.
5. THE Model_Card SHALL state the license of the Base_Model and the license of the Dataset.
6. IF the Base_Model license or the owner's employment terms do not permit publishing the Finetuned_Model weights, THEN THE Model_Card SHALL state that the weights are not published and the System SHALL NOT publish them.

### Requirement 9: Tests and continuous integration (M5)

**User Story:** As the repository owner, I want unit tests and a CI smoke evaluation, so that data-processing and metric code are protected and CI stays runnable without a GPU.

#### Acceptance Criteria

1. THE System SHALL provide unit tests for the data-processing code and for the Task_Metric code.
2. WHEN a change is pushed to the repository's default branch or a pull request targeting the default branch is opened or updated, THE CI_Pipeline SHALL run the unit tests and the Smoke_Evaluation.
3. THE Smoke_Evaluation SHALL run on CPU without a GPU and SHALL exercise the data-processing and Task_Metric code paths end to end on a tiny fixed input.
4. IF any unit test or the Smoke_Evaluation fails during a CI run, THEN THE CI_Pipeline SHALL report a failed status for that run.
5. WHEN the Task_Metric is unit-tested, THE System SHALL include at least one test with a known input and a hand-computed expected metric value and SHALL assert the computed value equals the expected value within the tolerance of Requirement 0.3.
6. THE README SHALL state the exact single command that runs the unit tests and the exact single command that runs the Smoke_Evaluation.

### Requirement 10: README and reproducibility (M5)

**User Story:** As a reader, I want a README with generated results, so that I can understand, run, and reproduce every number in the project.

#### Acceptance Criteria

1. THE README SHALL contain a generated Results_Table comparing the baselines and the Finetuned_Model on the Test_Split Task_Metric, each with a Confidence_Interval, and SHALL contain the pre- and post-quantization quality comparison and the Load_Test latency, throughput, and concurrency results.
2. THE README SHALL contain the generated loss-curve plot and a plot of the sweep or quality results.
3. THE README SHALL state, for each reported result, the exact single command that reproduces that result, and SHALL state the exact single command that starts the Serving_Endpoint, matching the command required by Requirement 6.6.
4. WHERE a number appears in the README or the Model_Card, THE System SHALL source that number from the Results_Table or another file generated by a repository script rather than from a hand-entered value.
5. IF a number in the README or the Model_Card cannot be traced to the Results_Table or another generated file, THEN THE System SHALL report that number as untraceable in the check output and SHALL report a failed status for that check.

## Out of Scope

The following are explicitly out of scope: pretraining a model from scratch; multi-GPU training; reinforcement learning from human feedback (RLHF); and publishing the Finetuned_Model weights or Adapter before the owner approves (the Base_Model's license and the owner's employment terms decide whether the weights may be published).
