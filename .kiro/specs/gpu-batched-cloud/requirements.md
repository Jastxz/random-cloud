# Requirements Document

## Introduction

RandomCloud.jl implements the Random Cloud Method for Neural Architecture Search. The current implementation processes data sample-by-sample and evaluates networks one-at-a-time on CPU. This feature modernizes the computational core to use batched matrix operations and GPU acceleration via CUDA.jl, targeting an NVIDIA GTX 1050 (4 GB VRAM, CUDA 13.0). The public API (`ConfiguracionNube`, `MotorNube`, `ejecutar`) remains backward-compatible; GPU support is opt-in.

## Glossary

- **RedNeuronal**: Feedforward neural network struct holding topology, weight matrices, and bias vectors.
- **MotorNube**: Orchestrator that runs the Random Cloud Method (cloud exploration + refinement).
- **ConfiguracionNube**: Immutable struct of hyperparameters for a cloud run.
- **InformeNube**: Result struct returned by `ejecutar`.
- **Cloud**: The set of N randomly initialised `RedNeuronal` instances explored in one run.
- **Feedforward_Engine**: The subsystem responsible for computing network outputs from inputs.
- **Batch_Evaluator**: The subsystem that evaluates the full dataset in a single matrix multiplication per layer instead of looping over samples.
- **Cloud_Evaluator**: The subsystem that evaluates all N networks in the cloud simultaneously using 3-D tensor operations.
- **GPU_Backend**: The subsystem that manages CUDA device memory, host↔device transfers, and kernel dispatch via CUDA.jl.
- **Batch_Trainer**: The subsystem that performs mini-batch backpropagation as matrix operations.
- **Validation_Suite**: The experimental pipeline that runs the 7-dataset comparison (Iris, Wine, Breast Cancer, Ionosphere, Sonar, Digits, Adult) against baselines.
- **Scalability_Harness**: The benchmarking subsystem that tests on larger datasets (MNIST 60K, Fashion-MNIST, CIFAR-10).

## Requirements

### Requirement 1: Batched Feedforward

**User Story:** As a researcher, I want the feedforward pass to evaluate the entire dataset as a single matrix multiplication per layer, so that I eliminate the sample-by-sample loop and exploit hardware parallelism.

#### Acceptance Criteria

1. WHEN a `RedNeuronal` and a data matrix X (features × samples) are provided, THE Feedforward_Engine SHALL compute the output matrix Y = activation(W × X + b) for each layer in a single matrix operation.
2. THE Feedforward_Engine SHALL produce outputs numerically equivalent (absolute difference ≤ 1e-10 per element) to the existing sample-by-sample `feedforward!` for the same network and data.
3. WHEN the activation vector `acts` is provided, THE Feedforward_Engine SHALL apply the corresponding activation function element-wise to each layer's pre-activation matrix.
4. WHEN the activation vector `acts` is omitted, THE Feedforward_Engine SHALL default to sigmoid activation on every layer.

### Requirement 2: Batched Cloud Evaluation via Tensor Operations

**User Story:** As a researcher, I want all N networks in the cloud evaluated simultaneously using 3-D tensor operations, so that the exploration phase does not loop over networks sequentially.

#### Acceptance Criteria

1. WHEN a cloud of N `RedNeuronal` instances sharing the same topology is provided, THE Cloud_Evaluator SHALL pack their weights into a 3-D tensor of shape (cloud_size × neurons_out × neurons_in) per layer.
2. WHEN the packed weight tensors and a data matrix X are provided, THE Cloud_Evaluator SHALL compute all N feedforward outputs in a single batched operation per layer.
3. THE Cloud_Evaluator SHALL produce per-network accuracy values equivalent (absolute difference ≤ 1e-10) to evaluating each network individually with the Batch_Evaluator.
4. WHEN a network in the cloud is reconstructed to a smaller topology, THE Cloud_Evaluator SHALL support re-packing the remaining networks for the next reduction step.

### Requirement 3: GPU Acceleration

**User Story:** As a researcher, I want to move all heavy computations to the GPU, so that exploration and refinement phases run faster on CUDA-capable hardware.

#### Acceptance Criteria

1. WHEN `ConfiguracionNube` specifies `gpu=true` and a CUDA-capable device is available, THE GPU_Backend SHALL transfer input data, weight tensors, and bias vectors to GPU memory before computation begins.
2. WHILE computations run on GPU, THE Feedforward_Engine SHALL execute matrix multiplications and activation functions using CUDA kernels via CUDA.jl.
3. WHEN `ConfiguracionNube` specifies `gpu=true` and no CUDA-capable device is detected, THE GPU_Backend SHALL raise an informative error indicating that no GPU is available.
4. WHEN `ConfiguracionNube` specifies `gpu=false` or omits the `gpu` field, THE MotorNube SHALL execute entirely on CPU using the existing code paths.
5. THE GPU_Backend SHALL keep all intermediate computation results on GPU memory and transfer only the final `InformeNube` results back to the host.
6. WHEN the total memory required for the cloud weight tensors and data matrix exceeds 3.5 GB, THE GPU_Backend SHALL raise an informative error before attempting allocation, reporting the estimated memory requirement and the device limit.

### Requirement 4: Batched Backpropagation

**User Story:** As a researcher, I want the refinement phase to process mini-batches as full matrix operations instead of looping over individual samples, so that training is faster and GPU-friendly.

#### Acceptance Criteria

1. WHEN a mini-batch of B samples is provided, THE Batch_Trainer SHALL compute the forward pass for all B samples as a single matrix multiplication per layer.
2. WHEN the forward pass for a mini-batch is complete, THE Batch_Trainer SHALL compute gradients for all B samples simultaneously and average them before updating weights.
3. THE Batch_Trainer SHALL produce weight updates that are numerically equivalent (absolute difference ≤ 1e-8 per element) to accumulating individual sample gradients over the same mini-batch.
4. WHEN `gpu=true`, THE Batch_Trainer SHALL execute all forward, backward, and weight-update operations on GPU without transferring intermediate data to the host.
5. WHEN `gpu=false`, THE Batch_Trainer SHALL execute batched matrix operations on CPU, still providing a speedup over the current sample-by-sample loop.

### Requirement 5: Backward Compatibility

**User Story:** As a developer, I want the existing CPU API and test suite to remain fully functional, so that current users are not affected by the GPU additions.

#### Acceptance Criteria

1. THE ConfiguracionNube SHALL accept all existing keyword arguments with their current defaults and behavior unchanged.
2. WHEN `gpu` is not specified, THE ConfiguracionNube SHALL default `gpu` to `false`.
3. THE MotorNube SHALL accept the same constructor signatures as before (`MotorNube(config, entradas, objetivos)` and `MotorNube(config, entradas, objetivos, fn_evaluar)`).
4. WHEN `gpu=false`, THE `ejecutar` function SHALL produce `InformeNube` results statistically equivalent to the current implementation for the same seed.
5. THE existing 130 CPU tests SHALL pass without modification after the GPU feature is added.

### Requirement 6: CUDA.jl Dependency Management

**User Story:** As a developer, I want CUDA.jl to be an optional dependency, so that users without a GPU can install and use RandomCloud.jl without CUDA-related compilation overhead.

#### Acceptance Criteria

1. THE RandomCloud module SHALL declare CUDA.jl as an optional dependency using Julia's package extension mechanism.
2. WHEN CUDA.jl is not installed, THE RandomCloud module SHALL load and function with CPU-only code paths and no import errors.
3. WHEN CUDA.jl is installed, THE RandomCloud module SHALL automatically load the GPU extension providing `gpu=true` support.
4. IF a user sets `gpu=true` but CUDA.jl is not installed, THEN THE ConfiguracionNube constructor SHALL raise an informative error explaining that CUDA.jl must be added to the project.

### Requirement 7: Validation on Standard Benchmarks

**User Story:** As a researcher, I want to run the existing 7-dataset experimental suite with the new batched/GPU implementation, so that I can verify correctness and measure speedup.

#### Acceptance Criteria

1. WHEN the Validation_Suite runs on the 7 standard datasets (Iris, Wine, Breast Cancer, Ionosphere, Sonar, Digits, Adult), THE batched CPU implementation SHALL achieve accuracy within ±1 percentage point of the current sample-by-sample implementation for each dataset using the same seed.
2. WHEN the Validation_Suite runs with `gpu=true`, THE GPU implementation SHALL achieve accuracy within ±1 percentage point of the batched CPU implementation for each dataset using the same seed.
3. THE Validation_Suite SHALL report wall-clock time for each dataset under three modes: original (sample-by-sample), batched CPU, and GPU.
4. THE Validation_Suite SHALL report peak GPU memory usage for each dataset when running in GPU mode.

### Requirement 8: Scalability Testing on Larger Datasets

**User Story:** As a researcher, I want to test the GPU-accelerated method on MNIST (60K), Fashion-MNIST, and CIFAR-10, so that I can assess whether GPU acceleration makes the method viable at scale.

#### Acceptance Criteria

1. WHEN the Scalability_Harness runs on MNIST (60,000 training samples, 784 features, 10 classes), THE GPU implementation SHALL complete a full cloud run (exploration + refinement) within the 4 GB VRAM limit.
2. WHEN the Scalability_Harness runs on Fashion-MNIST (60,000 training samples, 784 features, 10 classes), THE GPU implementation SHALL complete a full cloud run within the 4 GB VRAM limit.
3. WHEN the Scalability_Harness runs on CIFAR-10 (50,000 training samples, 3072 features, 10 classes), THE GPU_Backend SHALL either complete the run or raise the memory-limit error from Requirement 3.6 if the problem exceeds VRAM.
4. THE Scalability_Harness SHALL record and report: accuracy, total wall-clock time, GPU time fraction, and peak VRAM usage for each large dataset.
5. THE Scalability_Harness SHALL compare GPU wall-clock time against CPU wall-clock time and report the speedup factor.
