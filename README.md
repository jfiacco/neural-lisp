# Neural-Lisp

A pure Common Lisp neural network library showcasing **Lisp's unique strengths**: macros, condition system, homoiconicity, CLOS, and functional programming.

## 🎯 Advanced Architectures

Now includes **state-of-the-art deep learning architectures**:

- **🔄 Recurrent Networks:** LSTM, GRU, Bidirectional variants
- **🤖 Transformers:** Multi-head attention, encoder-decoder, positional encoding
- **🏷️ CRF:** Linear-chain, Tree CRF for structured prediction
- **🚀 State Space Models:** Mamba, S4 (100K+ token contexts!)

## Unique Features

### Training Macros
```lisp
(with-training (model :mode :train)
  (with-gradient-protection (params :max-norm 5.0 :on-explosion :clip)
    (with-frozen-layers (base-layer)
      ; Automatic mode management
      ; Gradient explosion handling with restarts
      ; Layer freezing with guaranteed restoration
      (train-epoch model data))))
```

### Functional Composition
```lisp
(defvar pipeline (compose #'normalize #'augment #'preprocess))
(-> data pipeline (to-tensor) (add-batch-dim))
```

### Layer Combinators  
```lisp
(defvar resnet-block 
  (residual (sequential (conv 64 3) (relu)) 
            (conv-shortcut 64)))
```

### Condition System
```lisp
; Gradient explodes? Choose interactively:
;  1. Clip and continue (no restart needed!)
;  2. Zero and continue
;  3. Abort training
(with-gradient-protection (params)
  (train model))
```

## Core Features

### Tensor Operations & Autograd
- **Automatic Differentiation**: PyTorch-style autograd with computational graph tracking
- **NumPy-style Broadcasting**: Automatic shape alignment for tensor operations
- **Multiple Backends**: Automatic CPU/BLAS/CUDA GPU selection with configurable thresholds

### Neural Network Layers
- **Basic Layers**: Linear, Sequential containers
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, Log-Softmax
- **ReLU Variants**: Leaky ReLU, ELU, SELU, GELU, Swish, Mish, HardSwish, ReLU6
- **Advanced Activations**: GLU, GeGLU, SwiGLU, Softsign, Softplus, Maxout
- **Normalization**: Layer Normalization, Batch Normalization

### Convolutional Networks
- **N-Dimensional Convolutions**: Conv1D, Conv2D, Conv3D, ConvND
- **Pooling Layers**: Max/Avg Pooling (1D/2D/3D), Global Pooling, Spatial Pyramid Pooling
- **Transposed Convolutions**: Deconvolution layers for upsampling
- **Separable & Dilated**: Depthwise separable, dilated/atrous convolutions

### Recurrent Networks
- **RNN Cells**: Vanilla RNN, LSTM, GRU
- **Bidirectional**: Bidirectional RNN, LSTM, GRU
- **Stacked RNN**: Multi-layer recurrent networks
- **Attention RNN**: Recurrent networks with attention mechanisms

### Transformers
- **Attention Mechanisms**: Scaled dot-product attention, Multi-head attention
- **Positional Encodings**: Sinusoidal, Rotary (RoPE), ALiBi
- **Transformer Blocks**: Encoder layers, Decoder layers, Full Transformer
- **Efficient Attention**: Flash attention, Sparse attention patterns

### State Space Models
- **S4**: Structured State Space Sequence models
- **Mamba**: Selective state space models for 100K+ token contexts
- **Samba**: Hybrid SSM architectures
- **HiPPO Initialization**: High-order Polynomial Projection Operator

### Conditional Random Fields
- **Linear-Chain CRF**: For sequence labeling (NER, POS tagging)
- **Tree CRF**: For tree-structured prediction
- **Semi-Markov CRF**: For segment-level prediction
- **Viterbi Decoding**: Optimal sequence inference

### Residual Architectures
- **ResNet Blocks**: Basic blocks, Bottleneck blocks, Downsample layers
- **EfficientNet Blocks**: MBConv, Squeeze-Excitation, Fused-MBConv
- **ConvNeXt Blocks**: Modern ConvNet building blocks
- **Regularization**: Stochastic Depth, Layer Scale

### Embeddings
- **Token Embeddings**: Word embeddings with padding support
- **Subword Embeddings**: For subword tokenization
- **Positional Encodings**: Sinusoidal and learned positional embeddings
- **Special Embeddings**: OOV handling, Numerical, Byte-level

### Variational Inference
- **Gumbel-Softmax**: Differentiable categorical sampling
- **Reparameterization**: Normal, Bernoulli, Categorical, Dirichlet
- **VAE Components**: Encoder, Decoder, KL divergence
- **Stochastic Layers**: Gaussian and Categorical variational layers

### Optimizers
- **SGD**: Vanilla and with momentum, dampening, weight decay
- **Adam Family**: Adam, AdamW (decoupled weight decay), NAdam, Adamax, AMSGrad
- **Adaptive**: RMSprop, AdaGrad, AdaDelta
- **Gradient Clipping**: By norm and by value

### Loss Functions
- **Regression**: MSE, MAE, Smooth L1 (Huber)
- **Classification**: Cross-Entropy, Binary Cross-Entropy, NLL
- **Divergence**: KL-Divergence
- **Structured**: CRF Loss

### Learning Rate Schedulers
- **Step-based**: StepLR, Exponential
- **Cyclic**: CosineAnnealing, Cyclic LR, One-Cycle
- **Adaptive**: ReduceOnPlateau

### Data Loading
- **Dataset Types**: Tensor, Sequence, Image, Text, CSV, JSON, Arrow
- **HuggingFace Compatible**: Load datasets in HuggingFace/PyTorch format
- **Samplers**: Sequential, Random, Weighted, Distributed
- **Processing**: Map, Filter, Train/Test split, Concatenation

### Testing
- **Comprehensive Test Suite**: 70+ tests using FiveAM

## Installation

### Prerequisites

- SBCL (Steel Bank Common Lisp)
- Quicklisp (for dependency management)

### Installing Dependencies

```lisp
;; In SBCL REPL:
(ql:quickload :fiveam)  ; For running tests
```

### Loading the System

Using ASDF:

```lisp
(asdf:load-system :neural-lisp)
```

Or manually load files:

```lisp
(load "src/neural-network.lisp")
(load "src/backend.lisp")
(load "src/optimizers.lisp")
(load "src/losses.lisp")
(load "src/integration.lisp")
```

## Quick Start

### Basic Tensor Operations

```lisp
(use-package :neural-network)

;; Create tensors
(defvar x (make-tensor #(1.0 2.0 3.0) :shape '(3)))
(defvar y (make-tensor #(4.0 5.0 6.0) :shape '(3)))

;; Operations
(t+ x y)  ; Addition with broadcasting
(t* x y)  ; Element-wise multiplication
(t@ x y)  ; Matrix multiplication
```

### Computation Backends

By default, Neural-Lisp uses **automatic backend selection** (`:auto`) which intelligently chooses between CPU and GPU based on matrix size:

```lisp
(use-package :neural-tensor-backend)

;; Default: Automatic selection (no configuration needed!)
;; - Small matrices (< 128x128): Uses CPU (faster due to lower overhead)
;; - Large matrices (≥ 128x128): Uses GPU (faster due to parallelism)

;; Check backend status
(backend-info)

;; Manual backend selection (if needed):
(use-backend :lisp)  ; Pure Lisp (portable, no dependencies)
(use-backend :blas)  ; Optimized CPU (10-100x faster)
(use-backend :gpu)   ; GPU acceleration (100-1000x faster for large matrices)
(use-backend :auto)  ; Automatic selection (default & recommended)

;; Customize auto-selection threshold
(setf *auto-gpu-threshold* 256)  ; Use GPU for 256x256+ matrices
```

For more details, see `docs/AUTO-BACKEND-SELECTION.md`.

### Training a Neural Network

```lisp
(use-package :neural-network)
(use-package :neural-tensor-optimizers)
(use-package :neural-tensor-losses)
(use-package :neural-tensor-complete)

;; Create model
(defvar model (sequential
                (linear 10 5)
                (relu-layer)
                (linear 5 2)))

;; Create optimizer
(defvar optimizer (make-adam (parameters model) :lr 0.001))

;; Prepare data
(defvar x-train (randn '(100 10)))
(defvar y-train (randn '(100 2)))

;; Train
(fit model optimizer x-train y-train
     :epochs 10
     :batch-size 32
     :loss-fn #'mse-loss
     :verbose t)
```

## Checkpointing Models

Use the built-in checkpoint API to persist any collection of tensors, layers, or
entire models:

```lisp
(defvar model (sequential (linear 10 5)
                          (relu-layer)
                          (linear 5 2)))

;; Save weights, biases, and optional gradients
(save-checkpoint model #P"checkpoints/model.ckpt"
                 :metadata '(:epoch 10 :accuracy 0.92)
                 :include-grad t)

;; ...later or in a different session
(load-checkpoint model #P"checkpoints/model.ckpt")
```

- Pass any tensor, layer, or nested list of objects to `save-checkpoint` /
  `load-checkpoint`; the system automatically discovers underlying parameters.
- Named tensors (e.g., the default `"weights"`/`"bias"` on linear layers)
  can be loaded independently by passing just the tensors you need.
- Set `:strict nil` on `load-checkpoint` when you want to ignore extra
  parameters (useful for fine-tuning or partial weight loading).
- Include gradients with `:include-grad t` to resume advanced optimizers that
  depend on momentum/history buffers.

## Running Tests

### Using ASDF Test System

```lisp
(asdf:test-system :neural-lisp)
```

### Running from Command Line

```bash
sbcl --eval "(progn (ql:quickload :fiveam) (asdf:load-system :neural-lisp) (asdf:test-system :neural-lisp) (quit))"
```

### Running Interactively

```lisp
;; Load the test system
(asdf:load-system :neural-lisp/tests)

;; Run all tests
(neural-lisp-tests:run-neural-lisp-tests)

;; Run specific test suite
(neural-lisp-tests:test-tensors)
(neural-lisp-tests:test-operations)
(neural-lisp-tests:test-layers)
(neural-lisp-tests:test-autograd)
(neural-lisp-tests:test-optimizers)
(neural-lisp-tests:test-losses)
(neural-lisp-tests:test-training)

;; Run performance benchmarks (not run by default)
(neural-lisp-tests:test-benchmarks)
```

### Test Coverage

The test suite includes:
- **Tensor Tests**: Creation, zeros, ones, randn, requires-grad, naming
- **Operation Tests**: Addition, subtraction, multiplication, broadcasting, matrix multiplication
- **Layer Tests**: Linear layers, activation functions, sequential models
- **Autograd Tests**: Backward propagation, chain rule, gradient computation
- **Optimizer Tests**: SGD, Adam, AdamW, RMSprop, learning rate schedulers
- **Loss Tests**: MSE, MAE, Cross-Entropy, BCE, Huber, KL-Divergence
- **Training Tests**: Fit function, batch training, evaluation mode, gradient accumulation

## Running Examples

### Complete Demo

```bash
cd examples
sbcl --script complete-demo.lisp
```

Features demonstrated:
- Multiple optimizer comparison
- Learning rate schedulers
- Different loss functions
- Mixed-precision training
- Backend benchmarking

### Lisp vs Python Comparison

```bash
cd examples
sbcl --script lisp-vs-python.lisp
```

Shows Lisp advantages:
- Symbolic differentiation
- Metaprogramming
- Interactive development
- Performance

## Development

### Adding New Features

1. Add implementation to appropriate `src/` file
2. Export symbols in package definition
3. Add tests in corresponding `tests/test-*.lisp` file
4. Update this README

### Code Style

- Use descriptive function names
- Add docstrings to exported functions
- Follow Common Lisp naming conventions
- Add type declarations where beneficial
- Include tests for new functionality

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

Inspired by PyTorch's design philosophy and API conventions.
