;;;; tests/package.lisp - Test Package Definition

(defpackage #:neural-lisp-tests
  (:use #:common-lisp #:fiveam #:neural-network)
  (:shadowing-import-from #:neural-tensor-optimizers #:step #:sgd)
  (:shadowing-import-from #:neural-tensor-losses #:step-scheduler)
  (:import-from #:neural-tensor-activations
                #:relu
                #:sigmoid
                #:tanh-activation
                #:softmax
                #:log-softmax
                #:leaky-relu
                #:parametric-relu
                #:elu
                #:selu
                #:gelu
                #:swish
                #:mish
                #:hardswish
                #:hard-sigmoid
                #:softsign
                #:softplus
                #:relu6
                #:celu
                #:silu
                #:hard-tanh
                #:tanh-shrink
                #:soft-shrink
                #:hard-shrink
                #:glu
                #:geglu
                #:swiglu
                #:relu-layer
                #:leaky-relu-layer
                #:elu-layer
                #:selu-layer
                #:gelu-layer
                #:swish-layer
                #:mish-layer
                #:sigmoid-layer
                #:tanh-layer
                #:softmax-layer
                #:log-softmax-layer
                #:hardswish-layer
                #:hard-sigmoid-layer
                #:softsign-layer
                #:softplus-layer
                #:relu6-layer
                #:glu-layer
                #:geglu-layer
                #:swiglu-layer)
  (:import-from #:neural-tensor-backend
                #:*backend*
                #:use-backend
                #:with-backend)
  (:import-from #:neural-data-loader
                #:dataset
                #:sequence-dataset
                #:text-dataset
                #:tensor-dataset
                #:make-data-loader
                #:get-batch
                #:reset-loader
                #:dataset-length
                #:num-batches
                #:get-item
                #:pad-sequences
                #:create-attention-mask
                #:normalize-batch
                #:shuffle-dataset
                #:max-length
                #:vocab-size
                #:batch-size)
  (:import-from #:neural-tensor-optimizers
                #:adam
                #:adamw
                #:rmsprop
                #:zero-grad
                #:get-lr)
  (:import-from #:neural-tensor-losses
                #:mse-loss
                #:mae-loss
                #:cross-entropy-loss
                #:binary-cross-entropy
                #:huber-loss
                #:smooth-l1-loss
                #:kl-divergence
                #:nll-loss
                #:step-lr-scheduler
                #:cosine-annealing-scheduler
                #:get-last-lr)
  (:import-from #:normalization
                #:layer-norm
                #:batch-norm
                #:layer-norm-layer
                #:batch-norm-layer
                #:norm-gamma
                #:norm-beta
                #:running-mean
                #:running-var
                #:momentum
                #:eps)
  (:import-from #:neural-tensor-complete
                #:fit
                #:train-epoch
                #:evaluate)
  (:import-from #:variational
                #:gumbel-softmax
                #:reparameterize
                #:normal-sample
                #:categorical-sample
                #:kl-divergence-normal
                #:kl-divergence-categorical
                #:make-vae-encoder
                #:make-vae-decoder
                #:make-vae
                #:entropy
                #:cross-entropy-distributions
                #:sample-from-logits)
  (:import-from #:neural-network.embedding
                #:word-embedding
                #:subword-embedding
                #:positional-encoding
                #:sinusoidal-positional-encoding
                #:learned-positional-encoding
                #:embedding-with-oov
                #:numerical-embedding
                #:byte-embedding
                #:combined-embedding
                #:emb-vocab-size
                #:embedding-dim
                #:embeddings
                #:emb-max-length
                #:oov-index
                #:pad-index
                #:lookup
                #:get-embedding
                #:max-subwords
                #:aggregation
                #:encodings
                #:position-embeddings
                #:oov-strategy
                #:input-dim
                #:projection
                #:byte-vocab-size
                #:max-bytes
                #:token-embedding
                #:padding-idx)
  (:import-from #:uiop
                #:temporary-directory))

(in-package #:neural-lisp-tests)

(def-suite neural-lisp-tests
  :description "Master test suite for neural-lisp")

(in-suite neural-lisp-tests)
