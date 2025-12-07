;;;; neural-lisp.asd - System Definition

(asdf:defsystem #:neural-lisp
  :description "A neural network library showcasing Lisp's unique strengths"
  :author "James"
  :version "2.0.0"
  :license "MIT"
  :depends-on (#:cffi)
  :serial t
  :components ((:module "src"
                :components
                ((:file "cuda-bindings")
                 (:file "neural-network")
                 (:file "activations")
                 (:file "backend")
                 (:file "data-loader")
                 (:file "optimizers")
                 (:file "losses")
                 (:file "lisp-idioms")
                 (:file "integration")
                 (:file "recurrent")
                 (:file "transformer")
                 (:file "crf")
                 (:file "state-space")
                 (:file "normalization")
                 (:file "convolution")
                 (:file "residual")
                 (:file "variational")
                 (:file "embedding"))))
  :in-order-to ((test-op (test-op #:neural-lisp/tests))))

(asdf:defsystem #:neural-lisp/tests
  :description "Test suite for neural-lisp"
  :depends-on (#:neural-lisp #:fiveam)
  :serial t
  :components ((:module "tests"
                :components
                ((:file "package")
                 (:file "test-tensor")
                 (:file "test-operations")
                 (:file "test-layers")
                 (:file "test-autograd")
                 (:file "test-optimizers")
                 (:file "test-losses")
                 (:file "test-training")
                 (:file "test-backend")
                 (:file "test-data-loader")
                 (:file "test-normalization")
                 (:file "test-convolution")
                 (:file "test-residual")
                 (:file "test-activations")
                 (:file "test-variational")
                 (:file "test-embedding")
                 (:file "test-checkpoint")
                 (:file "run-tests")))))
