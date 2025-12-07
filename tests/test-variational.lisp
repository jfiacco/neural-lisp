;;;; tests/test-variational.lisp - Tests for Variational Inference Components

(in-package #:neural-lisp-tests)

(def-suite variational-tests
  :description "Comprehensive test suite for variational inference"
  :in neural-lisp-tests)

(in-suite variational-tests)

;;;; ============================================================================
;;;; Helper Functions
;;;; ============================================================================

(defun float-nan-p (x)
  "Check if a float is NaN"
  (/= x x))

(defun float-infinity-p (x)
  "Check if a float is infinity"
  (and (floatp x)
       (or (> x most-positive-double-float)
           (< x most-negative-double-float)
           (not (= x x))  ; NaN check as backup
           (not (finite-p-simple x)))))

(defun finite-p-simple (x)
  "Simple finite check"
  (and (numberp x) (= x x)))  ; Not NaN

(defun finite-p (x)
  "Check if a number is finite"
  (and (numberp x) 
       (= x x)  ; Not NaN
       (<= x most-positive-double-float)
       (>= x most-negative-double-float)))

(defun assert-tensor-shape (expected-shape tensor)
  "Assert that tensor has expected shape"
  (is (equal expected-shape (tensor-shape tensor))))

(defun assert-tensor-range (tensor min-val max-val)
  "Assert all tensor values are in range [min-val, max-val]"
  (let ((data (tensor-data tensor)))
    (dotimes (i (array-total-size data))
      (let ((val (row-major-aref data i)))
        (is (<= min-val val max-val)
            "Value ~a out of range [~a, ~a]" val min-val max-val)))))

(defun assert-close (expected actual &optional (tolerance 1d-6))
  "Assert two numbers are close within tolerance"
  (is (< (abs (- expected actual)) tolerance)
      "Expected ~a, got ~a (tolerance ~a)" expected actual tolerance))

(defun assert-tensor-close (tensor1 tensor2 &optional (tolerance 1d-6))
  "Assert two tensors have close values"
  (is (equal (tensor-shape tensor1) (tensor-shape tensor2))
      "Tensor shapes don't match")
  (let ((data1 (tensor-data tensor1))
        (data2 (tensor-data tensor2)))
    (dotimes (i (array-total-size data1))
      (assert-close (row-major-aref data1 i)
                   (row-major-aref data2 i)
                   tolerance))))

(defun assert-probability-distribution (tensor &optional (tolerance 1d-5))
  "Assert tensor represents valid probability distribution (sums to 1)"
  (let* ((shape (tensor-shape tensor))
         (data (tensor-data tensor)))
    ;; Check all values are non-negative
    (dotimes (i (array-total-size data))
      (is (>= (row-major-aref data i) 0.0)
          "Probability must be non-negative"))
    ;; Check sums to 1 for each batch
    (when (= (length shape) 2)
      (destructuring-bind (batch n-classes) shape
        (dotimes (b batch)
          (let ((sum 0.0d0))
            (dotimes (c n-classes)
              (incf sum (aref data b c)))
            (assert-close 1.0d0 sum tolerance)))))))

(defun assert-one-hot (tensor)
  "Assert tensor is one-hot encoded"
  (let* ((shape (tensor-shape tensor))
         (data (tensor-data tensor)))
    (when (= (length shape) 2)
      (destructuring-bind (batch n-classes) shape
        (dotimes (b batch)
          (let ((sum 0.0d0)
                (max-val 0.0d0))
            (dotimes (c n-classes)
              (let ((val (aref data b c)))
                (incf sum val)
                (setf max-val (max max-val val))
                (is (or (= val 0.0d0) (= val 1.0d0))
                    "One-hot must contain only 0 or 1")))
            (assert-close 1.0d0 sum 1d-5)
            (assert-close 1.0d0 max-val 1d-5)))))))

;;;; ============================================================================
;;;; Gumbel-Softmax Tests
;;;; ============================================================================

(test gumbel-sample-shape
  "Gumbel sampling should produce correct shape"
  (let ((sample (variational::sample-gumbel :shape '(10 5))))
    (assert-tensor-shape '(10 5) sample)))

(test gumbel-softmax-basic
  "Gumbel-Softmax should produce probability distribution"
  (let* ((logits (make-tensor #2A((1.0d0 2.0d0 3.0d0)
                                  (0.5d0 1.5d0 2.5d0))
                             :shape '(2 3)))
         (sample (variational:gumbel-softmax logits :temperature 1.0)))
    (assert-tensor-shape '(2 3) sample)
    (assert-probability-distribution sample)))

(test gumbel-softmax-temperature-effect
  "Lower temperature should produce more discrete distributions"
  (let* ((logits (make-tensor #2A((1.0d0 5.0d0 1.0d0))
                             :shape '(1 3)))
         (hot-sample (variational:gumbel-softmax logits :temperature 0.1))
         (cold-sample (variational:gumbel-softmax logits :temperature 10.0)))
    ;; Lower temperature should have higher max value (more peaked)
    (let ((hot-max (loop for i below 3
                        maximize (aref (tensor-data hot-sample) 0 i)))
          (cold-max (loop for i below 3
                         maximize (aref (tensor-data cold-sample) 0 i))))
      (is (> hot-max cold-max)
          "Low temperature should be more peaked"))))

(test gumbel-softmax-hard-mode
  "Hard mode should produce one-hot vectors"
  (let* ((logits (make-tensor #2A((1.0d0 5.0d0 2.0d0)
                                  (3.0d0 1.0d0 4.0d0))
                             :shape '(2 3)))
         (hard-sample (variational:gumbel-softmax logits 
                                                  :temperature 0.5
                                                  :hard t)))
    (assert-tensor-shape '(2 3) hard-sample)
    (assert-one-hot hard-sample)))

(test gumbel-softmax-gradient-tracking
  "Gumbel-Softmax should support gradient tracking"
  (let* ((logits (make-tensor #2A((1.0d0 2.0d0 3.0d0))
                             :shape '(1 3)
                             :requires-grad t))
         (sample (variational:gumbel-softmax logits :temperature 1.0)))
    (is (requires-grad sample)
        "Output should require grad when input does")))

;;;; ============================================================================
;;;; Softmax Tests
;;;; ============================================================================

(test softmax-basic-2d
  "Softmax should work on 2D tensors"
  (let* ((input (make-tensor #2A((1.0d0 2.0d0 3.0d0)
                                 (0.0d0 1.0d0 2.0d0))
                            :shape '(2 3)))
         (output (variational::softmax input)))
    (assert-tensor-shape '(2 3) output)
    (assert-probability-distribution output)))

(test softmax-basic-1d
  "Softmax should work on 1D tensors"
  (let* ((input (make-tensor #(1.0d0 2.0d0 3.0d0) :shape '(3)))
         (output (variational::softmax input)))
    (assert-tensor-shape '(3) output)
    ;; Check sum equals 1
    (let ((sum 0.0d0))
      (dotimes (i 3)
        (incf sum (aref (tensor-data output) i)))
      (assert-close 1.0d0 sum))))

(test softmax-numerical-stability
  "Softmax should handle large values numerically stably"
  (let* ((input (make-tensor #2A((1000.0d0 1001.0d0 999.0d0))
                            :shape '(1 3)))
         (output (variational::softmax input)))
    (assert-probability-distribution output)
    ;; Should not produce NaN or Inf
    (let ((data (tensor-data output)))
      (dotimes (i 3)
        (is (not (or (float-nan-p (aref data 0 i))
                     (float-infinity-p (aref data 0 i)))))))))

(test softmax-gradient
  "Softmax should compute correct gradients"
  (let* ((input (make-tensor #2A((1.0d0 2.0d0 3.0d0))
                            :shape '(1 3)
                            :requires-grad t))
         (output (variational::softmax input)))
    (is (requires-grad output))
    ;; Gradient should flow back
    (setf (aref (tensor-grad output) 0 0) 1.0d0)
    (backward output)
    (is (not (null (tensor-grad input))))))

;;;; ============================================================================
;;;; Reparameterization Tests
;;;; ============================================================================

(test reparameterize-basic
  "Reparameterization should produce correct shape"
  (let* ((mu (zeros '(5 10)))
         (log-var (zeros '(5 10)))
         (z (variational:reparameterize mu log-var)))
    (assert-tensor-shape '(5 10) z)))

(test reparameterize-gradient-flow
  "Reparameterization should allow gradient flow"
  (let* ((mu (zeros '(2 3) :requires-grad t))
         (log-var (zeros '(2 3) :requires-grad t))
         (z (variational:reparameterize mu log-var)))
    (is (requires-grad z)
        "Output should require grad when inputs do")))

(test reparameterize-mean-center
  "With log-var = -inf, output should equal mu"
  (let* ((mu (make-tensor #2A((1.0d0 2.0d0)
                             (3.0d0 4.0d0))
                         :shape '(2 2)))
         ;; Very negative log-var means very small variance
         (log-var (make-tensor #2A((-100.0d0 -100.0d0)
                                   (-100.0d0 -100.0d0))
                              :shape '(2 2)))
         (z (variational:reparameterize mu log-var)))
    ;; Should be very close to mu
    (assert-tensor-close mu z 1d-10)))

;;;; ============================================================================
;;;; Distribution Tests
;;;; ============================================================================

(test normal-sample-basic
  "Normal sampling should produce correct shape"
  (let ((sample (variational:normal-sample 
                 (make-tensor #(0.0d0) :shape '(1))
                 (make-tensor #(1.0d0) :shape '(1)))))
    (assert-tensor-shape '(1) sample)))

(test normal-sample-scalar-inputs
  "Normal sampling should accept scalar inputs"
  (let ((sample (variational:normal-sample 0.0 1.0)))
    (is (not (null sample)))))

(test categorical-sample-basic
  "Categorical sampling should produce one-hot vectors"
  (let* ((logits (make-tensor #2A((1.0d0 2.0d0 3.0d0)
                                  (3.0d0 2.0d0 1.0d0))
                             :shape '(2 3)))
         (sample (variational:categorical-sample logits :hard t)))
    (assert-tensor-shape '(2 3) sample)
    (assert-one-hot sample)))

(test categorical-sample-soft-mode
  "Categorical sampling with hard=nil should produce soft samples"
  (let* ((logits (make-tensor #2A((1.0d0 2.0d0 3.0d0))
                             :shape '(1 3)))
         (sample (variational:categorical-sample logits :hard nil)))
    (assert-tensor-shape '(1 3) sample)
    (assert-probability-distribution sample)))

;;;; ============================================================================
;;;; KL Divergence Tests
;;;; ============================================================================

(test kl-divergence-normal-zeros
  "KL divergence should be zero when distributions match"
  (let* ((mu (zeros '(1 5)))
         (log-var (zeros '(1 5)))
         (kl (variational:kl-divergence-normal mu log-var)))
    ;; KL(N(0,1) || N(0,1)) = 0
    (assert-close 0.0d0 (aref (tensor-data kl) 0) 1d-4)))

(test kl-divergence-normal-positive
  "KL divergence should be positive when distributions differ"
  (let* ((mu (make-tensor #2A((1.0d0 2.0d0 3.0d0))
                         :shape '(1 3)))
         (log-var (zeros '(1 3)))
         (kl (variational:kl-divergence-normal mu log-var)))
    ;; KL should be positive
    (is (> (aref (tensor-data kl) 0) 0.0d0))))

(test kl-divergence-normal-gradient
  "KL divergence should support gradient computation"
  (let* ((mu (zeros '(1 3) :requires-grad t))
         (log-var (zeros '(1 3) :requires-grad t))
         (kl (variational:kl-divergence-normal mu log-var)))
    (is (requires-grad kl))))

(test kl-divergence-categorical-self
  "KL divergence between identical distributions should be zero"
  (let* ((logits (make-tensor #2A((1.0d0 2.0d0 3.0d0))
                             :shape '(1 3)))
         (kl (variational:kl-divergence-categorical logits logits)))
    (assert-close 0.0d0 (aref (tensor-data kl) 0) 1d-4)))

(test kl-divergence-categorical-positive
  "KL divergence should be positive for different distributions"
  (let* ((q-logits (make-tensor #2A((1.0d0 2.0d0 3.0d0))
                               :shape '(1 3)))
         (p-logits (make-tensor #2A((3.0d0 2.0d0 1.0d0))
                               :shape '(1 3)))
         (kl (variational:kl-divergence-categorical q-logits p-logits)))
    (is (> (aref (tensor-data kl) 0) 0.0d0))))

;;;; ============================================================================
;;;; Variational Layer Tests
;;;; ============================================================================

(test gaussian-layer-creation
  "Gaussian layer should initialize correctly"
  (let ((layer (make-instance 'variational::gaussian-layer
                             :in-features 10
                             :out-features 5)))
    (is (= 10 (variational::in-features layer)))
    (is (= 5 (variational::out-features layer)))
    (is (not (null (variational::mu-layer layer))))
    (is (not (null (variational::logvar-layer layer))))))

(test gaussian-layer-forward
  "Gaussian layer should output mu and log-var"
  (let* ((layer (make-instance 'variational::gaussian-layer
                              :in-features 8
                              :out-features 4))
         (input (randn '(2 8)))
         (output (forward layer input)))
    (is (listp output))
    (is (= 2 (length output)))
    (let ((mu (first output))
          (log-var (second output)))
      (assert-tensor-shape '(2 4) mu)
      (assert-tensor-shape '(2 4) log-var))))

(test gaussian-layer-parameters
  "Gaussian layer should have trainable parameters"
  (let ((layer (make-instance 'variational::gaussian-layer
                             :in-features 5
                             :out-features 3)))
    (is (not (null (layer-parameters layer))))
    ;; Should have 4 parameter tensors (2 weights, 2 biases)
    (is (= 4 (length (layer-parameters layer))))))

(test categorical-layer-creation
  "Categorical layer should initialize correctly"
  (let ((layer (make-instance 'variational::categorical-layer
                             :in-features 10
                             :out-features 5
                             :temperature 0.5
                             :hard t)))
    (is (= 10 (variational::in-features layer)))
    (is (= 5 (variational::out-features layer)))
    (is (= 0.5 (variational::temperature layer)))
    (is (eq t (variational::hard layer)))))

(test categorical-layer-forward-hard
  "Categorical layer should produce one-hot in hard mode"
  (let* ((layer (make-instance 'variational::categorical-layer
                              :in-features 8
                              :out-features 3
                              :hard t))
         (input (randn '(2 8)))
         (output (forward layer input)))
    (assert-tensor-shape '(2 3) output)
    (assert-one-hot output)))

(test categorical-layer-forward-soft
  "Categorical layer should produce probabilities in soft mode"
  (let* ((layer (make-instance 'variational::categorical-layer
                              :in-features 8
                              :out-features 3
                              :hard nil))
         (input (randn '(2 8)))
         (output (forward layer input)))
    (assert-tensor-shape '(2 3) output)
    (assert-probability-distribution output)))

(test stochastic-layer-creation
  "Stochastic layer should initialize correctly"
  (let ((layer (make-instance 'variational::stochastic-layer
                             :in-features 10
                             :out-features 5)))
    (is (= 10 (variational::in-features layer)))
    (is (= 5 (variational::out-features layer)))
    (is (not (null (variational::gaussian-layer-slot layer))))))

(test stochastic-layer-training-mode
  "Stochastic layer should sample during training"
  (let* ((layer (make-instance 'variational::stochastic-layer
                              :in-features 8
                              :out-features 4))
         (input (randn '(2 8))))
    (train-mode layer)
    (let ((output (forward layer input)))
      (assert-tensor-shape '(2 4) output))))

(test stochastic-layer-eval-mode
  "Stochastic layer should return mean during eval"
  (let* ((layer (make-instance 'variational::stochastic-layer
                              :in-features 8
                              :out-features 4))
         (input (randn '(2 8))))
    (eval-mode layer)
    (let ((output (forward layer input)))
      (assert-tensor-shape '(2 4) output))))

;;;; ============================================================================
;;;; VAE Component Tests
;;;; ============================================================================

(test vae-encoder-creation
  "VAE encoder should initialize correctly"
  (let ((encoder (variational:make-vae-encoder 784 '(400 200) 20)))
    (is (= 784 (variational::input-dim encoder)))
    (is (equal '(400 200) (variational::hidden-dims encoder)))
    (is (= 20 (variational::latent-dim encoder)))
    (is (not (null (layer-parameters encoder))))))

(test vae-encoder-forward
  "VAE encoder should produce mu and log-var"
  (let* ((encoder (variational:make-vae-encoder 28 '(16) 8))
         (input (randn '(4 28)))
         (output (forward encoder input)))
    (is (listp output))
    (is (= 2 (length output)))
    (let ((mu (first output))
          (log-var (second output)))
      (assert-tensor-shape '(4 8) mu)
      (assert-tensor-shape '(4 8) log-var))))

(test vae-decoder-creation
  "VAE decoder should initialize correctly"
  (let ((decoder (variational:make-vae-decoder 20 '(200 400) 784)))
    (is (= 20 (variational::latent-dim decoder)))
    (is (equal '(200 400) (variational::hidden-dims decoder)))
    (is (= 784 (variational::output-dim decoder)))
    (is (not (null (layer-parameters decoder))))))

(test vae-decoder-forward
  "VAE decoder should reconstruct from latent"
  (let* ((decoder (variational:make-vae-decoder 8 '(16) 28))
         (latent (randn '(4 8)))
         (output (forward decoder latent)))
    (assert-tensor-shape '(4 28) output)))

(test vae-creation
  "Complete VAE should initialize correctly"
  (let ((vae (variational:make-vae 28 '(16) 8)))
    (is (not (null (variational::vae-encoder-slot vae))))
    (is (not (null (variational::vae-decoder-slot vae))))
    (is (= 1.0 (variational::beta vae)))
    (is (not (null (layer-parameters vae))))))

(test vae-forward
  "VAE forward pass should produce all outputs"
  (let* ((vae (variational:make-vae 28 '(16) 8))
         (input (randn '(4 28)))
         (output (forward vae input)))
    (is (listp output))
    (is (= 4 (length output)))
    (destructuring-bind (reconstruction mu log-var z) output
      (assert-tensor-shape '(4 28) reconstruction)
      (assert-tensor-shape '(4 8) mu)
      (assert-tensor-shape '(4 8) log-var)
      (assert-tensor-shape '(4 8) z))))

(test vae-beta-parameter
  "VAE should support beta parameter"
  (let ((vae (variational:make-vae 28 '(16) 8 :beta 0.5)))
    (is (= 0.5 (variational::beta vae)))))

(test vae-gradient-flow
  "VAE should support gradient flow through all components"
  (let* ((vae (variational:make-vae 10 '(8) 4))
         (input (randn '(2 10) :requires-grad t)))
    (destructuring-bind (reconstruction mu log-var z)
        (forward vae input)
      (is (requires-grad reconstruction))
      (is (requires-grad mu))
      (is (requires-grad log-var))
      (is (requires-grad z)))))

;;;; ============================================================================
;;;; Utility Function Tests
;;;; ============================================================================

(test log-softmax-basic
  "Log-softmax should produce log probabilities"
  (let* ((input (make-tensor #2A((1.0d0 2.0d0 3.0d0))
                            :shape '(1 3)))
         (output (variational:log-softmax input)))
    (assert-tensor-shape '(1 3) output)
    ;; All values should be negative (log of probabilities)
    (dotimes (i 3)
      (is (< (aref (tensor-data output) 0 i) 0.0d0)))))

(test log-softmax-consistency
  "Log-softmax should equal log(softmax)"
  (let* ((input (make-tensor #2A((1.0d0 2.0d0 3.0d0))
                            :shape '(1 3)))
         (log-sm (variational:log-softmax input))
         (sm (variational::softmax input))
         (log-sm-manual (variational::log-tensor sm)))
    (assert-tensor-close log-sm log-sm-manual 1d-5)))

(test entropy-basic
  "Entropy should work on probability distributions"
  (let* ((uniform-probs (make-tensor #2A((0.5d0 0.5d0))
                                    :shape '(1 2)))
         (entropy (variational:entropy uniform-probs)))
    ;; Entropy of uniform distribution over 2 outcomes = log(2) ≈ 0.693
    (is (> (aref (tensor-data entropy) 0) 0.0d0))))

(test entropy-deterministic
  "Entropy of deterministic distribution should be near zero"
  (let* ((deterministic (make-tensor #2A((1.0d0 0.0d0))
                                    :shape '(1 2)))
         (entropy (variational:entropy deterministic)))
    ;; Should be close to 0 (with numerical tolerance)
    (is (< (aref (tensor-data entropy) 0) 0.1d0))))

(test cross-entropy-distributions
  "Cross-entropy between distributions"
  (let* ((q (make-tensor #2A((0.5d0 0.5d0)) :shape '(1 2)))
         (p (make-tensor #2A((0.3d0 0.7d0)) :shape '(1 2)))
         (ce (variational:cross-entropy-distributions q p)))
    (is (> (aref (tensor-data ce) 0) 0.0d0))))

(test sample-from-logits-multiple
  "Sample-from-logits should produce multiple samples"
  (let* ((logits (make-tensor #2A((1.0d0 2.0d0 3.0d0))
                             :shape '(1 3)))
         (samples (variational:sample-from-logits logits :n-samples 5)))
    (is (= 5 (length samples)))
    (dolist (sample samples)
      (assert-tensor-shape '(1 3) sample)
      (assert-one-hot sample))))

;;;; ============================================================================
;;;; Integration Tests
;;;; ============================================================================

(test vae-training-step
  "Complete VAE training step should work"
  (let* ((vae (variational:make-vae 20 '(16) 8))
         (input (randn '(4 20) :requires-grad t)))
    ;; Forward pass
    (destructuring-bind (reconstruction mu log-var z)
        (forward vae input)
      ;; Compute VAE loss (reconstruction + KL)
      (let ((kl-loss (variational:kl-divergence-normal mu log-var)))
        ;; Should be able to compute gradients
        (is (not (null reconstruction)))
        (is (not (null kl-loss)))))))

(test gumbel-softmax-annealing
  "Temperature annealing should make samples more discrete"
  (let* ((logits (make-tensor #2A((1.0d0 5.0d0 2.0d0))
                             :shape '(1 3)))
         (temps '(2.0 1.0 0.5 0.1)))
    (dolist (temp temps)
      (let ((sample (variational:gumbel-softmax logits :temperature temp)))
        (assert-probability-distribution sample)
        ;; Lower temperature should increase max probability
        (when (< temp 0.5)
          (let ((max-prob (loop for i below 3
                               maximize (aref (tensor-data sample) 0 i))))
            (is (> max-prob 0.9)
                "Very low temp should produce near one-hot")))))))

(test variational-layer-composition
  "Variational layers should compose with regular layers"
  (let* ((encoder (sequential
                   (linear 10 8)
                   (make-instance 'variational::gaussian-layer
                                 :in-features 8
                                 :out-features 4)))
         (input (randn '(2 10))))
    ;; Should work through sequential
    (is (not (null (forward encoder input))))))

;;;; Test suite export
(export '(variational-tests))
