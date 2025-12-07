;;;; variational.lisp - Variational Inference and Differentiable Sampling
;;;; Includes Gumbel-Softmax, VAE components, and reparameterizable distributions

(defpackage :variational
  (:use :common-lisp :neural-network)
  (:export ;; Sampling techniques
           #:gumbel-softmax
           #:concrete-distribution
           #:reparameterize
           #:sample-gumbel
           ;; Distributions
           #:normal-sample
           #:bernoulli-sample
           #:categorical-sample
           #:dirichlet-sample
           ;; VAE components
           #:vae-encoder
           #:vae-decoder
           #:vae
           #:kl-divergence-normal
           #:kl-divergence-categorical
           ;; Layers
           #:variational-layer
           #:stochastic-layer
           #:gaussian-layer
           #:categorical-layer
           #:make-vae-encoder
           #:make-vae-decoder
           #:make-vae
           ;; Utilities
           #:sample-from-logits
           #:log-softmax
           #:entropy
           #:cross-entropy-distributions
           ;; Random seed control
           #:set-random-seed
           #:get-random-state
           #:with-random-seed
           #:*variational-random-state*))

(in-package :variational)

;;;; ============================================================================
;;;; Random State Management
;;;; ============================================================================

(defvar *variational-random-state* (make-random-state t)
  "Global random state for reproducible sampling.
   Use SET-RANDOM-SEED to initialize with a specific seed.")

(defun set-random-seed (seed)
  "Set the random seed for reproducible sampling.
   
   Args:
     seed: Integer seed value
   
   Example:
     (set-random-seed 42)  ; All subsequent random operations will be deterministic"
  ;; Create a fresh random state from NIL (default state)
  (setf *variational-random-state* (make-random-state nil))
  ;; Seed by consuming random numbers to move the state forward deterministically
  (dotimes (i seed)
    (random 1.0d0 *variational-random-state*))
  *variational-random-state*)

(defun get-random-state ()
  "Get a copy of the current random state for later restoration."
  (make-random-state *variational-random-state*))

(defmacro with-random-seed (seed &body body)
  "Execute body with a specific random seed, then restore the previous state.
   
   Example:
     (with-random-seed 42
       (sample-gumbel :shape '(10)))"
  (let ((old-state (gensym "OLD-STATE")))
    `(let ((,old-state (get-random-state)))
       (unwind-protect
           (progn
             (set-random-seed ,seed)
             ,@body)
         (setf *variational-random-state* ,old-state)))))

;;;; ============================================================================
;;;; Gumbel-Softmax and Concrete Distribution
;;;; ============================================================================

(defun sample-gumbel (&key (shape '(1)) (eps 1e-10))
  "Sample from Gumbel(0, 1) distribution using inverse transform sampling"
  (let ((uniform (randn shape :scale 0.0)))  ; Start with array
    ;; Fill with uniform samples, then transform
    (let ((arr (tensor-data uniform)))
      (dotimes (i (array-total-size arr))
        (let ((u (loop for x = (random 1.0d0 *variational-random-state*) while (< x eps) finally (return x))))
          (setf (row-major-aref arr i)
                (- (log (- (log (coerce u 'double-float)))))))))
    uniform))

(defun gumbel-softmax (logits &key (temperature 1.0) (hard nil))
  "Gumbel-Softmax: differentiable sampling from categorical distribution
   
   Args:
     logits: tensor of unnormalized log probabilities (shape: [batch, n-classes])
     temperature: temperature parameter (lower = more discrete)
     hard: if t, return one-hot, but with gradients of soft sample
   
   Returns:
     tensor of samples from categorical distribution"
  (let* ((gumbel-noise (sample-gumbel :shape (tensor-shape logits)))
         ;; Add Gumbel noise to logits
         (noisy-logits (t+ logits gumbel-noise))
         ;; Apply softmax with temperature
         (soft-sample (softmax (t* noisy-logits 
                                   (make-tensor (list (/ 1.0 temperature))
                                              :shape '(1))))))
    (if hard
        ;; Straight-through estimator: forward uses argmax, backward uses soft
        (let* ((shape (tensor-shape soft-sample))
               (batch (first shape))
               (n-classes (second shape))
               ;; Create one-hot from argmax
               (hard-sample-data (make-array shape :element-type 'double-float 
                                            :initial-element 0.0d0)))
          ;; Find argmax for each batch element
          (dotimes (b batch)
            (let ((max-idx 0)
                  (max-val (aref (tensor-data soft-sample) b 0)))
              (loop for c from 1 below n-classes do
                (let ((val (aref (tensor-data soft-sample) b c)))
                  (when (> val max-val)
                    (setf max-val val
                          max-idx c))))
              (setf (aref hard-sample-data b max-idx) 1.0d0)))
          
          ;; Create tensor with gradient connection to soft sample
          (let ((hard-tensor (make-tensor hard-sample-data
                                        :shape shape
                                        :requires-grad (requires-grad soft-sample))))
            ;; Forward: hard one-hot, Backward: gradients from soft sample
            (when (requires-grad hard-tensor)
              (setf (grad-fn hard-tensor)
                    (lambda ()
                      (when (requires-grad soft-sample)
                        ;; Copy gradients from hard to soft (straight-through)
                        (let ((grad-hard (tensor-grad hard-tensor))
                              (grad-soft (tensor-grad soft-sample)))
                          (dotimes (i (array-total-size grad-soft))
                            (incf (row-major-aref grad-soft i)
                                  (row-major-aref grad-hard i)))))))
              (setf (children hard-tensor) (list soft-sample)))
            hard-tensor))
        soft-sample)))

(defun softmax (tensor)
  "Numerically stable softmax"
  (let* ((shape (tensor-shape tensor))
         (data (tensor-data tensor))
         (result-data (make-array shape :element-type 'double-float)))
    (cond
      ;; 2D: apply softmax to each row
      ((= (length shape) 2)
       (destructuring-bind (batch n-classes) shape
         (dotimes (b batch)
           ;; Find max for numerical stability
           (let ((max-val (aref data b 0)))
             (loop for c from 1 below n-classes do
               (setf max-val (max max-val (aref data b c))))
             ;; Compute exp and sum
             (let ((sum 0.0d0))
               (dotimes (c n-classes)
                 (let ((exp-val (exp (- (aref data b c) max-val))))
                   (setf (aref result-data b c) exp-val)
                   (incf sum exp-val)))
               ;; Normalize
               (dotimes (c n-classes)
                 (setf (aref result-data b c)
                       (/ (aref result-data b c) sum))))))))
      ;; 1D: single softmax
      ((= (length shape) 1)
       (let ((n (first shape))
             (max-val (row-major-aref data 0)))
         (loop for i from 1 below n do
           (setf max-val (max max-val (row-major-aref data i))))
         (let ((sum 0.0d0))
           (dotimes (i n)
             (let ((exp-val (exp (- (row-major-aref data i) max-val))))
               (setf (row-major-aref result-data i) exp-val)
               (incf sum exp-val)))
           (dotimes (i n)
             (setf (row-major-aref result-data i)
                   (/ (row-major-aref result-data i) sum))))))
      (t (error "Unsupported shape for softmax: ~a" shape)))
    
    (let ((result (make-tensor result-data
                              :shape shape
                              :requires-grad (requires-grad tensor))))
      (when (requires-grad result)
        (setf (grad-fn result)
              (lambda ()
                (when (requires-grad tensor)
                  (let ((grad-out (tensor-grad result))
                        (grad-in (tensor-grad tensor))
                        (out-data (tensor-data result)))
                    ;; Gradient of softmax: s * (grad - sum(s * grad))
                    (cond
                      ((= (length shape) 2)
                       (destructuring-bind (batch n-classes) shape
                         (dotimes (b batch)
                           (let ((dot-product 0.0d0))
                             (dotimes (c n-classes)
                               (incf dot-product 
                                     (* (aref out-data b c)
                                        (aref grad-out b c))))
                             (dotimes (c n-classes)
                               (incf (aref grad-in b c)
                                     (* (aref out-data b c)
                                        (- (aref grad-out b c)
                                           dot-product))))))))
                      ((= (length shape) 1)
                       (let ((n (first shape))
                             (dot-product 0.0d0))
                         (dotimes (i n)
                           (incf dot-product
                                 (* (row-major-aref out-data i)
                                    (row-major-aref grad-out i))))
                         (dotimes (i n)
                           (incf (row-major-aref grad-in i)
                                 (* (row-major-aref out-data i)
                                    (- (row-major-aref grad-out i)
                                       dot-product)))))))))))
        (setf (children result) (list tensor)))
      result)))

;;;; ============================================================================
;;;; Reparameterization Trick
;;;; ============================================================================

(defun reparameterize (mu log-var)
  "Reparameterization trick for Normal distribution: z = mu + sigma * epsilon
   
   Args:
     mu: mean tensor
     log-var: log variance tensor (for numerical stability)
   
   Returns:
     sample from N(mu, var)"
  (let* ((std (t* (make-tensor (list 0.5) :shape '(1))
                  log-var))  ; std = exp(0.5 * log_var)
         (std-exp (exp-tensor std))
         (eps (randn (tensor-shape mu) :scale 1.0)))
    (t+ mu (t* std-exp eps))))

(defun exp-tensor (tensor)
  "Element-wise exponential"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (setf (row-major-aref result-data i)
            (exp (row-major-aref data i))))
    (let ((result (make-tensor result-data
                              :shape (tensor-shape tensor)
                              :requires-grad (requires-grad tensor))))
      (when (requires-grad result)
        (setf (grad-fn result)
              (lambda ()
                (when (requires-grad tensor)
                  (let ((grad-out (tensor-grad result))
                        (grad-in (tensor-grad tensor))
                        (out-data (tensor-data result)))
                    (dotimes (i (array-total-size grad-in))
                      (incf (row-major-aref grad-in i)
                            (* (row-major-aref grad-out i)
                               (row-major-aref out-data i))))))))
        (setf (children result) (list tensor)))
      result)))

;;;; ============================================================================
;;;; Probability Distributions
;;;; ============================================================================

(defun normal-sample (mu sigma &key requires-grad)
  "Sample from Normal(mu, sigma) distribution
   
   Args:
     mu: mean (can be number or tensor)
     sigma: standard deviation (can be number or tensor)
   
   Returns:
     sample tensor"
  (let* ((mu-tensor (if (typep mu 'tensor) mu
                        (make-tensor (list (coerce mu 'double-float)) :shape '(1))))
         (sigma-tensor (if (typep sigma 'tensor) sigma
                          (make-tensor (list (coerce sigma 'double-float)) :shape '(1))))
         (eps (randn (tensor-shape mu-tensor) :scale 1.0)))
    (t+ mu-tensor (t* sigma-tensor eps))))

(defun bernoulli-sample (logits &key (temperature 1.0))
  "Sample from Bernoulli distribution using Gumbel-Softmax trick
   
   Args:
     logits: unnormalized log probabilities
     temperature: temperature for Gumbel-Softmax
   
   Returns:
     binary samples"
  (let* ((binary-logits (make-tensor 
                         (make-array (list (first (tensor-shape logits)) 2)
                                    :element-type 'double-float)
                         :shape (list (first (tensor-shape logits)) 2)
                         :requires-grad (requires-grad logits))))
    ;; Create [logits, 0] for binary choice
    (dotimes (i (first (tensor-shape logits)))
      (setf (aref (tensor-data binary-logits) i 0)
            (aref (tensor-data logits) i))
      (setf (aref (tensor-data binary-logits) i 1) 0.0d0))
    (gumbel-softmax binary-logits :temperature temperature :hard t)))

(defun categorical-sample (logits &key (temperature 1.0) (hard t))
  "Sample from categorical distribution using Gumbel-Softmax
   
   Args:
     logits: tensor of unnormalized log probabilities [batch, n-classes]
     temperature: temperature parameter
     hard: whether to return one-hot (hard) or soft samples
   
   Returns:
     samples from categorical distribution"
  (gumbel-softmax logits :temperature temperature :hard hard))

;;;; ============================================================================
;;;; KL Divergence Functions
;;;; ============================================================================

(defun kl-divergence-normal (mu log-var)
  "KL divergence between N(mu, var) and N(0, 1)
   
   KL(N(mu,var) || N(0,1)) = -0.5 * sum(1 + log(var) - mu^2 - var)
   
   Args:
     mu: mean tensor
     log-var: log variance tensor
   
   Returns:
     scalar KL divergence"
  (let* ((mu-squared (t* mu mu))
         (var (exp-tensor log-var))
         ;; KL = -0.5 * sum(1 + log_var - mu^2 - var)
         (kl-term (t- (t+ (ones (tensor-shape log-var)) log-var)
                      (t+ mu-squared var))))
    (t* (make-tensor (list -0.5d0) :shape '(1))
        (tsum kl-term))))

(defun kl-divergence-categorical (q-logits p-logits)
  "KL divergence between two categorical distributions
   
   KL(Q || P) = sum(Q * log(Q/P))
   
   Args:
     q-logits: logits for distribution Q
     p-logits: logits for distribution P
   
   Returns:
     KL divergence"
  (let* ((q (softmax q-logits))
         (p (softmax p-logits))
         (log-q (log-tensor q))
         (log-p (log-tensor p))
         (log-ratio (t- log-q log-p)))
    (tsum (t* q log-ratio))))

(defun log-tensor (tensor)
  "Element-wise logarithm with epsilon for numerical stability"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float))
         (eps 1e-10))
    (dotimes (i (array-total-size data))
      (setf (row-major-aref result-data i)
            (log (+ (row-major-aref data i) eps))))
    (let ((result (make-tensor result-data
                              :shape (tensor-shape tensor)
                              :requires-grad (requires-grad tensor))))
      (when (requires-grad result)
        (setf (grad-fn result)
              (lambda ()
                (when (requires-grad tensor)
                  (let ((grad-out (tensor-grad result))
                        (grad-in (tensor-grad tensor))
                        (in-data (tensor-data tensor)))
                    (dotimes (i (array-total-size grad-in))
                      (incf (row-major-aref grad-in i)
                            (/ (row-major-aref grad-out i)
                               (+ (row-major-aref in-data i) eps))))))))
        (setf (children result) (list tensor)))
      result)))

;;;; ============================================================================
;;;; Variational Layers
;;;; ============================================================================

(defclass gaussian-layer (layer)
  ((in-features :initarg :in-features :accessor in-features)
   (out-features :initarg :out-features :accessor out-features)
   (mu-layer :accessor mu-layer)
   (logvar-layer :accessor logvar-layer))
  (:documentation "Layer that outputs parameters of a Gaussian distribution"))

(defmethod initialize-instance :after ((layer gaussian-layer) &key)
  (with-slots (in-features out-features mu-layer logvar-layer parameters) layer
    (setf mu-layer (linear in-features out-features))
    (setf logvar-layer (linear in-features out-features))
    (setf parameters (append (layer-parameters mu-layer)
                            (layer-parameters logvar-layer)))))

(defmethod forward ((layer gaussian-layer) input)
  "Forward pass returns (mu, log-var) as a list"
  (let ((mu (forward (mu-layer layer) input))
        (log-var (forward (logvar-layer layer) input)))
    (list mu log-var)))

(defclass categorical-layer (layer)
  ((in-features :initarg :in-features :accessor in-features)
   (out-features :initarg :out-features :accessor out-features)
   (temperature :initarg :temperature :accessor temperature :initform 1.0)
   (hard :initarg :hard :accessor hard :initform nil)
   (logits-layer :accessor logits-layer))
  (:documentation "Layer that samples from categorical distribution"))

(defmethod initialize-instance :after ((layer categorical-layer) &key)
  (with-slots (in-features out-features logits-layer parameters) layer
    (setf logits-layer (linear in-features out-features))
    (setf parameters (layer-parameters logits-layer))))

(defmethod forward ((layer categorical-layer) input)
  "Forward pass returns categorical sample using Gumbel-Softmax"
  (let ((logits (forward (logits-layer layer) input)))
    (gumbel-softmax logits 
                   :temperature (temperature layer)
                   :hard (hard layer))))

(defclass stochastic-layer (layer)
  ((in-features :initarg :in-features :accessor in-features)
   (out-features :initarg :out-features :accessor out-features)
   (distribution :initarg :distribution :accessor distribution :initform :normal)
   (gaussian-layer :accessor gaussian-layer-slot)
   (sample-during-eval :initarg :sample-during-eval 
                       :accessor sample-during-eval 
                       :initform nil))
  (:documentation "Generic stochastic layer with multiple distribution types"))

(defmethod initialize-instance :after ((layer stochastic-layer) &key)
  (with-slots (in-features out-features gaussian-layer parameters) layer
    (setf gaussian-layer 
          (make-instance 'gaussian-layer
                        :in-features in-features
                        :out-features out-features))
    (setf parameters (layer-parameters gaussian-layer))))

(defmethod forward ((layer stochastic-layer) input)
  "Forward pass: return sample if training, mean if eval (unless sample-during-eval)"
  (destructuring-bind (mu log-var) (forward (gaussian-layer-slot layer) input)
    (if (or (layer-training layer) (sample-during-eval layer))
        (reparameterize mu log-var)
        mu)))

;;;; ============================================================================
;;;; VAE Components
;;;; ============================================================================

(defclass vae-encoder (layer)
  ((input-dim :initarg :input-dim :accessor input-dim)
   (hidden-dims :initarg :hidden-dims :accessor hidden-dims)
   (latent-dim :initarg :latent-dim :accessor latent-dim)
   (network :accessor encoder-network))
  (:documentation "VAE encoder network"))

(defmethod initialize-instance :after ((encoder vae-encoder) &key)
  (with-slots (input-dim hidden-dims latent-dim network parameters) encoder
    (let ((layers (list (linear input-dim (first hidden-dims)))))
      ;; Add hidden layers
      (loop for i from 0 below (1- (length hidden-dims)) do
        (push (linear (nth i hidden-dims) (nth (1+ i) hidden-dims)) layers))
      
      ;; Create Gaussian output layer
      (let ((final-gaussian (make-instance 'gaussian-layer
                                          :in-features (car (last hidden-dims))
                                          :out-features latent-dim)))
        (setf network (apply #'sequential (nreverse (cons final-gaussian layers))))
        (setf parameters (layer-parameters network))))))

(defmethod forward ((encoder vae-encoder) input)
  "Encode input to latent distribution parameters"
  (forward (encoder-network encoder) input))

(defclass vae-decoder (layer)
  ((latent-dim :initarg :latent-dim :accessor latent-dim)
   (hidden-dims :initarg :hidden-dims :accessor hidden-dims)
   (output-dim :initarg :output-dim :accessor output-dim)
   (network :accessor decoder-network))
  (:documentation "VAE decoder network"))

(defmethod initialize-instance :after ((decoder vae-decoder) &key)
  (with-slots (latent-dim hidden-dims output-dim network parameters) decoder
    (let ((layers (list (linear latent-dim (first hidden-dims)))))
      ;; Add hidden layers
      (loop for i from 0 below (1- (length hidden-dims)) do
        (push (linear (nth i hidden-dims) (nth (1+ i) hidden-dims)) layers))
      
      ;; Final output layer
      (push (linear (car (last hidden-dims)) output-dim) layers)
      
      (setf network (apply #'sequential (nreverse layers)))
      (setf parameters (layer-parameters network)))))

(defmethod forward ((decoder vae-decoder) latent)
  "Decode latent vector to reconstruction"
  (forward (decoder-network decoder) latent))

(defclass vae (layer)
  ((encoder :accessor vae-encoder-slot :initarg :encoder)
   (decoder :accessor vae-decoder-slot :initarg :decoder)
   (beta :initarg :beta :accessor beta :initform 1.0
         :documentation "Weight for KL divergence term (beta-VAE)"))
  (:documentation "Complete VAE model"))

(defmethod initialize-instance :after ((model vae) &key)
  (with-slots (encoder decoder parameters) model
    (setf parameters (append (layer-parameters encoder)
                            (layer-parameters decoder)))))

(defmethod forward ((model vae) input)
  "Forward pass: encode, sample, decode
   Returns: (reconstruction, mu, log-var, z)"
  (destructuring-bind (mu log-var) (forward (vae-encoder-slot model) input)
    (let* ((z (reparameterize mu log-var))
           (reconstruction (forward (vae-decoder-slot model) z)))
      (list reconstruction mu log-var z))))

;;;; ============================================================================
;;;; Utility Functions
;;;; ============================================================================

(defun log-softmax (tensor)
  "Compute log-softmax for numerical stability"
  (let ((softmax-result (softmax tensor)))
    (log-tensor softmax-result)))

(defun entropy (probs)
  "Compute entropy of probability distribution: H = -sum(p * log(p))"
  (let* ((log-probs (log-tensor probs))
         (neg-log-probs (t* (make-tensor (list -1.0) :shape '(1)) log-probs)))
    (tsum (t* probs neg-log-probs))))

(defun cross-entropy-distributions (q p)
  "Cross-entropy between two distributions: H(Q,P) = -sum(Q * log(P))"
  (let ((log-p (log-tensor p)))
    (t* (make-tensor (list -1.0) :shape '(1))
        (tsum (t* q log-p)))))

(defun sample-from-logits (logits &key (temperature 1.0) (n-samples 1))
  "Sample multiple times from categorical distribution defined by logits"
  (loop repeat n-samples
        collect (categorical-sample logits 
                                   :temperature temperature 
                                   :hard t)))

;;;; ============================================================================
;;;; Constructor Functions
;;;; ============================================================================

(defun make-vae-encoder (input-dim hidden-dims latent-dim)
  "Create a VAE encoder"
  (make-instance 'vae-encoder
                 :input-dim input-dim
                 :hidden-dims hidden-dims
                 :latent-dim latent-dim))

(defun make-vae-decoder (latent-dim hidden-dims output-dim)
  "Create a VAE decoder"
  (make-instance 'vae-decoder
                 :latent-dim latent-dim
                 :hidden-dims hidden-dims
                 :output-dim output-dim))

(defun make-vae (input-dim hidden-dims latent-dim &key (beta 1.0))
  "Create a complete VAE model
   
   Args:
     input-dim: dimension of input data
     hidden-dims: list of hidden layer dimensions
     latent-dim: dimension of latent space
     beta: weight for KL term (beta-VAE)
   
   Returns:
     VAE model"
  (let ((encoder (make-vae-encoder input-dim hidden-dims latent-dim))
        (decoder (make-vae-decoder latent-dim (reverse hidden-dims) input-dim)))
    (make-instance 'vae
                   :encoder encoder
                   :decoder decoder
                   :beta beta)))
