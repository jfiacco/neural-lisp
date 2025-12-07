;;;; Neural Tensor Library - Activation Functions
;;;; Modern activation functions for deep learning
;;;; Including ReLU variants, sigmoid variants, and advanced activations

(defpackage :neural-tensor-activations
  (:use :common-lisp)
  (:import-from :neural-network
                #:tensor
                #:tensor-data
                #:tensor-shape
                #:tensor-grad
                #:make-tensor
                #:requires-grad
                #:grad-fn
                #:children
                #:layer
                #:forward)
  (:export ;; Classic activations
           #:relu
           #:sigmoid
           #:tanh-activation
           #:softmax
           #:log-softmax
           ;; ReLU variants
           #:leaky-relu
           #:parametric-relu
           #:elu
           #:selu
           #:gelu
           #:swish
           #:mish
           #:hardswish
           #:hard-sigmoid
           ;; Advanced activations
           #:glu
           #:geglu
           #:swiglu
           #:softsign
           #:softplus
           #:maxout
           #:relu6
           #:celu
           #:silu
           #:hard-tanh
           #:tanh-shrink
           #:soft-shrink
           #:hard-shrink
           ;; Activation layers
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
           #:swiglu-layer))

(in-package :neural-tensor-activations)

;;;; ============================================================================
;;;; Shared Utilities
;;;; ============================================================================

(defun maybe-attach-grad (result tensor grad-fn &optional (return-value result))
  "Attach gradient callback and graph linkage when required.
RESULT is the output tensor, TENSOR is the input dependency, and GRAD-FN is a
lambda to compute gradients. Optionally RETURN-VALUE can override the returned
tensor (defaults to RESULT)."
  (when (and grad-fn (requires-grad result))
    (setf (grad-fn result) grad-fn)
    (setf (children result) (list tensor)))
  return-value)

;;;; ============================================================================
;;;; Classic Activation Functions
;;;; ============================================================================

(defun relu (tensor)
  "Rectified Linear Unit: ReLU(x) = max(0, x)
   The most widely used activation function"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (setf (row-major-aref result-data i)
            (max 0.0d0 (row-major-aref data i))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (when (> (row-major-aref data i) 0.0d0)
                 (incf (row-major-aref grad i)
                       (row-major-aref grad-out i))))))))
      result)))

(defun sigmoid (tensor)
  "Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
   Squashes values to (0, 1) range"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (setf (row-major-aref result-data i)
            (/ 1.0d0 (+ 1.0d0 (exp (- (row-major-aref data i)))))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (let ((sig-val (row-major-aref result-data i)))
                 (incf (row-major-aref grad i)
                       (* (row-major-aref grad-out i)
                          sig-val
                          (- 1.0d0 sig-val)))))))))
      result)))

(defun tanh-activation (tensor)
  "Hyperbolic tangent activation: tanh(x)
   Squashes values to (-1, 1) range"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (setf (row-major-aref result-data i)
            (tanh (row-major-aref data i))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (let ((tanh-val (row-major-aref result-data i)))
                 (incf (row-major-aref grad i)
                       (* (row-major-aref grad-out i)
                          (- 1.0d0 (* tanh-val tanh-val))))))))))
      result)))

(defun softmax (tensor &key (dim -1))
  "Softmax activation: softmax(x_i) = exp(x_i) / Σ(exp(x_j))
   Converts logits to probabilities"
  (declare (ignore dim)) ; Simplified: operates on last dimension
  (let* ((data (tensor-data tensor))
         (shape (tensor-shape tensor))
         (result-data (make-array shape :element-type 'double-float)))
    ;; Find max for numerical stability
    (let ((max-val (loop for i below (array-total-size data)
                        maximize (row-major-aref data i))))
      ;; Compute exp(x - max)
      (dotimes (i (array-total-size data))
        (setf (row-major-aref result-data i)
              (exp (- (row-major-aref data i) max-val))))
      ;; Normalize
      (let ((sum (loop for i below (array-total-size data)
                      sum (row-major-aref result-data i))))
        (dotimes (i (array-total-size data))
          (setf (row-major-aref result-data i)
                (/ (row-major-aref result-data i) sum)))))
    (make-tensor result-data
                 :shape shape
                 :requires-grad (requires-grad tensor))))

(defun log-softmax (tensor &key (dim -1))
  "Log-Softmax activation: log(softmax(x))
   Numerically stable alternative to log(softmax(x))"
  (declare (ignore dim))
  (let* ((data (tensor-data tensor))
         (shape (tensor-shape tensor))
         (result-data (make-array shape :element-type 'double-float)))
    ;; Find max for numerical stability
    (let ((max-val (loop for i below (array-total-size data)
                        maximize (row-major-aref data i))))
      ;; Compute log-sum-exp
      (let ((log-sum-exp (log (loop for i below (array-total-size data)
                                   sum (exp (- (row-major-aref data i) max-val))))))
        (dotimes (i (array-total-size data))
          (setf (row-major-aref result-data i)
                (- (row-major-aref data i) max-val log-sum-exp)))))
    (make-tensor result-data
                 :shape shape
                 :requires-grad (requires-grad tensor))))

;;;; ============================================================================
;;;; ReLU Variants
;;;; ============================================================================

(defun leaky-relu (tensor &optional (negative-slope 0.01d0))
  "Leaky ReLU: LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
   Allows small negative values to pass through"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let ((x (row-major-aref data i)))
        (setf (row-major-aref result-data i)
              (if (> x 0.0d0)
                  x
                  (* negative-slope x)))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (let ((x (row-major-aref data i)))
                 (incf (row-major-aref grad i)
                       (* (row-major-aref grad-out i)
                          (if (> x 0.0d0) 1.0d0 negative-slope)))))))))
      result)))

(defun parametric-relu (tensor alpha-tensor)
  "Parametric ReLU: PReLU(x) = max(0, x) + α * min(0, x)
   Where α is learned per channel"
  (declare (ignore alpha-tensor))
  ;; Simplified implementation - would need proper parameter handling
  (leaky-relu tensor 0.01d0))

(defun elu (tensor &optional (alpha 1.0d0))
  "Exponential Linear Unit: ELU(x) = x if x > 0, α(e^x - 1) otherwise
   Smooth activation with negative values"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let ((x (row-major-aref data i)))
        (setf (row-major-aref result-data i)
              (if (> x 0.0d0)
                  x
                  (* alpha (- (exp x) 1.0d0))))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (let ((x (row-major-aref data i)))
                 (incf (row-major-aref grad i)
                       (* (row-major-aref grad-out i)
                          (if (> x 0.0d0)
                              1.0d0
                              (* alpha (exp x)))))))))))
      result)))

(defun selu (tensor)
  "Scaled Exponential Linear Unit: SELU(x) = λ * (x if x > 0, α(e^x - 1) otherwise)
   Self-normalizing activation function (Klambauer et al., 2017)
   λ ≈ 1.0507, α ≈ 1.67326"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float))
         (alpha 1.6732632423543772d0)
         (scale 1.0507009873554805d0))
    (dotimes (i (array-total-size data))
      (let ((x (row-major-aref data i)))
        (setf (row-major-aref result-data i)
              (* scale
                 (if (> x 0.0d0)
                     x
                     (* alpha (- (exp x) 1.0d0)))))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
        (maybe-attach-grad
         result tensor
         (lambda ()
           (when (requires-grad tensor)
             (let ((grad (tensor-grad tensor))
                   (grad-out (tensor-grad result)))
               (dotimes (i (array-total-size data))
                 (let ((x (row-major-aref data i)))
                   (incf (row-major-aref grad i)
                         (* (row-major-aref grad-out i)
                            scale
                            (if (> x 0.0d0)
                                1.0d0
                                (* alpha (exp x)))))))))))
      result)))

(defun gelu (tensor)
  "Gaussian Error Linear Unit (GELU) activation function.
   GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
   Used in BERT, GPT, and many modern transformers"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float))
         (sqrt-2/pi 0.7978845608028654d0)) ; sqrt(2/pi)
    (dotimes (i (array-total-size data))
      (let* ((x (row-major-aref data i))
             (x3 (* x x x))
             (inner (+ x (* 0.044715d0 x3)))
             (tanh-val (tanh (* sqrt-2/pi inner))))
        (setf (row-major-aref result-data i)
              (* 0.5d0 x (+ 1.0d0 tanh-val)))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (let* ((x (row-major-aref data i))
                      (x2 (* x x))
                      (x3 (* x2 x))
                      (inner (+ x (* 0.044715d0 x3)))
                      (tanh-val (tanh (* sqrt-2/pi inner)))
                      (sech2 (- 1.0d0 (* tanh-val tanh-val)))
                      (grad-inner (* sqrt-2/pi (+ 1.0d0 (* 3.0d0 0.044715d0 x2))))
                      (grad-gelu (+ (* 0.5d0 (+ 1.0d0 tanh-val))
                                   (* 0.5d0 x sech2 grad-inner))))
                 (incf (row-major-aref grad i)
                       (* (row-major-aref grad-out i) grad-gelu))))))))
      result)))

(defun swish (tensor &optional (beta 1.0d0))
  "Swish activation: Swish(x) = x * σ(βx)
   Also known as SiLU when β=1 (used in EfficientNet, YOLOv5)"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let* ((x (row-major-aref data i))
             (sigmoid-val (/ 1.0d0 (+ 1.0d0 (exp (- (* beta x)))))))
        (setf (row-major-aref result-data i)
              (* x sigmoid-val))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (let* ((x (row-major-aref data i))
                      (sigmoid-val (/ 1.0d0 (+ 1.0d0 (exp (- (* beta x))))))
                      (grad-swish (+ sigmoid-val
                                    (* x sigmoid-val (- 1.0d0 sigmoid-val) beta))))
                 (incf (row-major-aref grad i)
                       (* (row-major-aref grad-out i) grad-swish))))))))
      result)))

(defun silu (tensor)
  "SiLU (Sigmoid Linear Unit): SiLU(x) = x * σ(x)
   Equivalent to Swish with β=1"
  (swish tensor 1.0d0))

(defun mish (tensor)
  "Mish activation: Mish(x) = x * tanh(softplus(x))
   Where softplus(x) = ln(1 + e^x)
   Smooth, self-regularized activation (Misra, 2019)"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let* ((x (row-major-aref data i))
             (softplus (log (+ 1.0d0 (exp x))))
             (tanh-val (tanh softplus)))
        (setf (row-major-aref result-data i)
              (* x tanh-val))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (let* ((x (row-major-aref data i))
                      (exp-x (exp x))
                      (softplus (log (+ 1.0d0 exp-x)))
                      (tanh-val (tanh softplus))
                      (sigmoid-val (/ 1.0d0 (+ 1.0d0 (exp (- x)))))
                      (sech2 (- 1.0d0 (* tanh-val tanh-val)))
                      (grad-mish (+ tanh-val (* x sech2 sigmoid-val))))
                 (incf (row-major-aref grad i)
                       (* (row-major-aref grad-out i) grad-mish))))))))
      result)))

(defun hardswish (tensor)
  "Hard Swish: HardSwish(x) = x * ReLU6(x + 3) / 6
   Efficient approximation of Swish (used in MobileNetV3)"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let* ((x (row-major-aref data i))
             (relu6-val (max 0.0d0 (min 6.0d0 (+ x 3.0d0)))))
        (setf (row-major-aref result-data i)
              (* x (/ relu6-val 6.0d0)))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (let* ((x (row-major-aref data i))
                      (x-plus-3 (+ x 3.0d0))
                      (grad-hardswish (cond
                                        ((< x-plus-3 0.0d0) 0.0d0)
                                        ((> x-plus-3 6.0d0) 1.0d0)
                                        (t (+ (/ x-plus-3 6.0d0) (/ x 6.0d0))))))
                 (incf (row-major-aref grad i)
                       (* (row-major-aref grad-out i) grad-hardswish))))))))
      result)))

(defun hard-sigmoid (tensor)
  "Hard Sigmoid: HardSigmoid(x) = clip((x + 3) / 6, 0, 1)
   Efficient approximation of sigmoid"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let ((x (row-major-aref data i)))
        (setf (row-major-aref result-data i)
              (max 0.0d0 (min 1.0d0 (/ (+ x 3.0d0) 6.0d0))))))
    (make-tensor result-data
                 :shape (tensor-shape tensor)
                 :requires-grad (requires-grad tensor))))

(defun relu6 (tensor)
  "ReLU6: ReLU6(x) = min(max(0, x), 6)
   Bounded ReLU used in MobileNets"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (setf (row-major-aref result-data i)
            (min 6.0d0 (max 0.0d0 (row-major-aref data i)))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (let ((x (row-major-aref data i)))
                 (when (and (> x 0.0d0) (< x 6.0d0))
                   (incf (row-major-aref grad i)
                         (row-major-aref grad-out i)))))))))
      result)))

(defun celu (tensor &optional (alpha 1.0d0))
  "Continuously Differentiable ELU: smooth version of ELU with parameter α."
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let ((x (row-major-aref data i)))
        (setf (row-major-aref result-data i)
              (if (> x 0.0d0)
                  x
                  (* alpha (- (exp (/ x alpha)) 1.0d0))))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (let ((x (row-major-aref data i))
                     (grad-scale (if (> x 0.0d0)
                                     1.0d0
                                     (exp (/ x alpha)))))
                 (incf (row-major-aref grad i)
                       (* (row-major-aref grad-out i) grad-scale))))))))
      result)))

;;;; ============================================================================
;;;; Advanced Activation Functions
;;;; ============================================================================

(defun softsign (tensor)
  "Softsign: x / (1 + |x|), an alternative to tanh with polynomial tail."
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let ((x (row-major-aref data i)))
        (setf (row-major-aref result-data i)
              (/ x (+ 1.0d0 (abs x))))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (let* ((x (row-major-aref data i))
                      (denom (+ 1.0d0 (abs x)))
                      (grad-softsign (/ 1.0d0 (* denom denom))))
                 (incf (row-major-aref grad i)
                       (* (row-major-aref grad-out i) grad-softsign))))))))
      result)))

(defun softplus (tensor &optional (beta 1.0d0))
  "Softplus: Softplus(x) = (1/β) * ln(1 + e^(βx))
   Smooth approximation of ReLU"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let ((x (row-major-aref data i)))
        (setf (row-major-aref result-data i)
              (/ (log (+ 1.0d0 (exp (* beta x)))) beta))))
    (let ((result (make-tensor result-data
                               :shape (tensor-shape tensor)
                               :requires-grad (requires-grad tensor))))
      (maybe-attach-grad
       result tensor
       (lambda ()
         (when (requires-grad tensor)
           (let ((grad (tensor-grad tensor))
                 (grad-out (tensor-grad result)))
             (dotimes (i (array-total-size data))
               (let* ((x (row-major-aref data i))
                      (grad-softplus (/ 1.0d0 (+ 1.0d0 (exp (- (* beta x)))))))
                 (incf (row-major-aref grad i)
                       (* (row-major-aref grad-out i) grad-softplus))))))))
      result)))

(defun hard-tanh (tensor &optional (min-val -1.0d0) (max-val 1.0d0))
  "Hard Tanh: HardTanh(x) = clip(x, min_val, max_val)
   Linear approximation of tanh"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let ((x (row-major-aref data i)))
        (setf (row-major-aref result-data i)
              (max min-val (min max-val x)))))
    (make-tensor result-data
                 :shape (tensor-shape tensor)
                 :requires-grad (requires-grad tensor))))

(defun tanh-shrink (tensor)
  "Tanh Shrink: TanhShrink(x) = x - tanh(x)"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let ((x (row-major-aref data i)))
        (setf (row-major-aref result-data i)
              (- x (tanh x)))))
    (make-tensor result-data
                 :shape (tensor-shape tensor)
                 :requires-grad (requires-grad tensor))))

(defun soft-shrink (tensor &optional (lambda-val 0.5d0))
  "Soft Shrink: SoftShrink(x) = x - λ if x > λ, x + λ if x < -λ, 0 otherwise"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let ((x (row-major-aref data i)))
        (setf (row-major-aref result-data i)
              (cond
                ((> x lambda-val) (- x lambda-val))
                ((< x (- lambda-val)) (+ x lambda-val))
                (t 0.0d0)))))
    (make-tensor result-data
                 :shape (tensor-shape tensor)
                 :requires-grad (requires-grad tensor))))

(defun hard-shrink (tensor &optional (lambda-val 0.5d0))
  "Hard Shrink: HardShrink(x) = x if |x| > λ, 0 otherwise"
  (let* ((data (tensor-data tensor))
         (result-data (make-array (tensor-shape tensor) :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (let ((x (row-major-aref data i)))
        (setf (row-major-aref result-data i)
              (if (> (abs x) lambda-val) x 0.0d0))))
    (make-tensor result-data
                 :shape (tensor-shape tensor)
                 :requires-grad (requires-grad tensor))))

;;;; ============================================================================
;;;; Gated Linear Units (GLU variants)
;;;; ============================================================================

(defun glu (tensor &key (dim -1))
  "Gated Linear Unit: GLU(x) = x₁ ⊗ σ(x₂)
   Splits input in half along dimension and applies gating"
  (declare (ignore dim))
  (let* ((data (tensor-data tensor))
         (shape (tensor-shape tensor))
         (last-dim (car (last shape)))
         (half-dim (floor last-dim 2)))
    (when (/= (* half-dim 2) last-dim)
      (error "GLU requires even number of features in last dimension"))
    (let ((result-data (make-array (append (butlast shape) (list half-dim))
                                   :element-type 'double-float)))
      ;; Simplified: assumes 2D tensor (batch, features)
      (dotimes (i (first shape))
        (dotimes (j half-dim)
          (let* ((idx1 (+ (* i last-dim) j))
                 (idx2 (+ (* i last-dim) j half-dim))
                 (x1 (row-major-aref data idx1))
                 (x2 (row-major-aref data idx2))
                 (gate (/ 1.0d0 (+ 1.0d0 (exp (- x2))))))
            (setf (aref result-data i j) (* x1 gate)))))
      (make-tensor result-data
                   :requires-grad (requires-grad tensor)))))

(defun geglu (tensor)
  "GELU Gated Linear Unit: GeGLU(x) = x₁ ⊗ GELU(x₂)
   Used in transformers (Shazeer, 2020)"
  (let* ((data (tensor-data tensor))
         (shape (tensor-shape tensor))
         (last-dim (car (last shape)))
         (half-dim (floor last-dim 2)))
    (when (/= (* half-dim 2) last-dim)
      (error "GeGLU requires even number of features in last dimension"))
    (let ((result-data (make-array (append (butlast shape) (list half-dim))
                                   :element-type 'double-float))
          (sqrt-2/pi 0.7978845608028654d0))
      ;; Simplified: assumes 2D tensor
      (dotimes (i (first shape))
        (dotimes (j half-dim)
          (let* ((idx1 (+ (* i last-dim) j))
                 (idx2 (+ (* i last-dim) j half-dim))
                 (x1 (row-major-aref data idx1))
                 (x2 (row-major-aref data idx2))
                 (x2-3 (* x2 x2 x2))
                 (inner (+ x2 (* 0.044715d0 x2-3)))
                 (gelu-val (* 0.5d0 x2 (+ 1.0d0 (tanh (* sqrt-2/pi inner))))))
            (setf (aref result-data i j) (* x1 gelu-val)))))
      (make-tensor result-data
                   :requires-grad (requires-grad tensor)))))

(defun swiglu (tensor)
  "Swish/SiLU Gated Linear Unit: SwiGLU(x) = x₁ ⊗ Swish(x₂)
   Used in modern transformers (PaLM, LLaMA)"
  (let* ((data (tensor-data tensor))
         (shape (tensor-shape tensor))
         (last-dim (car (last shape)))
         (half-dim (floor last-dim 2)))
    (when (/= (* half-dim 2) last-dim)
      (error "SwiGLU requires even number of features in last dimension"))
    (let ((result-data (make-array (append (butlast shape) (list half-dim))
                                   :element-type 'double-float)))
      ;; Simplified: assumes 2D tensor
      (dotimes (i (first shape))
        (dotimes (j half-dim)
          (let* ((idx1 (+ (* i last-dim) j))
                 (idx2 (+ (* i last-dim) j half-dim))
                 (x1 (row-major-aref data idx1))
                 (x2 (row-major-aref data idx2))
                 (sigmoid-val (/ 1.0d0 (+ 1.0d0 (exp (- x2)))))
                 (swish-val (* x2 sigmoid-val)))
            (setf (aref result-data i j) (* x1 swish-val)))))
      (make-tensor result-data
                   :requires-grad (requires-grad tensor)))))

(defun maxout (tensor num-pieces)
  "Maxout activation: takes max over groups of features
   Used for learning piecewise linear activations"
  (let* ((data (tensor-data tensor))
         (shape (tensor-shape tensor))
         (last-dim (car (last shape)))
         (out-dim (floor last-dim num-pieces)))
    (when (/= (* out-dim num-pieces) last-dim)
      (error "Maxout requires features divisible by num-pieces"))
    (let ((result-data (make-array (append (butlast shape) (list out-dim))
                                   :element-type 'double-float)))
      ;; Simplified: assumes 2D tensor
      (dotimes (i (first shape))
        (dotimes (j out-dim)
          (let ((max-val (loop for k below num-pieces
                              maximize (row-major-aref data (+ (* i last-dim)
                                                               (* j num-pieces)
                                                               k)))))
            (setf (aref result-data i j) max-val))))
      (make-tensor result-data
                   :requires-grad (requires-grad tensor)))))

;;;; ============================================================================
;;;; Activation Layer Classes
;;;; ============================================================================

(defclass relu-layer (layer) ())
(defmethod forward ((layer relu-layer) input)
  (relu input))

(defclass leaky-relu-layer (layer)
  ((negative-slope :initarg :negative-slope
                   :initform 0.01d0
                   :reader negative-slope)))
(defmethod forward ((layer leaky-relu-layer) input)
  (leaky-relu input (negative-slope layer)))

(defclass elu-layer (layer)
  ((alpha :initarg :alpha
          :initform 1.0d0
          :reader elu-alpha)))
(defmethod forward ((layer elu-layer) input)
  (elu input (elu-alpha layer)))

(defclass selu-layer (layer) ())
(defmethod forward ((layer selu-layer) input)
  (selu input))

(defclass gelu-layer (layer) ())
(defmethod forward ((layer gelu-layer) input)
  (gelu input))

(defclass swish-layer (layer)
  ((beta :initarg :beta
         :initform 1.0d0
         :reader swish-beta)))
(defmethod forward ((layer swish-layer) input)
  (swish input (swish-beta layer)))

(defclass mish-layer (layer) ())
(defmethod forward ((layer mish-layer) input)
  (mish input))

(defclass sigmoid-layer (layer) ())
(defmethod forward ((layer sigmoid-layer) input)
  (sigmoid input))

(defclass tanh-layer (layer) ())
(defmethod forward ((layer tanh-layer) input)
  (tanh-activation input))

(defclass softmax-layer (layer)
  ((dim :initarg :dim
        :initform -1
        :reader softmax-dim)))
(defmethod forward ((layer softmax-layer) input)
  (softmax input :dim (softmax-dim layer)))

(defclass log-softmax-layer (layer)
  ((dim :initarg :dim
        :initform -1
        :reader log-softmax-dim)))
(defmethod forward ((layer log-softmax-layer) input)
  (log-softmax input :dim (log-softmax-dim layer)))

(defclass hardswish-layer (layer) ())
(defmethod forward ((layer hardswish-layer) input)
  (hardswish input))

(defclass hard-sigmoid-layer (layer) ())
(defmethod forward ((layer hard-sigmoid-layer) input)
  (hard-sigmoid input))

(defclass softsign-layer (layer) ())
(defmethod forward ((layer softsign-layer) input)
  (softsign input))

(defclass softplus-layer (layer)
  ((beta :initarg :beta
         :initform 1.0d0
         :reader softplus-beta)))
(defmethod forward ((layer softplus-layer) input)
  (softplus input (softplus-beta layer)))

(defclass relu6-layer (layer) ())
(defmethod forward ((layer relu6-layer) input)
  (relu6 input))

(defclass glu-layer (layer)
  ((dim :initarg :dim
        :initform -1
        :reader glu-dim)))
(defmethod forward ((layer glu-layer) input)
  (glu input :dim (glu-dim layer)))

(defclass geglu-layer (layer) ())
(defmethod forward ((layer geglu-layer) input)
  (geglu input))

(defclass swiglu-layer (layer) ())
(defmethod forward ((layer swiglu-layer) input)
  (swiglu input))

;;;; ============================================================================
;;;; Constructor Functions
;;;; ============================================================================

(defun relu-layer () (make-instance 'relu-layer))
(defun sigmoid-layer () (make-instance 'sigmoid-layer))
(defun tanh-layer () (make-instance 'tanh-layer))
(defun leaky-relu-layer (&optional (negative-slope 0.01d0))
  (make-instance 'leaky-relu-layer :negative-slope negative-slope))
(defun elu-layer (&optional (alpha 1.0d0))
  (make-instance 'elu-layer :alpha alpha))
(defun selu-layer () (make-instance 'selu-layer))
(defun gelu-layer () (make-instance 'gelu-layer))
(defun swish-layer (&optional (beta 1.0d0))
  (make-instance 'swish-layer :beta beta))
(defun mish-layer () (make-instance 'mish-layer))
(defun softmax-layer (&key (dim -1))
  (make-instance 'softmax-layer :dim dim))
(defun log-softmax-layer (&key (dim -1))
  (make-instance 'log-softmax-layer :dim dim))
(defun hardswish-layer () (make-instance 'hardswish-layer))
(defun hard-sigmoid-layer () (make-instance 'hard-sigmoid-layer))
(defun softsign-layer () (make-instance 'softsign-layer))
(defun softplus-layer (&optional (beta 1.0d0))
  (make-instance 'softplus-layer :beta beta))
(defun relu6-layer () (make-instance 'relu6-layer))
(defun glu-layer (&key (dim -1))
  (make-instance 'glu-layer :dim dim))
(defun geglu-layer () (make-instance 'geglu-layer))
(defun swiglu-layer () (make-instance 'swiglu-layer))
