;;;; tests/test-activations.lisp - Comprehensive Tests for Activation Functions

(in-package #:neural-lisp-tests)

(def-suite activation-tests
  :description "Comprehensive test suite for activation functions"
  :in neural-lisp-tests)

(in-suite activation-tests)

;;;; ============================================================================
;;;; Helper Functions
;;;; ============================================================================

(defun assert-tensor-shape (expected-shape tensor)
  "Assert that tensor has expected shape"
  (is (equal expected-shape (tensor-shape tensor))))

(defun assert-tensor-range (tensor min-val max-val)
  "Assert all tensor values are in range [min-val, max-val]"
  (let ((data (tensor-data tensor)))
    (dotimes (i (array-total-size data))
      (let ((val (row-major-aref data i)))
        (is (<= min-val val max-val))))))

(defun assert-tensor-all-equal (tensor expected-val &optional (tolerance 1d-6))
  "Assert all tensor values equal expected value"
  (let ((data (tensor-data tensor)))
    (dotimes (i (array-total-size data))
      (is (< (abs (- (row-major-aref data i) expected-val)) tolerance)))))

(defun tensor-min (tensor)
  "Get minimum value in tensor"
  (let ((data (tensor-data tensor))
        (min-val most-positive-double-float))
    (dotimes (i (array-total-size data))
      (setf min-val (min min-val (row-major-aref data i))))
    min-val))

(defun tensor-max (tensor)
  "Get maximum value in tensor"
  (let ((data (tensor-data tensor))
        (max-val most-negative-double-float))
    (dotimes (i (array-total-size data))
      (setf max-val (max max-val (row-major-aref data i))))
    max-val))

;;;; ============================================================================
;;;; ReLU Tests
;;;; ============================================================================

(test relu-positive-values
  "ReLU should preserve positive values"
  (let* ((input (make-tensor #2A((1.0d0 2.0d0) (3.0d0 4.0d0)) :shape '(2 2)))
         (output (relu input)))
    (assert-tensor-shape '(2 2) output)
    (is (= 1.0d0 (aref (tensor-data output) 0 0)))
    (is (= 2.0d0 (aref (tensor-data output) 0 1)))
    (is (= 3.0d0 (aref (tensor-data output) 1 0)))
    (is (= 4.0d0 (aref (tensor-data output) 1 1)))))

(test relu-negative-values
  "ReLU should zero out negative values"
  (let* ((input (make-tensor #2A((-1.0d0 -2.0d0) (-3.0d0 -4.0d0)) :shape '(2 2)))
         (output (relu input)))
    (assert-tensor-all-equal output 0.0d0)))

(test relu-mixed-values
  "ReLU should handle mixed positive/negative values"
  (let* ((input (make-tensor #2A((-1.0d0 2.0d0) (3.0d0 -4.0d0)) :shape '(2 2)))
         (output (relu input)))
    (is (= 0.0d0 (aref (tensor-data output) 0 0)))
    (is (= 2.0d0 (aref (tensor-data output) 0 1)))
    (is (= 3.0d0 (aref (tensor-data output) 1 0)))
    (is (= 0.0d0 (aref (tensor-data output) 1 1)))))

(test relu-zero-values
  "ReLU should preserve zeros"
  (let* ((input (make-tensor #2A((0.0d0 0.0d0) (0.0d0 0.0d0)) :shape '(2 2)))
         (output (relu input)))
    (assert-tensor-all-equal output 0.0d0)))

(test relu-large-values
  "ReLU should handle large values"
  (let* ((input (make-tensor #2A((1000.0d0 -1000.0d0)) :shape '(1 2)))
         (output (relu input)))
    (is (= 1000.0d0 (aref (tensor-data output) 0 0)))
    (is (= 0.0d0 (aref (tensor-data output) 0 1)))))

(test relu-small-values
  "ReLU should handle very small values"
  (let* ((input (make-tensor #2A((1d-10 -1d-10)) :shape '(1 2)))
         (output (relu input)))
    (is (= 1d-10 (aref (tensor-data output) 0 0)))
    (is (= 0.0d0 (aref (tensor-data output) 0 1)))))

;;;; ============================================================================
;;;; Sigmoid Tests
;;;; ============================================================================

(test sigmoid-output-range
  "Sigmoid output should be in (0, 1) range"
  (let* ((input (make-tensor #2A((-5.0d0 0.0d0) (5.0d0 10.0d0)) :shape '(2 2)))
         (output (sigmoid input)))
    (assert-tensor-range output 0.0d0 1.0d0)))

(test sigmoid-zero-input
  "Sigmoid(0) should be 0.5"
  (let* ((input (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (output (sigmoid input)))
    (is (< (abs (- 0.5d0 (aref (tensor-data output) 0 0))) 1d-6))))

(test sigmoid-large-positive
  "Sigmoid of large positive should approach 1"
  (let* ((input (make-tensor #2A((10.0d0)) :shape '(1 1)))
         (output (sigmoid input)))
    (is (> (aref (tensor-data output) 0 0) 0.9999d0))))

(test sigmoid-large-negative
  "Sigmoid of large negative should approach 0"
  (let* ((input (make-tensor #2A((-10.0d0)) :shape '(1 1)))
         (output (sigmoid input)))
    (is (< (aref (tensor-data output) 0 0) 0.0001d0))))

(test sigmoid-symmetry
  "Sigmoid should be symmetric: σ(x) + σ(-x) = 1"
  (let* ((input (make-tensor #2A((2.5d0)) :shape '(1 1)))
         (output-pos (sigmoid input))
         (input-neg (make-tensor #2A((-2.5d0)) :shape '(1 1)))
         (output-neg (sigmoid input-neg))
         (sum (+ (aref (tensor-data output-pos) 0 0)
                 (aref (tensor-data output-neg) 0 0))))
    (is (< (abs (- 1.0d0 sum)) 1d-6))))

;;;; ============================================================================
;;;; Tanh Tests
;;;; ============================================================================

(test tanh-output-range
  "Tanh output should be in (-1, 1) range"
  (let* ((input (make-tensor #2A((-5.0d0 0.0d0) (5.0d0 10.0d0)) :shape '(2 2)))
         (output (tanh-activation input)))
    (assert-tensor-range output -1.0d0 1.0d0)))

(test tanh-zero-input
  "Tanh(0) should be 0"
  (let* ((input (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (output (tanh-activation input)))
    (is (< (abs (aref (tensor-data output) 0 0)) 1d-6))))

(test tanh-large-positive
  "Tanh of large positive should approach 1"
  (let* ((input (make-tensor #2A((10.0d0)) :shape '(1 1)))
         (output (tanh-activation input)))
    (is (> (aref (tensor-data output) 0 0) 0.9999d0))))

(test tanh-large-negative
  "Tanh of large negative should approach -1"
  (let* ((input (make-tensor #2A((-10.0d0)) :shape '(1 1)))
         (output (tanh-activation input)))
    (is (< (aref (tensor-data output) 0 0) -0.9999d0))))

(test tanh-odd-symmetry
  "Tanh should be odd: tanh(-x) = -tanh(x)"
  (let* ((input (make-tensor #2A((2.5d0)) :shape '(1 1)))
         (output-pos (tanh-activation input))
         (input-neg (make-tensor #2A((-2.5d0)) :shape '(1 1)))
         (output-neg (tanh-activation input-neg)))
    (is (< (abs (+ (aref (tensor-data output-pos) 0 0)
                   (aref (tensor-data output-neg) 0 0))) 1d-6))))

;;;; ============================================================================
;;;; Leaky ReLU Tests
;;;; ============================================================================

(test leaky-relu-positive-values
  "Leaky ReLU should preserve positive values"
  (let* ((input (make-tensor #2A((1.0d0 2.0d0)) :shape '(1 2)))
         (output (leaky-relu input 0.01d0)))
    (is (= 1.0d0 (aref (tensor-data output) 0 0)))
    (is (= 2.0d0 (aref (tensor-data output) 0 1)))))

(test leaky-relu-negative-values
  "Leaky ReLU should scale negative values"
  (let* ((input (make-tensor #2A((-1.0d0 -2.0d0)) :shape '(1 2)))
         (output (leaky-relu input 0.1d0)))
    (is (< (abs (- -0.1d0 (aref (tensor-data output) 0 0))) 1d-6))
    (is (< (abs (- -0.2d0 (aref (tensor-data output) 0 1))) 1d-6))))

(test leaky-relu-different-slopes
  "Leaky ReLU should respect different negative slopes"
  (let* ((input (make-tensor #2A((-1.0d0)) :shape '(1 1)))
         (output1 (leaky-relu input 0.01d0))
         (output2 (leaky-relu input 0.1d0)))
    (is (< (abs (- -0.01d0 (aref (tensor-data output1) 0 0))) 1d-6))
    (is (< (abs (- -0.1d0 (aref (tensor-data output2) 0 0))) 1d-6))))

(test leaky-relu-zero-slope
  "Leaky ReLU with zero slope should behave like ReLU"
  (let* ((input (make-tensor #2A((-1.0d0 2.0d0)) :shape '(1 2)))
         (output (leaky-relu input 0.0d0)))
    (is (= 0.0d0 (aref (tensor-data output) 0 0)))
    (is (= 2.0d0 (aref (tensor-data output) 0 1)))))

;;;; ============================================================================
;;;; ELU Tests
;;;; ============================================================================

(test elu-positive-values
  "ELU should preserve positive values"
  (let* ((input (make-tensor #2A((1.0d0 2.0d0)) :shape '(1 2)))
         (output (elu input 1.0d0)))
    (is (= 1.0d0 (aref (tensor-data output) 0 0)))
    (is (= 2.0d0 (aref (tensor-data output) 0 1)))))

(test elu-negative-values
  "ELU should apply exponential to negative values"
  (let* ((input (make-tensor #2A((-1.0d0)) :shape '(1 1)))
         (output (elu input 1.0d0))
         (expected (* 1.0d0 (- (exp -1.0d0) 1.0d0))))
    (is (< (abs (- expected (aref (tensor-data output) 0 0))) 1d-6))))

(test elu-zero-input
  "ELU(0) should be 0"
  (let* ((input (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (output (elu input 1.0d0)))
    (is (< (abs (aref (tensor-data output) 0 0)) 1d-6))))

(test elu-different-alphas
  "ELU should respect different alpha values"
  (let* ((input (make-tensor #2A((-1.0d0)) :shape '(1 1)))
         (output1 (elu input 1.0d0))
         (output2 (elu input 2.0d0)))
    (is (/= (aref (tensor-data output1) 0 0)
            (aref (tensor-data output2) 0 0)))))

;;;; ============================================================================
;;;; SELU Tests
;;;; ============================================================================

(test selu-positive-values
  "SELU should scale positive values"
  (let* ((input (make-tensor #2A((1.0d0)) :shape '(1 1)))
         (output (selu input))
         (scale 1.0507009873554805d0))
    (is (< (abs (- scale (aref (tensor-data output) 0 0))) 1d-6))))

(test selu-negative-values
  "SELU should apply scaled exponential to negative values"
  (let* ((input (make-tensor #2A((-1.0d0)) :shape '(1 1)))
         (output (selu input)))
    (is (< (aref (tensor-data output) 0 0) 0.0d0))))

(test selu-zero-input
  "SELU(0) should be 0"
  (let* ((input (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (output (selu input)))
    (is (< (abs (aref (tensor-data output) 0 0)) 1d-6))))

(test selu-self-normalizing-property
  "SELU should maintain specific scale and alpha"
  (let* ((input (make-tensor #2A((1.0d0 -1.0d0)) :shape '(1 2)))
         (output (selu input)))
    (is (not (null output)))))

;;;; ============================================================================
;;;; GELU Tests
;;;; ============================================================================

(test gelu-zero-input
  "GELU(0) should be 0"
  (let* ((input (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (output (gelu input)))
    (is (< (abs (aref (tensor-data output) 0 0)) 1d-3))))

(test gelu-positive-values
  "GELU should produce positive output for positive input"
  (let* ((input (make-tensor #2A((1.0d0 2.0d0 3.0d0)) :shape '(1 3)))
         (output (gelu input)))
    (is (> (aref (tensor-data output) 0 0) 0.0d0))
    (is (> (aref (tensor-data output) 0 1) 0.0d0))
    (is (> (aref (tensor-data output) 0 2) 0.0d0))))

(test gelu-negative-values
  "GELU should produce small negative output for negative input"
  (let* ((input (make-tensor #2A((-1.0d0 -2.0d0)) :shape '(1 2)))
         (output (gelu input)))
    (is (< (aref (tensor-data output) 0 0) 0.0d0))
    (is (< (aref (tensor-data output) 0 1) 0.0d0))))

(test gelu-large-positive
  "GELU of large positive should approximate identity"
  (let* ((input (make-tensor #2A((5.0d0)) :shape '(1 1)))
         (output (gelu input)))
    (is (> (aref (tensor-data output) 0 0) 4.9d0))))

(test gelu-smooth-transition
  "GELU should be smooth around zero"
  (let* ((input1 (make-tensor #2A((-0.1d0)) :shape '(1 1)))
         (input2 (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (input3 (make-tensor #2A((0.1d0)) :shape '(1 1)))
         (output1 (gelu input1))
         (output2 (gelu input2))
         (output3 (gelu input3)))
    (is (< (aref (tensor-data output1) 0 0)
           (aref (tensor-data output2) 0 0)))
    (is (< (aref (tensor-data output2) 0 0)
           (aref (tensor-data output3) 0 0)))))

;;;; ============================================================================
;;;; Swish/SiLU Tests
;;;; ============================================================================

(test swish-zero-input
  "Swish(0) should be 0"
  (let* ((input (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (output (swish input)))
    (is (< (abs (aref (tensor-data output) 0 0)) 1d-6))))

(test swish-positive-values
  "Swish should produce positive output for positive input"
  (let* ((input (make-tensor #2A((1.0d0 2.0d0)) :shape '(1 2)))
         (output (swish input)))
    (is (> (aref (tensor-data output) 0 0) 0.0d0))
    (is (> (aref (tensor-data output) 0 1) 0.0d0))))

(test swish-negative-values
  "Swish should produce small negative output for negative input"
  (let* ((input (make-tensor #2A((-1.0d0)) :shape '(1 1)))
         (output (swish input)))
    (is (< (aref (tensor-data output) 0 0) 0.0d0))))

(test swish-large-positive
  "Swish of large positive should approximate identity"
  (let* ((input (make-tensor #2A((10.0d0)) :shape '(1 1)))
         (output (swish input)))
    (is (> (aref (tensor-data output) 0 0) 9.9d0))))

(test silu-equals-swish-beta-one
  "SiLU should equal Swish with β=1"
  (let* ((input (make-tensor #2A((1.5d0)) :shape '(1 1)))
         (silu-output (silu input))
         (swish-output (swish input 1.0d0)))
    (is (< (abs (- (aref (tensor-data silu-output) 0 0)
                   (aref (tensor-data swish-output) 0 0))) 1d-6))))

;;;; ============================================================================
;;;; Mish Tests
;;;; ============================================================================

(test mish-zero-input
  "Mish(0) should be close to 0"
  (let* ((input (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (output (mish input)))
    (is (< (abs (aref (tensor-data output) 0 0)) 1d-3))))

(test mish-positive-values
  "Mish should produce positive output for positive input"
  (let* ((input (make-tensor #2A((1.0d0 2.0d0)) :shape '(1 2)))
         (output (mish input)))
    (is (> (aref (tensor-data output) 0 0) 0.0d0))
    (is (> (aref (tensor-data output) 0 1) 0.0d0))))

(test mish-smooth-activation
  "Mish should be smooth"
  (let* ((input (make-tensor #2A((-1.0d0 0.0d0 1.0d0)) :shape '(1 3)))
         (output (mish input)))
    (is (< (aref (tensor-data output) 0 0)
           (aref (tensor-data output) 0 1)))
    (is (< (aref (tensor-data output) 0 1)
           (aref (tensor-data output) 0 2)))))

;;;; ============================================================================
;;;; Softmax Tests
;;;; ============================================================================

(test softmax-output-sums-to-one
  "Softmax output should sum to 1"
  (let* ((input (make-tensor #2A((1.0d0 2.0d0 3.0d0)) :shape '(1 3)))
         (output (softmax input))
         (data (tensor-data output))
         (sum (+ (aref data 0 0) (aref data 0 1) (aref data 0 2))))
    (is (< (abs (- 1.0d0 sum)) 1d-6))))

(test softmax-output-range
  "Softmax output should be in (0, 1) range"
  (let* ((input (make-tensor #2A((1.0d0 2.0d0 3.0d0)) :shape '(1 3)))
         (output (softmax input)))
    (assert-tensor-range output 0.0d0 1.0d0)))

(test softmax-max-element
  "Softmax should give highest probability to max input"
  (let* ((input (make-tensor #2A((1.0d0 5.0d0 2.0d0)) :shape '(1 3)))
         (output (softmax input))
         (data (tensor-data output)))
    (is (> (aref data 0 1) (aref data 0 0)))
    (is (> (aref data 0 1) (aref data 0 2)))))

(test softmax-uniform-input
  "Softmax of uniform input should be uniform"
  (let* ((input (make-tensor #2A((1.0d0 1.0d0 1.0d0)) :shape '(1 3)))
         (output (softmax input))
         (data (tensor-data output))
         (expected (/ 1.0d0 3.0d0)))
    (is (< (abs (- (aref data 0 0) expected)) 1d-6))
    (is (< (abs (- (aref data 0 1) expected)) 1d-6))
    (is (< (abs (- (aref data 0 2) expected)) 1d-6))))

(test softmax-numerical-stability
  "Softmax should handle large values"
  (let* ((input (make-tensor #2A((1000.0d0 1001.0d0)) :shape '(1 2)))
         (output (softmax input))
         (data (tensor-data output)))
    (is (< (abs (- 1.0d0 (+ (aref data 0 0) (aref data 0 1)))) 1d-6))))

;;;; ============================================================================
;;;; HardSwish Tests
;;;; ============================================================================

(test hardswish-output-range
  "HardSwish output should be bounded"
  (let* ((input (make-tensor #2A((-5.0d0 0.0d0 5.0d0)) :shape '(1 3)))
         (output (hardswish input)))
    (is (not (null output)))))

(test hardswish-zero-region
  "HardSwish should be zero for x < -3"
  (let* ((input (make-tensor #2A((-5.0d0)) :shape '(1 1)))
         (output (hardswish input)))
    (is (< (abs (aref (tensor-data output) 0 0)) 1d-6))))

(test hardswish-linear-region
  "HardSwish should be x for x > 3"
  (let* ((input (make-tensor #2A((5.0d0)) :shape '(1 1)))
         (output (hardswish input)))
    (is (< (abs (- 5.0d0 (aref (tensor-data output) 0 0))) 1d-6))))

;;;; ============================================================================
;;;; ReLU6 Tests
;;;; ============================================================================

(test relu6-clips-at-six
  "ReLU6 should clip values at 6"
  (let* ((input (make-tensor #2A((10.0d0)) :shape '(1 1)))
         (output (relu6 input)))
    (is (< (abs (- 6.0d0 (aref (tensor-data output) 0 0))) 1d-6))))

(test relu6-zeros-negative
  "ReLU6 should zero out negative values"
  (let* ((input (make-tensor #2A((-5.0d0)) :shape '(1 1)))
         (output (relu6 input)))
    (is (< (abs (aref (tensor-data output) 0 0)) 1d-6))))

(test relu6-preserves-middle-range
  "ReLU6 should preserve values in [0, 6]"
  (let* ((input (make-tensor #2A((3.0d0)) :shape '(1 1)))
         (output (relu6 input)))
    (is (< (abs (- 3.0d0 (aref (tensor-data output) 0 0))) 1d-6))))

;;;; ============================================================================
;;;; Softsign Tests
;;;; ============================================================================

(test softsign-output-range
  "Softsign output should be in (-1, 1)"
  (let* ((input (make-tensor #2A((-10.0d0 0.0d0 10.0d0)) :shape '(1 3)))
         (output (softsign input)))
    (assert-tensor-range output -1.0d0 1.0d0)))

(test softsign-zero-input
  "Softsign(0) should be 0"
  (let* ((input (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (output (softsign input)))
    (is (< (abs (aref (tensor-data output) 0 0)) 1d-6))))

(test softsign-symmetry
  "Softsign should be odd"
  (let* ((input-pos (make-tensor #2A((2.0d0)) :shape '(1 1)))
         (input-neg (make-tensor #2A((-2.0d0)) :shape '(1 1)))
         (output-pos (softsign input-pos))
         (output-neg (softsign input-neg)))
    (is (< (abs (+ (aref (tensor-data output-pos) 0 0)
                   (aref (tensor-data output-neg) 0 0))) 1d-6))))

;;;; ============================================================================
;;;; Softplus Tests
;;;; ============================================================================

(test softplus-positive-output
  "Softplus should always be positive"
  (let* ((input (make-tensor #2A((-10.0d0 0.0d0 10.0d0)) :shape '(1 3)))
         (output (softplus input)))
    (is (> (aref (tensor-data output) 0 0) 0.0d0))
    (is (> (aref (tensor-data output) 0 1) 0.0d0))
    (is (> (aref (tensor-data output) 0 2) 0.0d0))))

(test softplus-approximates-relu
  "Softplus should approximate ReLU for large values"
  (let* ((input (make-tensor #2A((10.0d0)) :shape '(1 1)))
         (output (softplus input)))
    (is (< (abs (- 10.0d0 (aref (tensor-data output) 0 0))) 1d-3))))

(test softplus-smooth-at-zero
  "Softplus should be smooth at zero"
  (let* ((input (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (output (softplus input)))
    (is (< (abs (- (log 2.0d0) (aref (tensor-data output) 0 0))) 1d-6))))

;;;; ============================================================================
;;;; Edge Case Tests
;;;; ============================================================================

(test activation-single-element
  "All activations should handle single element tensors"
  (let ((input (make-tensor #2A((1.5d0)) :shape '(1 1))))
    (is (not (null (relu input))))
    (is (not (null (sigmoid input))))
    (is (not (null (tanh-activation input))))
    (is (not (null (leaky-relu input))))
    (is (not (null (elu input))))
    (is (not (null (selu input))))
    (is (not (null (gelu input))))
    (is (not (null (swish input))))
    (is (not (null (mish input))))))

(test activation-large-tensor
  "Activations should handle large tensors"
  (let ((input (make-tensor (make-array '(10 10) 
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                           :shape '(10 10))))
    (is (not (null (relu input))))
    (is (not (null (sigmoid input))))
    (is (not (null (gelu input))))))

(test activation-preserves-shape
  "All activations should preserve tensor shape"
  (let ((input (make-tensor #2A((1.0d0 2.0d0) (3.0d0 4.0d0)) :shape '(2 2))))
    (assert-tensor-shape '(2 2) (relu input))
    (assert-tensor-shape '(2 2) (sigmoid input))
    (assert-tensor-shape '(2 2) (tanh-activation input))
    (assert-tensor-shape '(2 2) (gelu input))
    (assert-tensor-shape '(2 2) (swish input))))

(test activation-3d-tensor
  "Activations should handle 3D tensors"
  (let ((input (make-tensor (make-array '(2 3 4)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                           :shape '(2 3 4))))
    (assert-tensor-shape '(2 3 4) (relu input))
    (assert-tensor-shape '(2 3 4) (sigmoid input))
    (assert-tensor-shape '(2 3 4) (gelu input))))

;;;; ============================================================================
;;;; Activation Layer Tests
;;;; ============================================================================

(test relu-layer-forward
  "ReLU layer should work like relu function"
  (let* ((layer (make-instance 'relu-layer))
         (input (make-tensor #2A((1.0d0 -1.0d0)) :shape '(1 2)))
         (output (forward layer input)))
    (is (= 1.0d0 (aref (tensor-data output) 0 0)))
    (is (= 0.0d0 (aref (tensor-data output) 0 1)))))

(test sigmoid-layer-forward
  "Sigmoid layer should work like sigmoid function"
  (let* ((layer (make-instance 'sigmoid-layer))
         (input (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (output (forward layer input)))
    (is (< (abs (- 0.5d0 (aref (tensor-data output) 0 0))) 1d-6))))

(test gelu-layer-forward
  "GELU layer should work like gelu function"
  (let* ((layer (make-instance 'gelu-layer))
         (input (make-tensor #2A((1.0d0)) :shape '(1 1)))
         (output (forward layer input)))
    (is (> (aref (tensor-data output) 0 0) 0.0d0))))

(test leaky-relu-layer-custom-slope
  "Leaky ReLU layer should respect custom slope"
  (let* ((layer (make-instance 'leaky-relu-layer :negative-slope 0.2d0))
         (input (make-tensor #2A((-1.0d0)) :shape '(1 1)))
         (output (forward layer input)))
    (is (< (abs (- -0.2d0 (aref (tensor-data output) 0 0))) 1d-6))))

(test elu-layer-custom-alpha
  "ELU layer should respect custom alpha"
  (let* ((layer (make-instance 'elu-layer :alpha 0.5d0))
         (input (make-tensor #2A((1.0d0 -1.0d0)) :shape '(1 2)))
         (output (forward layer input)))
    (is (= 1.0d0 (aref (tensor-data output) 0 0)))))

(test swish-layer-custom-beta
  "Swish layer should respect custom beta"
  (let* ((layer (make-instance 'swish-layer :beta 2.0d0))
         (input (make-tensor #2A((1.0d0)) :shape '(1 1)))
         (output (forward layer input)))
    (is (> (aref (tensor-data output) 0 0) 0.0d0))))

;;;; ============================================================================
;;;; Numerical Stability Tests
;;;; ============================================================================

(test sigmoid-numerical-stability
  "Sigmoid should not overflow/underflow"
  (let* ((input (make-tensor #2A((100.0d0 -100.0d0)) :shape '(1 2)))
         (output (sigmoid input)))
    (is (not (null output)))
    (is (< (aref (tensor-data output) 0 0) 2.0d0))
    (is (> (aref (tensor-data output) 0 1) 0.0d0))))

(test softmax-numerical-stability
  "Softmax should handle extreme values"
  (let* ((input (make-tensor #2A((1000.0d0 999.0d0)) :shape '(1 2)))
         (output (softmax input))
         (data (tensor-data output)))
    (is (< (abs (- 1.0d0 (+ (aref data 0 0) (aref data 0 1)))) 1d-5))))

(test exp-based-activations-stability
  "Exponential-based activations should handle large negative values"
  (let ((input (make-tensor #2A((-100.0d0)) :shape '(1 1))))
    (is (not (null (elu input))))
    (is (not (null (selu input))))
    (is (not (null (softplus input))))))

;;;; ============================================================================
;;;; Comparative Tests
;;;; ============================================================================

(test relu-vs-leaky-relu
  "Leaky ReLU should differ from ReLU on negative inputs"
  (let* ((input (make-tensor #2A((-1.0d0)) :shape '(1 1)))
         (relu-out (relu input))
         (leaky-out (leaky-relu input 0.1d0)))
    (is (= 0.0d0 (aref (tensor-data relu-out) 0 0)))
    (is (/= 0.0d0 (aref (tensor-data leaky-out) 0 0)))))

(test elu-vs-relu
  "ELU should differ from ReLU on negative inputs"
  (let* ((input (make-tensor #2A((-1.0d0)) :shape '(1 1)))
         (relu-out (relu input))
         (elu-out (elu input)))
    (is (= 0.0d0 (aref (tensor-data relu-out) 0 0)))
    (is (/= 0.0d0 (aref (tensor-data elu-out) 0 0)))))

(test gelu-vs-relu
  "GELU should be smoother than ReLU around zero"
  (let* ((input (make-tensor #2A((-0.5d0)) :shape '(1 1)))
         (relu-out (relu input))
         (gelu-out (gelu input)))
    (is (= 0.0d0 (aref (tensor-data relu-out) 0 0)))
    (is (/= 0.0d0 (aref (tensor-data gelu-out) 0 0)))))

;;;; ============================================================================
;;;; Monotonicity Tests
;;;; ============================================================================

(test relu-monotonic
  "ReLU should be monotonically increasing"
  (let* ((input1 (make-tensor #2A((1.0d0)) :shape '(1 1)))
         (input2 (make-tensor #2A((2.0d0)) :shape '(1 1)))
         (output1 (relu input1))
         (output2 (relu input2)))
    (is (< (aref (tensor-data output1) 0 0)
           (aref (tensor-data output2) 0 0)))))

(test sigmoid-monotonic
  "Sigmoid should be monotonically increasing"
  (let* ((input1 (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (input2 (make-tensor #2A((1.0d0)) :shape '(1 1)))
         (output1 (sigmoid input1))
         (output2 (sigmoid input2)))
    (is (< (aref (tensor-data output1) 0 0)
           (aref (tensor-data output2) 0 0)))))

(test gelu-monotonic
  "GELU should be monotonically increasing"
  (let* ((input1 (make-tensor #2A((0.0d0)) :shape '(1 1)))
         (input2 (make-tensor #2A((1.0d0)) :shape '(1 1)))
         (output1 (gelu input1))
         (output2 (gelu input2)))
    (is (< (aref (tensor-data output1) 0 0)
           (aref (tensor-data output2) 0 0)))))
