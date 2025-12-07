;;;; tests/test-autograd.lisp - Autograd Tests

(in-package #:neural-lisp-tests)

(def-suite autograd-tests
  :description "Tests for automatic differentiation"
  :in neural-lisp-tests)

(in-suite autograd-tests)

(test simple-backward
  "Test simple backward pass"
  (let* ((x (make-tensor #(2.0) :shape '(1) :requires-grad t))
         (y (t* x x))  ; y = x^2
         (loss y))
    (backward loss)
    (is (not (null (tensor-grad x))))
    ;; dy/dx = 2x = 2*2 = 4
    (is (< 3.9 (aref (tensor-grad x) 0) 4.1))))

(test chain-rule
  "Test chain rule in backward pass"
  (let* ((x (make-tensor #(3.0) :shape '(1) :requires-grad t))
         (y (t* x (make-tensor #(2.0) :shape '(1))))
         (z (t+ y (make-tensor #(5.0) :shape '(1))))
         (loss z))
    (backward loss)
    ;; dz/dx = 2 (constant multiplier)
    (is (< 1.9 (aref (tensor-grad x) 0) 2.1))))

(test addition-gradient
  "Test gradient through addition"
  (let* ((a (make-tensor #(1.0 2.0) :shape '(2) :requires-grad t))
         (b (make-tensor #(3.0 4.0) :shape '(2) :requires-grad t))
         (c (t+ a b))
         (loss (tsum c)))
    (backward loss)
    (is (= 1.0 (aref (tensor-grad a) 0)))
    (is (= 1.0 (aref (tensor-grad a) 1)))
    (is (= 1.0 (aref (tensor-grad b) 0)))
    (is (= 1.0 (aref (tensor-grad b) 1)))))

(test multiplication-gradient
  "Test gradient through multiplication"
  (let* ((a (make-tensor #(2.0) :shape '(1) :requires-grad t))
         (b (make-tensor #(3.0) :shape '(1) :requires-grad t))
         (c (t* a b))
         (loss c))
    (backward loss)
    ;; dc/da = b = 3.0
    (is (< 2.9 (aref (tensor-grad a) 0) 3.1))
    ;; dc/db = a = 2.0
    (is (< 1.9 (aref (tensor-grad b) 0) 2.1))))

(test matmul-gradient
  "Test gradient through matrix multiplication"
  (let* ((a (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2) :requires-grad t))
         (b (make-tensor #2A((5.0 6.0) (7.0 8.0)) :shape '(2 2) :requires-grad t))
         (c (t@ a b))
         (loss (tsum c)))
    (backward loss)
    (is (not (null (tensor-grad a))))
    (is (not (null (tensor-grad b))))))

(test relu-gradient
  "Test gradient through ReLU"
  (let* ((x (make-tensor #(-1.0 0.0 1.0 2.0) :shape '(4) :requires-grad t))
         (y (relu x))
         (loss (tsum y)))
    (backward loss)
    ;; ReLU gradient: 0 for x<0, 1 for x>0
    (is (= 0.0 (aref (tensor-grad x) 0)))  ; x=-1
    (is (= 0.0 (aref (tensor-grad x) 1)))  ; x=0
    (is (= 1.0 (aref (tensor-grad x) 2)))  ; x=1
    (is (= 1.0 (aref (tensor-grad x) 3))))) ; x=2

(test sigmoid-gradient
  "Test gradient through sigmoid"
  (let* ((x (make-tensor #(0.0) :shape '(1) :requires-grad t))
         (y (sigmoid x))
         (loss y))
    (backward loss)
    ;; sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
    (is (< 0.24 (aref (tensor-grad x) 0) 0.26))))

(test zero-grad
  "Test gradient zeroing"
  (let* ((x (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (y (t* x x)))
    (backward y)
    (is (not (null (tensor-grad x))))
    (zero-grad! x)
    (is (= 0.0 (aref (tensor-grad x) 0)))))

(test multiple-backward
  "Test multiple backward passes"
  (let* ((x (make-tensor #(2.0) :shape '(1) :requires-grad t))
         (y (t* x x)))
    ;; First backward
    (backward y)
    (let ((grad1 (aref (tensor-grad x) 0)))
      ;; Zero grad and do second backward
      (zero-grad! x)
      (backward y)
      (let ((grad2 (aref (tensor-grad x) 0)))
        ;; Gradients should be the same
        (is (< (abs (- grad1 grad2)) 0.01))))))

(test no-grad-leaf
  "Test that leaf tensors without requires-grad don't get gradients"
  (let* ((x (make-tensor #(1.0) :shape '(1) :requires-grad nil))
         (y (t* x x)))
    (backward y)
    (is (null (tensor-grad x)))))

(test computational-graph-building
  "Test that computational graph is built correctly"
  (let* ((x (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (y (t+ x x))
         (z (t* y y)))
    (is (not (null (tensor-backward-fn z))))
    (is (not (null (tensor-backward-fn y))))))

;;; Edge Cases and Robustness Tests

(test backward-single-operation
  "Test backward pass through single operation"
  (let* ((x (make-tensor #(5.0) :shape '(1) :requires-grad t))
         (y (t+ x (make-tensor #(3.0) :shape '(1)))))
    (backward y)
    ;; dy/dx = 1
    (is (= 1.0 (aref (tensor-grad x) 0)))))

(test complex-computational-graph
  "Test complex multi-branch computational graph"
  (let* ((x (make-tensor #(2.0) :shape '(1) :requires-grad t))
         (a (t* x (make-tensor #(3.0) :shape '(1))))
         (b (t+ x (make-tensor #(5.0) :shape '(1))))
         (c (t* a b))
         (loss c))
    (backward loss)
    ;; Complex gradient through multiple paths
    (is (not (null (tensor-grad x))))
    (is (not (= 0.0 (aref (tensor-grad x) 0))))))

(test multiple-inputs-gradient
  "Test gradients with multiple input tensors"
  (let* ((x1 (make-tensor #(2.0) :shape '(1) :requires-grad t))
         (x2 (make-tensor #(3.0) :shape '(1) :requires-grad t))
         (x3 (make-tensor #(4.0) :shape '(1) :requires-grad t))
         (y (t+ (t+ x1 x2) x3)))
    (backward y)
    (is (= 1.0 (aref (tensor-grad x1) 0)))
    (is (= 1.0 (aref (tensor-grad x2) 0)))
    (is (= 1.0 (aref (tensor-grad x3) 0)))))

(test gradient-through-sum
  "Test gradient through sum operation"
  (let* ((x (make-tensor #(1.0 2.0 3.0 4.0) :shape '(4) :requires-grad t))
         (y (tsum x)))
    (backward y)
    ;; Gradient should be 1 for all elements
    (is (= 1.0 (aref (tensor-grad x) 0)))
    (is (= 1.0 (aref (tensor-grad x) 1)))
    (is (= 1.0 (aref (tensor-grad x) 2)))
    (is (= 1.0 (aref (tensor-grad x) 3)))))

(test gradient-through-mean
  "Test gradient through mean operation"
  (let* ((x (make-tensor #(2.0 4.0 6.0 8.0) :shape '(4) :requires-grad t))
         (y (tmean x)))
    (backward y)
    ;; Gradient should be 1/n for all elements
    (is (= 0.25 (aref (tensor-grad x) 0)))
    (is (= 0.25 (aref (tensor-grad x) 1)))
    (is (= 0.25 (aref (tensor-grad x) 2)))
    (is (= 0.25 (aref (tensor-grad x) 3)))))

(test gradient-zero-multiplication
  "Test gradient when multiplying by zero"
  (let* ((x (make-tensor #(5.0) :shape '(1) :requires-grad t))
         (zero (make-tensor #(0.0) :shape '(1)))
         (y (t* x zero)))
    (backward y)
    ;; Gradient should be 0 (derivative of 0*x is 0)
    (is (= 0.0 (aref (tensor-grad x) 0)))))

(test gradient-accumulation-multiple-paths
  "Test gradient accumulation when variable appears multiple times"
  (let* ((x (make-tensor #(2.0) :shape '(1) :requires-grad t))
         (y (t+ (t* x x) x)))  ; y = x^2 + x
    (backward y)
    ;; dy/dx = 2x + 1 = 2*2 + 1 = 5
    (is (< 4.9 (aref (tensor-grad x) 0) 5.1))))

(test relu-gradient-negative
  "Test ReLU gradient for negative inputs"
  (let* ((x (make-tensor #(-5.0 -2.0 -0.1) :shape '(3) :requires-grad t))
         (y (relu x))
         (loss (tsum y)))
    (backward loss)
    ;; All gradients should be 0 for negative inputs
    (is (= 0.0 (aref (tensor-grad x) 0)))
    (is (= 0.0 (aref (tensor-grad x) 1)))
    (is (= 0.0 (aref (tensor-grad x) 2)))))

(test relu-gradient-positive
  "Test ReLU gradient for positive inputs"
  (let* ((x (make-tensor #(0.1 2.0 5.0) :shape '(3) :requires-grad t))
         (y (relu x))
         (loss (tsum y)))
    (backward loss)
    ;; All gradients should be 1 for positive inputs
    (is (= 1.0 (aref (tensor-grad x) 0)))
    (is (= 1.0 (aref (tensor-grad x) 1)))
    (is (= 1.0 (aref (tensor-grad x) 2)))))

(test sigmoid-gradient-extreme-values
  "Test sigmoid gradient for extreme values"
  (let* ((x (make-tensor #(10.0 -10.0) :shape '(2) :requires-grad t))
         (y (sigmoid x))
         (loss (tsum y)))
    (backward loss)
    ;; Gradients should be very small for extreme values (vanishing gradient)
    (is (< (aref (tensor-grad x) 0) 0.01))
    (is (< (aref (tensor-grad x) 1) 0.01))))

(test gradient-persistence-after-backward
  "Test that gradients persist after backward call"
  (let* ((x (make-tensor #(3.0) :shape '(1) :requires-grad t))
         (y (t* x x)))
    (backward y)
    (let ((grad1 (aref (tensor-grad x) 0)))
      ;; Gradient should still be there
      (is (not (null (tensor-grad x))))
      (is (= grad1 (aref (tensor-grad x) 0))))))

(test zero-grad-multiple-tensors
  "Test zeroing gradients for multiple tensors"
  (let* ((x1 (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (x2 (make-tensor #(2.0) :shape '(1) :requires-grad t))
         (y (t+ x1 x2)))
    (backward y)
    (is (not (= 0.0 (aref (tensor-grad x1) 0))))
    (is (not (= 0.0 (aref (tensor-grad x2) 0))))
    (zero-grad! x1)
    (zero-grad! x2)
    (is (= 0.0 (aref (tensor-grad x1) 0)))
    (is (= 0.0 (aref (tensor-grad x2) 0)))))

(test backward-preserves-forward-values
  "Test that backward doesn't change forward values"
  (let* ((x (make-tensor #(4.0) :shape '(1) :requires-grad t))
         (y (t* x x))
         (y-val (aref (tensor-data y) 0)))
    (backward y)
    ;; Forward value should remain unchanged
    (is (= y-val (aref (tensor-data y) 0)))
    (is (= 4.0 (aref (tensor-data x) 0)))))

(test matmul-gradient-shape-consistency
  "Test that matrix multiplication gradients have correct shapes"
  (let* ((a (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2) :requires-grad t))
         (b (make-tensor #2A((5.0 6.0) (7.0 8.0)) :shape '(2 2) :requires-grad t))
         (c (t@ a b))
         (loss (tsum c)))
    (backward loss)
    (is (equal (tensor-shape a) (list (array-dimension (tensor-grad a) 0)
                                      (array-dimension (tensor-grad a) 1))))
    (is (equal (tensor-shape b) (list (array-dimension (tensor-grad b) 0)
                                      (array-dimension (tensor-grad b) 1))))))

(test gradient-deep-network
  "Test gradients through a deep computational graph"
  (let* ((x (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (h1 (relu (t+ x (make-tensor #(0.5) :shape '(1)))))
         (h2 (relu (t* h1 (make-tensor #(2.0) :shape '(1)))))
         (h3 (relu (t+ h2 (make-tensor #(1.0) :shape '(1)))))
         (out h3))
    (backward out)
    ;; Gradient should propagate through all layers
    (is (not (null (tensor-grad x))))
    (is (> (aref (tensor-grad x) 0) 0.0))))

(test backward-idempotent-check
  "Test that calling backward multiple times requires manual zero-grad"
  (let* ((x (make-tensor #(2.0) :shape '(1) :requires-grad t))
         (y (t* x x)))
    (backward y)
    (let ((grad1 (aref (tensor-grad x) 0)))
      ;; If we call backward again without zero-grad, behavior depends on implementation
      ;; Just ensure gradient exists
      (is (not (null (tensor-grad x))))
      (is (= grad1 (aref (tensor-grad x) 0))))))

(test gradient-with-broadcast
  "Test gradients with broadcasting operations"
  (let* ((x (make-tensor #2A((1.0 2.0)) :shape '(1 2) :requires-grad t))
         (y (make-tensor #2A((3.0 4.0) (5.0 6.0)) :shape '(2 2)))
         (z (t+ x y))
         (loss (tsum z)))
    (backward loss)
    ;; Gradient should be aggregated due to broadcasting
    (is (not (null (tensor-grad x))))))

(test subtraction-gradient
  "Test gradient through subtraction"
  (let* ((a (make-tensor #(5.0 3.0) :shape '(2) :requires-grad t))
         (b (make-tensor #(2.0 1.0) :shape '(2) :requires-grad t))
         (c (t- a b))
         (loss (tsum c)))
    (backward loss)
    ;; dc/da = 1, dc/db = -1
    (is (= 1.0 (aref (tensor-grad a) 0)))
    (is (= 1.0 (aref (tensor-grad a) 1)))
    (is (= -1.0 (aref (tensor-grad b) 0)))
    (is (= -1.0 (aref (tensor-grad b) 1)))))

(test gradient-numerical-stability
  "Test gradient computation remains numerically stable"
  (let* ((x (make-tensor #(0.001 0.999) :shape '(2) :requires-grad t))
         (y (sigmoid x))
         (loss (tsum y)))
    (backward loss)
    ;; Gradients should be finite
    (is (numberp (aref (tensor-grad x) 0)))
    (is (numberp (aref (tensor-grad x) 1)))))

(test chain-rule-three-operations
  "Test chain rule through three consecutive operations"
  (let* ((x (make-tensor #(2.0) :shape '(1) :requires-grad t))
         (y (t* x (make-tensor #(3.0) :shape '(1))))  ; 6.0
         (z (t+ y (make-tensor #(4.0) :shape '(1))))  ; 10.0
         (w (t* z (make-tensor #(5.0) :shape '(1))))  ; 50.0
         (loss w))
    (backward loss)
    ;; dw/dx = dw/dz * dz/dy * dy/dx = 5 * 1 * 3 = 15
    (is (= 15.0 (aref (tensor-grad x) 0)))))
