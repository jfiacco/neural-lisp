;;;; tests/test-operations.lisp - Tensor Operations Tests

(in-package #:neural-lisp-tests)

(def-suite operations-tests
  :description "Tests for tensor operations"
  :in neural-lisp-tests)

(in-suite operations-tests)

(test tensor-addition
  "Test tensor addition"
  (let* ((t1 (make-tensor #(1.0 2.0 3.0) :shape '(3)))
         (t2 (make-tensor #(4.0 5.0 6.0) :shape '(3)))
         (result (t+ t1 t2)))
    (is (= 5.0 (aref (tensor-data result) 0)))
    (is (= 7.0 (aref (tensor-data result) 1)))
    (is (= 9.0 (aref (tensor-data result) 2)))))

(test tensor-addition-broadcast
  "Test tensor addition with broadcasting"
  (let* ((t1 (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2)))
         (t2 (make-tensor #2A((10.0 20.0)) :shape '(1 2)))
         (result (t+ t1 t2)))
    (is (= 11.0 (aref (tensor-data result) 0 0)))
    (is (= 22.0 (aref (tensor-data result) 0 1)))
    (is (= 13.0 (aref (tensor-data result) 1 0)))
    (is (= 24.0 (aref (tensor-data result) 1 1)))))

(test tensor-subtraction
  "Test tensor subtraction"
  (let* ((t1 (make-tensor #(5.0 6.0 7.0) :shape '(3)))
         (t2 (make-tensor #(1.0 2.0 3.0) :shape '(3)))
         (result (t- t1 t2)))
    (is (= 4.0 (aref (tensor-data result) 0)))
    (is (= 4.0 (aref (tensor-data result) 1)))
    (is (= 4.0 (aref (tensor-data result) 2)))))

(test tensor-multiplication
  "Test element-wise multiplication"
  (let* ((t1 (make-tensor #(2.0 3.0 4.0) :shape '(3)))
         (t2 (make-tensor #(5.0 6.0 7.0) :shape '(3)))
         (result (t* t1 t2)))
    (is (= 10.0 (aref (tensor-data result) 0)))
    (is (= 18.0 (aref (tensor-data result) 1)))
    (is (= 28.0 (aref (tensor-data result) 2)))))

(test tensor-multiplication-scalar-broadcast
  "Test multiplication with scalar broadcasting"
  (let* ((t1 (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (t2 (make-tensor #(3.0) :shape '(1)))
         (result (t* t1 t2)))
    (is (= 3.0 (aref (tensor-data result) 0 0)))
    (is (= 6.0 (aref (tensor-data result) 0 1)))))

(test matrix-multiplication
  "Test matrix multiplication"
  (let* ((t1 (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2)))
         (t2 (make-tensor #2A((5.0 6.0) (7.0 8.0)) :shape '(2 2)))
         (result (t@ t1 t2)))
    (is (= 19.0 (aref (tensor-data result) 0 0)))  ; 1*5 + 2*7
    (is (= 22.0 (aref (tensor-data result) 0 1)))  ; 1*6 + 2*8
    (is (= 43.0 (aref (tensor-data result) 1 0)))  ; 3*5 + 4*7
    (is (= 50.0 (aref (tensor-data result) 1 1))))) ; 3*6 + 4*8

(test tensor-sum
  "Test tensor sum"
  (let* ((t1 (make-tensor #(1.0 2.0 3.0 4.0) :shape '(4)))
         (result (tsum t1)))
    (is (= 10.0 (aref (tensor-data result) 0)))))

(test tensor-mean
  "Test tensor mean"
  (let* ((t1 (make-tensor #(2.0 4.0 6.0 8.0) :shape '(4)))
         (result (tmean t1)))
    (is (= 5.0 (aref (tensor-data result) 0)))))

(test relu-activation
  "Test ReLU activation"
  (let* ((t1 (make-tensor #(-2.0 -1.0 0.0 1.0 2.0) :shape '(5)))
         (result (relu t1)))
    (is (= 0.0 (aref (tensor-data result) 0)))
    (is (= 0.0 (aref (tensor-data result) 1)))
    (is (= 0.0 (aref (tensor-data result) 2)))
    (is (= 1.0 (aref (tensor-data result) 3)))
    (is (= 2.0 (aref (tensor-data result) 4)))))

(test sigmoid-activation
  "Test sigmoid activation"
  (let* ((t1 (make-tensor #(0.0) :shape '(1)))
         (result (sigmoid t1)))
    (is (< 0.499 (aref (tensor-data result) 0) 0.501))))

(test shape-mismatch-error
  "Test that shape mismatch throws error"
  (let ((t1 (make-tensor #(1.0 2.0) :shape '(2)))
        (t2 (make-tensor #(1.0 2.0 3.0) :shape '(3))))
    (signals simple-error (t+ t1 t2))))

;;; Edge Cases and Robustness Tests

(test addition-with-zeros
  "Test addition with zero tensors"
  (let* ((t1 (make-tensor #(1.0 2.0 3.0) :shape '(3)))
         (t2 (zeros '(3)))
         (result (t+ t1 t2)))
    (is (= 1.0 (aref (tensor-data result) 0)))
    (is (= 2.0 (aref (tensor-data result) 1)))
    (is (= 3.0 (aref (tensor-data result) 2)))))

(test addition-with-negative-values
  "Test addition with negative values"
  (let* ((t1 (make-tensor #(5.0 3.0 1.0) :shape '(3)))
         (t2 (make-tensor #(-2.0 -3.0 -4.0) :shape '(3)))
         (result (t+ t1 t2)))
    (is (= 3.0 (aref (tensor-data result) 0)))
    (is (= 0.0 (aref (tensor-data result) 1)))
    (is (= -3.0 (aref (tensor-data result) 2)))))

(test addition-commutative
  "Test that addition is commutative"
  (let* ((t1 (make-tensor #(1.0 2.0) :shape '(2)))
         (t2 (make-tensor #(3.0 4.0) :shape '(2)))
         (r1 (t+ t1 t2))
         (r2 (t+ t2 t1)))
    (is (= (aref (tensor-data r1) 0) (aref (tensor-data r2) 0)))
    (is (= (aref (tensor-data r1) 1) (aref (tensor-data r2) 1)))))

(test addition-associative
  "Test that addition is associative"
  (let* ((t1 (make-tensor #(1.0) :shape '(1)))
         (t2 (make-tensor #(2.0) :shape '(1)))
         (t3 (make-tensor #(3.0) :shape '(1)))
         (r1 (t+ (t+ t1 t2) t3))
         (r2 (t+ t1 (t+ t2 t3))))
    (is (< (abs (- (aref (tensor-data r1) 0) (aref (tensor-data r2) 0))) 1.0e-10))))

(test subtraction-self-is-zero
  "Test that subtracting a tensor from itself gives zero"
  (let* ((t1 (make-tensor #(5.0 10.0 15.0) :shape '(3)))
         (result (t- t1 t1)))
    (is (< (abs (aref (tensor-data result) 0)) 1.0e-10))
    (is (< (abs (aref (tensor-data result) 1)) 1.0e-10))
    (is (< (abs (aref (tensor-data result) 2)) 1.0e-10))))

(test subtraction-with-negative-values
  "Test subtraction with negative values"
  (let* ((t1 (make-tensor #(5.0 3.0) :shape '(2)))
         (t2 (make-tensor #(-2.0 -3.0) :shape '(2)))
         (result (t- t1 t2)))
    (is (= 7.0 (aref (tensor-data result) 0)))
    (is (= 6.0 (aref (tensor-data result) 1)))))

(test multiplication-with-zeros
  "Test multiplication with zero tensor"
  (let* ((t1 (make-tensor #(5.0 10.0 15.0) :shape '(3)))
         (t2 (zeros '(3)))
         (result (t* t1 t2)))
    (is (= 0.0 (aref (tensor-data result) 0)))
    (is (= 0.0 (aref (tensor-data result) 1)))
    (is (= 0.0 (aref (tensor-data result) 2)))))

(test multiplication-with-ones
  "Test multiplication with ones (identity)"
  (let* ((t1 (make-tensor #(5.0 10.0 15.0) :shape '(3)))
         (t2 (ones '(3)))
         (result (t* t1 t2)))
    (is (= 5.0 (aref (tensor-data result) 0)))
    (is (= 10.0 (aref (tensor-data result) 1)))
    (is (= 15.0 (aref (tensor-data result) 2)))))

(test multiplication-commutative
  "Test that element-wise multiplication is commutative"
  (let* ((t1 (make-tensor #(2.0 3.0) :shape '(2)))
         (t2 (make-tensor #(4.0 5.0) :shape '(2)))
         (r1 (t* t1 t2))
         (r2 (t* t2 t1)))
    (is (= (aref (tensor-data r1) 0) (aref (tensor-data r2) 0)))
    (is (= (aref (tensor-data r1) 1) (aref (tensor-data r2) 1)))))

(test multiplication-with-negative-values
  "Test multiplication with negative values"
  (let* ((t1 (make-tensor #(2.0 -3.0 4.0) :shape '(3)))
         (t2 (make-tensor #(-1.0 2.0 -5.0) :shape '(3)))
         (result (t* t1 t2)))
    (is (= -2.0 (aref (tensor-data result) 0)))
    (is (= -6.0 (aref (tensor-data result) 1)))
    (is (= -20.0 (aref (tensor-data result) 2)))))

(test matmul-non-square-matrices
  "Test matrix multiplication with non-square matrices"
  (let* ((t1 (make-tensor #2A((1.0 2.0 3.0) (4.0 5.0 6.0)) :shape '(2 3)))
         (t2 (make-tensor #2A((1.0 2.0) (3.0 4.0) (5.0 6.0)) :shape '(3 2)))
         (result (t@ t1 t2)))
    (is (equal '(2 2) (tensor-shape result)))
    ;; (1*1 + 2*3 + 3*5) = 22
    (is (= 22.0 (aref (tensor-data result) 0 0)))
    ;; (4*1 + 5*3 + 6*5) = 49
    (is (= 49.0 (aref (tensor-data result) 1 0)))))

(test matmul-vector-matrix
  "Test matrix multiplication with vector and matrix"
  (let* ((t1 (make-tensor #2A((1.0 2.0 3.0)) :shape '(1 3)))
         (t2 (make-tensor #2A((1.0) (2.0) (3.0)) :shape '(3 1)))
         (result (t@ t1 t2)))
    (is (equal '(1 1) (tensor-shape result)))
    ;; 1*1 + 2*2 + 3*3 = 14
    (is (= 14.0 (aref (tensor-data result) 0 0)))))

(test matmul-identity-matrix
  "Test matrix multiplication with identity matrix"
  (let* ((t1 (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2)))
         (identity (make-tensor #2A((1.0 0.0) (0.0 1.0)) :shape '(2 2)))
         (result (t@ t1 identity)))
    (is (= 1.0 (aref (tensor-data result) 0 0)))
    (is (= 2.0 (aref (tensor-data result) 0 1)))
    (is (= 3.0 (aref (tensor-data result) 1 0)))
    (is (= 4.0 (aref (tensor-data result) 1 1)))))

(test matmul-incompatible-shapes-error
  "Test that incompatible matrix shapes throw error"
  (let ((t1 (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
        (t2 (make-tensor #2A((1.0 2.0)) :shape '(1 2))))
    (signals simple-error (t@ t1 t2))))

(test sum-single-element
  "Test sum of single element tensor"
  (let* ((t1 (make-tensor #(42.0) :shape '(1)))
         (result (tsum t1)))
    (is (= 42.0 (aref (tensor-data result) 0)))))

(test sum-all-zeros
  "Test sum of all zeros"
  (let* ((t1 (zeros '(10)))
         (result (tsum t1)))
    (is (= 0.0 (aref (tensor-data result) 0)))))

(test sum-negative-values
  "Test sum with negative values"
  (let* ((t1 (make-tensor #(-1.0 -2.0 -3.0) :shape '(3)))
         (result (tsum t1)))
    (is (= -6.0 (aref (tensor-data result) 0)))))

(test sum-large-tensor
  "Test sum of large tensor"
  (let* ((t1 (ones '(1000)))
         (result (tsum t1)))
    (is (< 999.0 (aref (tensor-data result) 0) 1001.0))))

(test mean-single-element
  "Test mean of single element tensor"
  (let* ((t1 (make-tensor #(42.0) :shape '(1)))
         (result (tmean t1)))
    (is (= 42.0 (aref (tensor-data result) 0)))))

(test mean-negative-values
  "Test mean with negative values"
  (let* ((t1 (make-tensor #(-2.0 -4.0 -6.0) :shape '(3)))
         (result (tmean t1)))
    (is (= -4.0 (aref (tensor-data result) 0)))))

(test mean-all-same
  "Test mean when all values are the same"
  (let* ((t1 (make-tensor #(5.0 5.0 5.0 5.0) :shape '(4)))
         (result (tmean t1)))
    (is (= 5.0 (aref (tensor-data result) 0)))))

(test relu-all-positive
  "Test ReLU with all positive values"
  (let* ((t1 (make-tensor #(1.0 2.0 3.0) :shape '(3)))
         (result (relu t1)))
    (is (= 1.0 (aref (tensor-data result) 0)))
    (is (= 2.0 (aref (tensor-data result) 1)))
    (is (= 3.0 (aref (tensor-data result) 2)))))

(test relu-all-negative
  "Test ReLU with all negative values"
  (let* ((t1 (make-tensor #(-1.0 -2.0 -3.0) :shape '(3)))
         (result (relu t1)))
    (is (= 0.0 (aref (tensor-data result) 0)))
    (is (= 0.0 (aref (tensor-data result) 1)))
    (is (= 0.0 (aref (tensor-data result) 2)))))

(test relu-at-zero
  "Test ReLU exactly at zero"
  (let* ((t1 (make-tensor #(0.0 0.0) :shape '(2)))
         (result (relu t1)))
    (is (= 0.0 (aref (tensor-data result) 0)))
    (is (= 0.0 (aref (tensor-data result) 1)))))

(test sigmoid-extreme-positive
  "Test sigmoid with large positive value"
  (let* ((t1 (make-tensor #(10.0) :shape '(1)))
         (result (sigmoid t1)))
    ;; Should be very close to 1
    (is (> (aref (tensor-data result) 0) 0.9999))))

(test sigmoid-extreme-negative
  "Test sigmoid with large negative value"
  (let* ((t1 (make-tensor #(-10.0) :shape '(1)))
         (result (sigmoid t1)))
    ;; Should be very close to 0
    (is (< (aref (tensor-data result) 0) 0.0001))))

(test sigmoid-symmetry
  "Test sigmoid symmetry: sigmoid(-x) = 1 - sigmoid(x)"
  (let* ((t1 (make-tensor #(2.0) :shape '(1)))
         (t2 (make-tensor #(-2.0) :shape '(1)))
         (r1 (sigmoid t1))
         (r2 (sigmoid t2)))
    (is (< (abs (- (+ (aref (tensor-data r1) 0) (aref (tensor-data r2) 0)) 1.0)) 1.0e-10))))

(test broadcast-scalar-to-vector
  "Test broadcasting scalar to vector"
  (let* ((t1 (make-tensor #(5.0) :shape '(1)))
         (t2 (make-tensor #(1.0 2.0 3.0 4.0 5.0) :shape '(5)))
         (result (t* t1 t2)))
    (is (= 5.0 (aref (tensor-data result) 0)))
    (is (= 25.0 (aref (tensor-data result) 4)))))

(test broadcast-row-to-matrix
  "Test broadcasting row vector to matrix"
  (let* ((t1 (make-tensor #2A((1.0 2.0 3.0)) :shape '(1 3)))
         (t2 (make-tensor #2A((1.0 1.0 1.0) (2.0 2.0 2.0)) :shape '(2 3)))
         (result (t+ t1 t2)))
    (is (= 2.0 (aref (tensor-data result) 0 0)))
    (is (= 5.0 (aref (tensor-data result) 1 2)))))

(test operations-preserve-shape
  "Test that operations preserve expected output shape"
  (let* ((t1 (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2)))
         (t2 (make-tensor #2A((5.0 6.0) (7.0 8.0)) :shape '(2 2)))
         (add-result (t+ t1 t2))
         (sub-result (t- t1 t2))
         (mul-result (t* t1 t2)))
    (is (equal '(2 2) (tensor-shape add-result)))
    (is (equal '(2 2) (tensor-shape sub-result)))
    (is (equal '(2 2) (tensor-shape mul-result)))))
