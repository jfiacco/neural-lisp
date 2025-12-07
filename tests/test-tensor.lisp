;;;; tests/test-tensor.lisp - Tensor Creation and Basic Properties Tests

(in-package #:neural-lisp-tests)

(def-suite tensor-tests
  :description "Tests for tensor creation and properties"
  :in neural-lisp-tests)

(in-suite tensor-tests)

(test tensor-creation
  "Test basic tensor creation"
  (let ((t1 (make-tensor #(1.0 2.0 3.0) :shape '(3))))
    (is (equal '(3) (tensor-shape t1)))
    (is (= 1.0 (aref (tensor-data t1) 0)))
    (is (= 2.0 (aref (tensor-data t1) 1)))
    (is (= 3.0 (aref (tensor-data t1) 2)))))

(test tensor-2d-creation
  "Test 2D tensor creation"
  (let ((t1 (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2))))
    (is (equal '(2 2) (tensor-shape t1)))
    (is (= 1.0 (aref (tensor-data t1) 0 0)))
    (is (= 4.0 (aref (tensor-data t1) 1 1)))))

(test zeros-creation
  "Test zeros tensor creation"
  (let ((t1 (zeros '(2 3))))
    (is (equal '(2 3) (tensor-shape t1)))
    (is (= 0.0 (aref (tensor-data t1) 0 0)))
    (is (= 0.0 (aref (tensor-data t1) 1 2)))))

(test ones-creation
  "Test ones tensor creation"
  (let ((t1 (ones '(3 2))))
    (is (equal '(3 2) (tensor-shape t1)))
    (is (= 1.0 (aref (tensor-data t1) 0 0)))
    (is (= 1.0 (aref (tensor-data t1) 2 1)))))

(test randn-creation
  "Test random tensor creation"
  (let ((t1 (randn '(5 5))))
    (is (equal '(5 5) (tensor-shape t1)))
    (is (numberp (aref (tensor-data t1) 0 0)))
    ;; Check that values are different (not all same)
    (is (not (= (aref (tensor-data t1) 0 0)
                (aref (tensor-data t1) 1 1))))))

(test tensor-requires-grad
  "Test requires-grad flag"
  (let ((t1 (make-tensor #(1.0 2.0) :shape '(2) :requires-grad t))
        (t2 (make-tensor #(3.0 4.0) :shape '(2) :requires-grad nil)))
    (is (requires-grad t1))
    (is (not (requires-grad t2)))))

(test tensor-name
  "Test tensor naming"
  (let ((t1 (make-tensor #(1.0) :shape '(1) :name "my-tensor")))
    (is (string= "my-tensor" (tensor-name t1)))))

(test tensor-grad-initialization
  "Test gradient initialization"
  (let ((t1 (make-tensor #(1.0 2.0) :shape '(2) :requires-grad t)))
    (is (not (null (tensor-grad t1))))
    (is (= 0.0 (aref (tensor-grad t1) 0)))
    (is (= 0.0 (aref (tensor-grad t1) 1)))))

;;; Edge Cases and Robustness Tests

(test single-element-tensor
  "Test creation and manipulation of single-element tensors"
  (let ((t1 (make-tensor #(42.0) :shape '(1))))
    (is (equal '(1) (tensor-shape t1)))
    (is (= 42.0 (aref (tensor-data t1) 0)))
    (is (= 1 (array-total-size (tensor-data t1))))))

(test scalar-tensor
  "Test scalar-like tensor (shape ())"
  (let ((t1 (make-tensor #0A(3.14) :shape '())))
    (is (equal '() (tensor-shape t1)))
    (is (= 3.14 (aref (tensor-data t1))))))

(test high-dimensional-tensor
  "Test creation of high-dimensional tensors"
  (let ((t1 (zeros '(2 3 4 5))))
    (is (equal '(2 3 4 5) (tensor-shape t1)))
    (is (= (* 2 3 4 5) (array-total-size (tensor-data t1))))))

(test large-tensor-creation
  "Test creation of large tensors"
  (let ((t1 (zeros '(100 100))))
    (is (equal '(100 100) (tensor-shape t1)))
    (is (= 10000 (array-total-size (tensor-data t1))))))

(test tensor-with-negative-values
  "Test tensor with negative values"
  (let ((t1 (make-tensor #(-1.0 -2.0 -3.0) :shape '(3))))
    (is (= -1.0 (aref (tensor-data t1) 0)))
    (is (= -2.0 (aref (tensor-data t1) 1)))
    (is (= -3.0 (aref (tensor-data t1) 2)))))

(test tensor-with-zero-values
  "Test tensor filled with zeros"
  (let ((t1 (make-tensor #(0.0 0.0 0.0) :shape '(3))))
    (is (every (lambda (x) (= x 0.0))
               (loop for i below 3 collect (aref (tensor-data t1) i))))))

(test tensor-with-very-small-values
  "Test tensor with very small floating point values"
  (let ((t1 (make-tensor #(1.0e-10 1.0e-20) :shape '(2))))
    (is (< (aref (tensor-data t1) 0) 1.0e-9))
    (is (< (aref (tensor-data t1) 1) 1.0e-19))))

(test tensor-with-very-large-values
  "Test tensor with very large floating point values"
  (let ((t1 (make-tensor #(1.0e10 1.0e20) :shape '(2))))
    (is (> (aref (tensor-data t1) 0) 1.0e9))
    (is (> (aref (tensor-data t1) 1) 1.0e19))))

(test tensor-shape-consistency
  "Test that tensor shape matches actual data dimensions"
  (let ((t1 (make-tensor #2A((1.0 2.0 3.0) (4.0 5.0 6.0)) :shape '(2 3))))
    (is (equal '(2 3) (tensor-shape t1)))
    (is (equal '(2 3) (array-dimensions (tensor-data t1))))))

(test tensor-data-type-double-float
  "Test that tensor data is stored as double-float"
  (let ((t1 (make-tensor #(1.0 2.0) :shape '(2))))
    (is (eq 'double-float (array-element-type (tensor-data t1))))))

(test randn-statistical-properties
  "Test that randn produces values with reasonable statistical properties"
  (let* ((t1 (randn '(1000)))
         (data (tensor-data t1))
         (values (loop for i below 1000 collect (aref data i)))
         (mean (/ (reduce #'+ values) 1000))
         (variance (/ (reduce #'+ (mapcar (lambda (x) (expt (- x mean) 2)) values)) 1000)))
    ;; Mean should be close to 0 (within 0.2 for random sample)
    (is (< (abs mean) 0.2))
    ;; Variance should be close to 1 (within 0.3 for random sample)
    (is (< (abs (- variance 1.0)) 0.3))))

(test randn-uniqueness
  "Test that randn produces different values"
  (let ((t1 (randn '(10)))
        (t2 (randn '(10))))
    ;; At least one value should be different between two random tensors
    (is (not (equalp (tensor-data t1) (tensor-data t2))))))

(test zeros-all-elements
  "Test that zeros creates tensor with all zeros"
  (let ((t1 (zeros '(5 5))))
    (is (every (lambda (x) (= x 0.0d0))
               (loop for i below (array-total-size (tensor-data t1))
                     collect (row-major-aref (tensor-data t1) i))))))

(test ones-all-elements
  "Test that ones creates tensor with all ones"
  (let ((t1 (ones '(3 4))))
    (is (every (lambda (x) (= x 1.0d0))
               (loop for i below (array-total-size (tensor-data t1))
                     collect (row-major-aref (tensor-data t1) i))))))

(test tensor-grad-nil-when-no-grad
  "Test that gradient is nil when requires-grad is false"
  (let ((t1 (make-tensor #(1.0 2.0) :shape '(2) :requires-grad nil)))
    (is (null (tensor-grad t1)))))

(test tensor-grad-shape-matches-data
  "Test that gradient shape matches tensor shape"
  (let ((t1 (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2) :requires-grad t)))
    (is (equal (array-dimensions (tensor-data t1))
               (array-dimensions (tensor-grad t1))))))

(test tensor-name-default
  "Test that tensor name defaults to nil"
  (let ((t1 (make-tensor #(1.0) :shape '(1))))
    (is (null (tensor-name t1)))))

(test tensor-multiple-names
  "Test tensors can have different names"
  (let ((t1 (make-tensor #(1.0) :shape '(1) :name "tensor-1"))
        (t2 (make-tensor #(2.0) :shape '(1) :name "tensor-2")))
    (is (string= "tensor-1" (tensor-name t1)))
    (is (string= "tensor-2" (tensor-name t2)))
    (is (not (string= (tensor-name t1) (tensor-name t2))))))

(test tensor-3d-creation
  "Test 3D tensor creation"
  (let ((t1 (make-tensor #3A(((1.0 2.0) (3.0 4.0)) 
                             ((5.0 6.0) (7.0 8.0))) 
                         :shape '(2 2 2))))
    (is (equal '(2 2 2) (tensor-shape t1)))
    (is (= 1.0 (aref (tensor-data t1) 0 0 0)))
    (is (= 8.0 (aref (tensor-data t1) 1 1 1)))))

(test tensor-row-vector
  "Test row vector tensor"
  (let ((t1 (make-tensor #2A((1.0 2.0 3.0)) :shape '(1 3))))
    (is (equal '(1 3) (tensor-shape t1)))
    (is (= 3 (array-dimension (tensor-data t1) 1)))))

(test tensor-column-vector
  "Test column vector tensor"
  (let ((t1 (make-tensor #2A((1.0) (2.0) (3.0)) :shape '(3 1))))
    (is (equal '(3 1) (tensor-shape t1)))
    (is (= 3 (array-dimension (tensor-data t1) 0)))))

(test tensor-backward-fn-nil-initially
  "Test that backward function is nil for leaf tensors"
  (let ((t1 (make-tensor #(1.0) :shape '(1) :requires-grad t)))
    (is (null (tensor-backward-fn t1)))))

(test tensor-independence
  "Test that modifying one tensor doesn't affect another"
  (let ((t1 (make-tensor #(1.0 2.0) :shape '(2)))
        (t2 (make-tensor #(1.0 2.0) :shape '(2))))
    (setf (aref (tensor-data t1) 0) 99.0d0)
    (is (= 99.0d0 (aref (tensor-data t1) 0)))
    (is (= 1.0d0 (aref (tensor-data t2) 0)))))
