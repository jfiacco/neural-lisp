;;;; tests/test-checkpoint.lisp - Checkpoint Save/Load Tests

(in-package #:neural-lisp-tests)

(def-suite checkpoint-tests
  :description "Tests for checkpoint serialization APIs"
  :in neural-lisp-tests)

(in-suite checkpoint-tests)

(defun tensor-values (tensor)
  "Return a flat list of the tensor's numerical data."
  (let* ((data (tensor-data tensor))
         (size (array-total-size data)))
    (loop for idx from 0 below size
          collect (row-major-aref data idx))))

(defun tensor-grad-values (tensor)
  "Return a flat list of the tensor's gradient data, if any."
  (let ((grad (tensor-grad tensor)))
    (when grad
      (let ((size (array-total-size grad)))
        (loop for idx from 0 below size
              collect (row-major-aref grad idx))))))

(defun set-tensor-values! (tensor values)
  "Overwrite TENSOR with VALUES in row-major order."
  (let* ((data (tensor-data tensor))
         (size (array-total-size data))
         (vals (coerce values 'list)))
    (assert (= size (length vals)) ()
            "Value count (~a) does not match tensor size (~a)"
            (length vals) size)
    (loop for val in vals
          for idx from 0
          do (setf (row-major-aref data idx) (coerce val 'double-float))))
  tensor)

(defun fill-tensor! (tensor value)
  "Fill TENSOR data with VALUE."
  (let* ((data (tensor-data tensor))
         (size (array-total-size data))
         (val (coerce value 'double-float)))
    (dotimes (idx size)
      (setf (row-major-aref data idx) val)))
  tensor)

(defmacro with-temp-checkpoint ((path) &body body)
  "Create a temporary checkpoint file PATH for BODY, cleaning it afterwards."
  `(let ((,path (merge-pathnames
                 (format nil "checkpoint-~a.ckpt" (gensym))
                 (temporary-directory))))
     (unwind-protect
         (progn ,@body)
       (when (probe-file ,path)
         (ignore-errors (delete-file ,path))))))

(test checkpoint-linear-roundtrip
  "Saving and loading a standard linear layer restores weights and bias."
  (with-temp-checkpoint (path)
    (let ((layer (linear 2 2)))
      (set-tensor-values! (weights layer)
                          '(0.5d0 1.5d0 -2.0d0 3.25d0))
      (set-tensor-values! (bias layer)
                          '(0.1d0 -0.7d0))
      (let ((expected-w (tensor-values (weights layer)))
            (expected-b (tensor-values (bias layer))))
        (save-checkpoint layer path :metadata '(:test :roundtrip))
        (fill-tensor! (weights layer) 42.0d0)
        (fill-tensor! (bias layer) -99.0d0)
        (load-checkpoint layer path)
        (is (equalp expected-w (tensor-values (weights layer))))
  (is (equalp expected-b (tensor-values (bias layer))))))))

(test checkpoint-strict-vs-lenient
  "Strict loading errors when parameters are missing; lenient loading ignores extras."
  (with-temp-checkpoint (path)
    (let* ((composite (sequential (linear 3 3)
                                  (linear 3 1)))
           (single (linear 3 3)))
      (save-checkpoint composite path)
      (signals error (load-checkpoint (weights single) path))
      ;; Lenient loading consumes only the first matching entry
      (fill-tensor! (weights single) 0.0d0)
      (load-checkpoint (weights single) path :strict nil)
  (is (notany #'zerop (tensor-values (weights single)))))))

(test checkpoint-includes-gradients
  "Gradients are optionally serialized and restored."
  (with-temp-checkpoint (path)
    (let ((tensor (make-tensor #(1.0d0 2.0d0)
                               :shape '(2)
                               :requires-grad t
                               :name "grad-test")))
      (let ((grad (tensor-grad tensor)))
        (setf (row-major-aref grad 0) 10.0d0)
        (setf (row-major-aref grad 1) -5.5d0))
      (save-checkpoint tensor path :include-grad t)
      ;; Mutate both data and grad to ensure they are restored
      (fill-tensor! tensor 0.0d0)
      (let ((grad (tensor-grad tensor)))
        (dotimes (idx (array-total-size grad))
          (setf (row-major-aref grad idx) 0.0d0)))
      (load-checkpoint tensor path)
      (is (equalp '(1.0d0 2.0d0) (tensor-values tensor)))
      (is (equalp '(10.0d0 -5.5d0) (tensor-grad-values tensor))))))
