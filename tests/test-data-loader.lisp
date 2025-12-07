;;;; tests/test-data-loader.lisp - Data Loader Tests

(in-package :neural-lisp-tests)

;;;; ============================================================================
;;;; Data Loader Test Suite
;;;; ============================================================================

(defvar *data-loader-tests-passed* 0)
(defvar *data-loader-tests-failed* 0)

(defmacro deftest-dl (name &body body)
  `(progn
     (format t "Testing ~a... " ',name)
     (handler-case
         (progn
           ,@body
           (format t "✓ PASSED~%")
           (incf *data-loader-tests-passed*))
       (error (e)
         (format t "✗ FAILED: ~a~%" e)
         (incf *data-loader-tests-failed*)))))

(defun run-data-loader-tests ()
  "Run all data loader tests"
  (setf *data-loader-tests-passed* 0
        *data-loader-tests-failed* 0)
  
  (format t "~%")
  (format t "╔════════════════════════════════════════════════════════════════╗~%")
  (format t "║  Data Loader Tests                                            ║~%")
  (format t "╚════════════════════════════════════════════════════════════════╝~%")
  (format t "~%")
  
  ;; Basic Dataset Tests
  (deftest-dl test-dataset-creation
    (let ((ds (make-instance 'neural-data-loader:dataset
                            :data '((1 2 3) (4 5 6) (7 8 9))
                            :labels '(0 1 0))))
      (assert (= 3 (neural-data-loader:dataset-length ds)))))
  
  (deftest-dl test-dataset-get-item
    (let ((ds (make-instance 'neural-data-loader:dataset
                            :data '((1 2 3) (4 5 6))
                            :labels '(0 1))))
      (multiple-value-bind (item label)
          (neural-data-loader:get-item ds 0)
        (assert (equal '(1 2 3) item))
        (assert (= 0 label)))))
  
  (deftest-dl test-dataset-no-labels
    (let ((ds (make-instance 'neural-data-loader:dataset
                            :data '((1 2 3) (4 5 6)))))
      (let ((item (neural-data-loader:get-item ds 1)))
        (assert (equal '(4 5 6) item)))))
  
  (deftest-dl test-dataset-with-transform
    (let ((ds (make-instance 'neural-data-loader:dataset
                            :data '((1 2) (3 4))
                            :transform #'(lambda (x) (mapcar #'1+ x)))))
      (let ((item (neural-data-loader:get-item ds 0)))
        (assert (equal '(2 3) item)))))
  
  ;; Sequence Dataset Tests
  (deftest-dl test-sequence-dataset-creation
    (let ((ds (make-instance 'neural-data-loader:sequence-dataset
                            :data '((1 2 3) (4 5) (6 7 8 9))
                            :max-length 10
                            :vocab-size 100)))
      (assert (= 3 (neural-data-loader:dataset-length ds)))
      (assert (= 10 (neural-data-loader:max-length ds)))
      (assert (= 100 (neural-data-loader:vocab-size ds)))))
  
  (deftest-dl test-text-dataset
    (let ((ds (make-instance 'neural-data-loader:text-dataset
                            :data '("hello world" "foo bar")
                            :tokenizer #'(lambda (s) (list (length s)))
                            :max-length 50)))
      (assert (= 2 (neural-data-loader:dataset-length ds)))))
  
  ;; Data Loader Tests
  (deftest-dl test-data-loader-creation
    (let* ((ds (make-instance 'neural-data-loader:dataset
                             :data '((1 2) (3 4) (5 6) (7 8))))
           (loader (neural-data-loader:make-data-loader ds :batch-size 2)))
      (assert (= 2 (neural-data-loader:batch-size loader)))
      (assert (= 2 (neural-data-loader:num-batches loader)))))
  
  (deftest-dl test-data-loader-batch-iteration
    (let* ((ds (make-instance 'neural-data-loader:dataset
                             :data '((1 2) (3 4) (5 6))))
           (loader (neural-data-loader:make-data-loader ds :batch-size 2))
           (batch-count 0))
      (loop for batch = (neural-data-loader:get-batch loader)
            while batch
            do (incf batch-count))
      (assert (= 2 batch-count)))) ; 2 full batches, last incomplete
  
  (deftest-dl test-data-loader-drop-last
    (let* ((ds (make-instance 'neural-data-loader:dataset
                             :data '((1 2) (3 4) (5 6))))
           (loader (neural-data-loader:make-data-loader ds 
                                                        :batch-size 2
                                                        :drop-last t))
           (batch-count 0))
      (loop for batch = (neural-data-loader:get-batch loader)
            while batch
            do (incf batch-count))
      (assert (= 1 batch-count)))) ; Only 1 complete batch
  
  (deftest-dl test-data-loader-reset
    (let* ((ds (make-instance 'neural-data-loader:dataset
                             :data '((1 2) (3 4) (5 6) (7 8))))
           (loader (neural-data-loader:make-data-loader ds :batch-size 2)))
      ;; First pass
      (loop for batch = (neural-data-loader:get-batch loader)
            while batch
            count t)
      ;; Reset and second pass should work
      (neural-data-loader:reset-loader loader)
      (let ((first-batch (neural-data-loader:get-batch loader)))
        (assert (not (null first-batch))))))
  
  (deftest-dl test-shuffle-indices
    (let ((indices (neural-data-loader::shuffle-indices 10)))
      (assert (= 10 (length indices)))
      (assert (every #'(lambda (i) (and (>= i 0) (< i 10))) indices))
      ;; All unique
      (assert (= 10 (length (remove-duplicates indices))))))
  
  ;; Collation Tests
  (deftest-dl test-collate-numbers
    (let ((batch (neural-data-loader::default-collate '(1 2 3) nil nil)))
      (assert (typep batch 'neural-network:tensor))
      (assert (equal '(3) (neural-network::tensor-shape batch)))))
  
  (deftest-dl test-collate-sequences
    (let ((batch (neural-data-loader::collate-sequences 
                  '((1 2 3) (4 5) (6 7 8 9)) nil)))
      (assert (typep batch 'neural-network:tensor))
      (assert (equal '(3 4) (neural-network::tensor-shape batch))))) ; 3 seqs, max len 4
  
  (deftest-dl test-stack-tensors
    (let* ((t1 (randn '(2 3)))
           (t2 (randn '(2 3)))
           (t3 (randn '(2 3)))
           (stacked (neural-data-loader::stack-tensors (list t1 t2 t3))))
      (assert (typep stacked 'neural-network:tensor))
      (assert (equal '(3 2 3) (neural-network::tensor-shape stacked)))))
  
  ;; Utility Function Tests
  (deftest-dl test-pad-sequences
    (let ((padded (neural-data-loader:pad-sequences 
                   '((1 2) (3 4 5) (6))
                   :max-length 5
                   :pad-value 0)))
      (assert (= 3 (length padded)))
      (assert (every #'(lambda (seq) (= 5 (length seq))) padded))
      (assert (equal '(1 2 0 0 0) (first padded)))))
  
  (deftest-dl test-create-attention-mask
    (let ((mask (neural-data-loader:create-attention-mask '(3 2 4) 5)))
      (assert (typep mask 'neural-network:tensor))
      (assert (equal '(3 5) (neural-network::tensor-shape mask)))
      (let ((data (neural-network::tensor-data mask)))
        ;; First sequence: 3 valid positions
        (assert (= 1.0d0 (aref data 0 0)))
        (assert (= 1.0d0 (aref data 0 2)))
        (assert (= 0.0d0 (aref data 0 3)))
        ;; Second sequence: 2 valid positions
        (assert (= 1.0d0 (aref data 1 1)))
        (assert (= 0.0d0 (aref data 1 2))))))
  
  (deftest-dl test-normalize-batch
    (let* ((batch (make-tensor (make-array '(2 3) 
                                           :element-type 'double-float
                                           :initial-contents '((2.0d0 4.0d0 6.0d0)
                                                              (8.0d0 10.0d0 12.0d0)))))
           (normalized (neural-data-loader:normalize-batch batch 
                                                           :mean 5.0d0 
                                                           :std 2.0d0)))
      (assert (typep normalized 'neural-network:tensor))
      (let ((data (neural-network::tensor-data normalized)))
        ;; (2 - 5) / 2 = -1.5
        (assert (< (abs (+ (aref data 0 0) 1.5d0)) 1d-6)))))
  
  (deftest-dl test-shuffle-dataset
    (let ((ds (make-instance 'neural-data-loader:dataset
                            :data '((1) (2) (3) (4) (5))
                            :labels '(0 1 0 1 0))))
      (neural-data-loader:shuffle-dataset ds)
      ;; Still same length
      (assert (= 5 (neural-data-loader:dataset-length ds)))
      ;; Data and labels still paired (sum should be same)
      (assert (= 5 (length (neural-data-loader::dataset-data ds))))
      (assert (= 5 (length (neural-data-loader::dataset-labels ds))))))
  
  ;; Integration Tests
  (deftest-dl test-integration-simple-training-loop
    (let* ((ds (make-instance 'neural-data-loader:dataset
                             :data '((1 2) (3 4) (5 6) (7 8))
                             :labels '(0 1 0 1)))
           (loader (neural-data-loader:make-data-loader ds :batch-size 2))
           (epochs-completed 0))
      (dotimes (epoch 2)
        (neural-data-loader:reset-loader loader)
        (loop for (batch-x batch-y) = (multiple-value-list 
                                        (neural-data-loader:get-batch loader))
              while batch-x
              count t)
        (incf epochs-completed))
      (assert (= 2 epochs-completed))))
  
  (deftest-dl test-integration-with-rnn
    (let* ((sequences '((1 2 3) (4 5 6) (7 8 9) (10 11 12)))
           (labels '(6 15 24 33))
           (ds (make-instance 'neural-data-loader:sequence-dataset
                             :data sequences
                             :labels labels))
           (loader (neural-data-loader:make-data-loader ds :batch-size 2)))
      (neural-data-loader:reset-loader loader)
      (let ((batch (neural-data-loader:get-batch loader)))
        (assert (not (null batch)))
        (assert (typep batch 'neural-network:tensor)))))
  
  (deftest-dl test-integration-variable-length-sequences
    (let* ((sequences '((1 2) (3 4 5 6) (7)))
           (ds (make-instance 'neural-data-loader:sequence-dataset
                             :data sequences))
           (loader (neural-data-loader:make-data-loader ds :batch-size 3)))
      (let ((batch (neural-data-loader:get-batch loader)))
        (assert (not (null batch)))
        ;; Should be padded to max length (4)
        (assert (equal '(3 4) (neural-network::tensor-shape batch))))))
  
  ;; Edge Cases
  (deftest-dl test-empty-dataset
    (let ((ds (make-instance 'neural-data-loader:dataset :data '())))
      (assert (= 0 (neural-data-loader:dataset-length ds)))))
  
  (deftest-dl test-single-item-dataset
    (let* ((ds (make-instance 'neural-data-loader:dataset
                             :data '((1 2 3))))
           (loader (neural-data-loader:make-data-loader ds :batch-size 10)))
      (let ((batch (neural-data-loader:get-batch loader)))
        (assert (not (null batch))))))
  
  (deftest-dl test-batch-size-larger-than-dataset
    (let* ((ds (make-instance 'neural-data-loader:dataset
                             :data '((1) (2))))
           (loader (neural-data-loader:make-data-loader ds :batch-size 10))
           (batch-count 0))
      (loop for batch = (neural-data-loader:get-batch loader)
            while batch
            do (incf batch-count))
      (assert (= 1 batch-count))))
  
  ;; Print summary
  (format t "~%Data Loader Tests: ~d passed, ~d failed~%~%" 
          *data-loader-tests-passed* *data-loader-tests-failed*)
  
  (values *data-loader-tests-passed* *data-loader-tests-failed*))

;; Export the test runner
(export 'run-data-loader-tests)
