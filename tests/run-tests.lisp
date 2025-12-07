;;;; tests/run-tests.lisp - Unified Test Runner

(in-package #:neural-lisp-tests)

;;; This file runs all test suites in the neural-lisp project

;;; Test result tracking
(defvar *total-tests-passed* 0)
(defvar *total-tests-failed* 0)
(defvar *suite-results* nil)

(defun print-section-header (title)
  "Print a styled section header"
  (format t "~%╔════════════════════════════════════════════════════════════════╗~%")
  (format t "║  ~64@<~A~>║~%" title)
  (format t "╚════════════════════════════════════════════════════════════════╝~%"))

(defun print-test-summary ()
  "Print final test summary"
  (format t "~%~%")
  (format t "╔════════════════════════════════════════════════════════════════╗~%")
  (format t "║  NEURAL-LISP TEST SUMMARY                                     ║~%")
  (format t "╠════════════════════════════════════════════════════════════════╣~%")
  (dolist (result (reverse *suite-results*))
    (let ((name (first result))
          (passed (second result))
          (failed (third result)))
      (format t "║  ~20A  Passed: ~4d  Failed: ~4d               ║~%" 
              name passed failed)))
  (format t "╠════════════════════════════════════════════════════════════════╣~%")
  (format t "║  TOTAL                Passed: ~4d  Failed: ~4d               ║~%"
          *total-tests-passed* *total-tests-failed*)
  (format t "╚════════════════════════════════════════════════════════════════╝~%")
  (if (= *total-tests-failed* 0)
      (format t "~%✓ All tests passed!~%~%")
      (format t "~%✗ Some tests failed!~%~%")))

(defun record-suite-result (suite-name passed failed)
  "Record the results of a test suite"
  (push (list suite-name passed failed) *suite-results*)
  (incf *total-tests-passed* passed)
  (incf *total-tests-failed* failed))

(defun count-test-results (result)
  "Count passed and failed tests from a FiveAM result object"
  (let ((passed 0)
        (failed 0))
    ;; FiveAM run returns a list of test result objects directly
    (dolist (test-result result)
      (if (typep test-result 'fiveam::test-passed)
          (incf passed)
          (incf failed)))
    (values passed failed)))

(defun cleanup-test-resources (&optional (verbose nil))
  "Clean up resources after each test suite to prevent heap exhaustion.
   
   This function:
   - Forces a full garbage collection to reclaim memory from tensors and arrays
   - Pauses briefly to allow GC to complete
   - Helps prevent heap exhaustion when running many test suites sequentially
   
   Each test suite creates many temporary tensors, gradients, and large arrays
   that can accumulate in memory. Regular cleanup between suites ensures tests
   can complete without running out of heap space."
  ;; Force garbage collection to free accumulated tensors and arrays
  (when verbose
    (format t "  [GC] Cleaning up resources...~%"))
  
  #+sbcl (sb-ext:gc :full t)
  #+ccl (ccl:gc)
  #+ecl (si:gc t)
  #+clisp (ext:gc)
  #+allegro (excl:gc t)
  #+lispworks (hcl:normal-gc)
  #-(or sbcl ccl ecl clisp allegro lispworks) (trivial-garbage:gc :full t)
  
  ;; Brief pause to allow GC to complete
  (sleep 0.1)
  
  (when verbose
    (format t "  [GC] Cleanup complete~%")))

(defun run-neural-lisp-tests (&key (verbose nil))
  "Run all neural-lisp test suites and report results"
  ;; Reset counters
  (setf *total-tests-passed* 0
        *total-tests-failed* 0
        *suite-results* nil)
  
  (format t "~%~%")
  (print-section-header "NEURAL-LISP TEST SUITE")
  (format t "~%")
  
  ;; Run core test suites individually for clarity
  (print-section-header "Core Tests")
  (format t "~%")
  
  ;; Run tensor tests
  (format t "Running Tensor Tests...~%")
  (let ((result (run 'tensor-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Tensor Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Tensor" passed failed)))
  (cleanup-test-resources)
  
  ;; Run operations tests
  (format t "Running Operations Tests...~%")
  (let ((result (run 'operations-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Operations Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Operations" passed failed)))
  (cleanup-test-resources)
  
  ;; Run layer tests
  (format t "Running Layer Tests...~%")
  (let ((result (run 'layer-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Layer Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Layers" passed failed)))
  (cleanup-test-resources)
  
  ;; Run autograd tests
  (format t "Running Autograd Tests...~%")
  (let ((result (run 'autograd-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Autograd Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Autograd" passed failed)))
  (cleanup-test-resources)
  
  ;; Run optimizer tests
  (format t "Running Optimizer Tests...~%")
  (let ((result (run 'optimizer-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Optimizer Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Optimizers" passed failed)))
  (cleanup-test-resources)
  
  ;; Run loss tests
  (format t "Running Loss Tests...~%")
  (let ((result (run 'loss-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Loss Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Losses" passed failed)))
  (cleanup-test-resources)
  
  ;; Run training tests
  (format t "Running Training Tests...~%")
  (let ((result (run 'training-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Training Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Training" passed failed)))
  (cleanup-test-resources)
  
  ;; Run backend tests
  (format t "Running Backend Tests...~%")
  (let ((result (run 'backend-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Backend Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Backend" passed failed)))
  (cleanup-test-resources)
  
  ;; Run convolution tests
  (format t "Running Convolution Tests...~%")
  (let ((result (run 'convolution-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Convolution Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Convolution" passed failed)))
  (cleanup-test-resources)
  
  ;; Run residual tests
  (format t "Running Residual Tests...~%")
  (let ((result (run 'residual-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Residual Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Residual" passed failed)))
  (cleanup-test-resources)
  
  ;; Run activation tests
  (format t "~%Running Activation Tests...~%")
  (let ((result (run 'activation-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Activation Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Activations" passed failed)))
  (cleanup-test-resources)
  
  ;; Run variational tests
  (format t "Running Variational Tests...~%")
  (let ((result (run 'variational-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Variational Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Variational" passed failed)))
  (cleanup-test-resources)
  
  ;; Run embedding tests
  (format t "Running Embedding Tests...~%")
  (let ((result (run 'embedding-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Embedding Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Embedding" passed failed)))
  (cleanup-test-resources)

  ;; Run checkpoint tests
  (format t "Running Checkpoint Tests...~%")
  (let ((result (run 'checkpoint-tests)))
    (when verbose (explain! result))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "  Checkpoint Tests: ~d passed, ~d failed~%" passed failed)
      (record-suite-result "Checkpoint" passed failed)))
  (cleanup-test-resources)
  
  ;; Run data loader tests
  (multiple-value-bind (passed failed)
      (run-data-loader-tests)
    (record-suite-result "Data Loaders" passed failed))
  (cleanup-test-resources)
  
  ;; Run normalization tests
  (multiple-value-bind (passed failed)
      (run-normalization-tests)
    (record-suite-result "Normalization" passed failed))
  (cleanup-test-resources)
  
  ;; Run advanced architecture tests
  (test-advanced-architectures)
  
  ;; Print final summary
  (print-test-summary)
  
  ;; Return exit code
  (if (= *total-tests-failed* 0) 0 1))

(defun run-suite (suite-name)
  "Run a specific test suite by name"
  (format t "~%Running ~A...~%" suite-name)
  (run suite-name))

;; Convenience functions for running individual suites
(defun test-tensors () (run-suite 'tensor-tests))
(defun test-operations () (run-suite 'operations-tests))
(defun test-layers () (run-suite 'layer-tests))
(defun test-autograd () (run-suite 'autograd-tests))
(defun test-optimizers () (run-suite 'optimizer-tests))
(defun test-losses () (run-suite 'loss-tests))
(defun test-training () (run-suite 'training-tests))
(defun test-backend () (run-suite 'backend-tests))
(defun test-convolution () (run-suite 'convolution-tests))
(defun test-residual () (run-suite 'residual-tests))
(defun test-activations () (run-suite 'activation-tests))
(defun test-variational () (run-suite 'variational-tests))
(defun test-embedding () (run-suite 'embedding-tests))

;; Functions for running advanced architecture tests
(defun test-recurrent ()
  "Run recurrent network tests"
  (format t "~%Loading recurrent network tests...~%")
  (load "tests/test-recurrent.lisp")
  (funcall (find-symbol "RUN-RECURRENT-TESTS" "NEURAL-TENSOR-RECURRENT-TESTS")))

(defun test-transformer ()
  "Run transformer tests"
  (format t "~%Loading transformer tests...~%")
  (load "tests/test-transformer.lisp")
  (funcall (find-symbol "RUN-TRANSFORMER-TESTS" "NEURAL-TENSOR-TRANSFORMER-TESTS")))

(defun test-crf ()
  "Run CRF tests"
  (format t "~%Loading CRF tests...~%")
  (load "tests/test-crf.lisp")
  (funcall (find-symbol "RUN-CRF-TESTS" "NEURAL-TENSOR-CRF-TESTS")))

(defun test-state-space ()
  "Run state space model tests"
  (format t "~%Loading state space model tests...~%")
  (load "tests/test-state-space.lisp")
  (funcall (find-symbol "RUN-SSM-TESTS" "NEURAL-TENSOR-SSM-TESTS")))

(defun test-benchmarks ()
  "Run performance benchmark tests"
  (format t "~%Loading benchmark tests...~%")
  (load "tests/test-benchmarks.lisp")
  (funcall (find-symbol "RUN-BENCHMARK-TESTS" "NEURAL-LISP-TESTS")))

(defun test-advanced-architectures ()
  "Run all advanced architecture tests"
  (print-section-header "Advanced Architectures")
  (format t "~%")
  
  (multiple-value-bind (passed failed) (test-recurrent)
    (record-suite-result "RNN/LSTM/GRU" passed failed))
  (cleanup-test-resources)
  
  (multiple-value-bind (passed failed) (test-transformer)
    (record-suite-result "Transformer" passed failed))
  (cleanup-test-resources)
  
  (multiple-value-bind (passed failed) (test-crf)
    (record-suite-result "CRF" passed failed))
  (cleanup-test-resources)
  
  (multiple-value-bind (passed failed) (test-state-space)
    (record-suite-result "State Space" passed failed))
  (cleanup-test-resources)
  
  (values))

;; Export test runner functions
(export '(run-neural-lisp-tests
          run-suite
          cleanup-test-resources
          test-tensors
          test-operations
          test-layers
          test-autograd
          test-optimizers
          test-losses
          test-training
          test-backend
          test-convolution
          test-residual
          test-activations
          test-variational
          test-embedding
          test-recurrent
          test-transformer
          test-crf
          test-state-space
          test-advanced-architectures
          test-benchmarks))
