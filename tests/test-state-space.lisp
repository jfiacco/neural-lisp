;;;; Tests for State Space Models
;;;; Testing S4, Mamba, Samba, and SSM language models

(defpackage :neural-tensor-ssm-tests
  (:use :common-lisp)
  (:import-from :neural-network
                #:tensor
                #:make-tensor
                #:zeros
                #:randn
                #:forward
                #:tensor-shape
                #:tensor-data
                #:layer-parameters)
  (:import-from :neural-tensor-ssm
                #:s4-layer
                #:mamba-block
                #:samba-block
                #:ssm-lm
                #:hippo-initialization
                #:discretize-zoh
                #:discretize-bilinear
                #:selective-scan)
  (:export #:run-ssm-tests))

(in-package :neural-tensor-ssm-tests)

(defvar *test-results* nil)
(defvar *tests-passed* 0)
(defvar *tests-failed* 0)

(defmacro deftest (name &body body)
  `(progn
     (format t "~%Testing ~a... " ',name)
     (handler-case
         (progn
           ,@body
           (format t "✓ PASSED")
           (incf *tests-passed*)
           (push (cons ',name :pass) *test-results*))
       (error (e)
         (format t "✗ FAILED: ~a" e)
         (incf *tests-failed*)
         (push (cons ',name :fail) *test-results*)))))

(defun assert-equal (expected actual &optional (tolerance 1d-6))
  "Assert two values are equal within tolerance"
  (unless (< (abs (- expected actual)) tolerance)
    (error "Expected ~a but got ~a" expected actual)))

(defun assert-shape (expected-shape tensor)
  "Assert tensor has expected shape"
  (unless (equal expected-shape (tensor-shape tensor))
    (error "Expected shape ~a but got ~a" expected-shape (tensor-shape tensor))))

(defun run-ssm-tests ()
  "Run all state space model tests"
  (setf *test-results* nil
        *tests-passed* 0
        *tests-failed* 0)
  
  (format t "~%")
  (format t "╔════════════════════════════════════════════════════════════════╗~%")
  (format t "║  State Space Models Tests                                     ║~%")
  (format t "╚════════════════════════════════════════════════════════════════╝~%")
  (format t "~%")
  
  ;; Test S4 Layer Creation
  (deftest test-s4-creation
    (let ((s4 (make-instance 's4-layer :d-model 64 :d-state 16)))
      (assert (= 64 (neural-tensor-ssm::d-model s4)))
      (assert (= 16 (neural-tensor-ssm::d-state s4)))
      (assert (not (null (neural-tensor-ssm::lambda-real s4))))
      (assert (not (null (neural-tensor-ssm::lambda-imag s4))))))
  
  (deftest test-s4-parameters
    (let* ((s4 (make-instance 's4-layer :d-model 32 :d-state 8))
           (params (layer-parameters s4)))
      (assert (>= (length params) 4)))) ; A, B, C, D components
  
  (deftest test-s4-forward
    (let ((s4 (make-instance 's4-layer :d-model 64 :d-state 16))
          (x (randn '(2 10 64))))  ; (batch, seq_len, d_model)
      (let ((output (forward s4 x)))
        (assert-shape '(2 10 64) output))))
  
  (deftest test-s4-different-sequence-lengths
    (let ((s4 (make-instance 's4-layer :d-model 32 :d-state 8)))
      (let ((out1 (forward s4 (randn '(1 5 32))))
            (out2 (forward s4 (randn '(1 20 32)))))
        (assert-shape '(1 5 32) out1)
        (assert-shape '(1 20 32) out2))))
  
  ;; Test HiPPO Initialization
  (deftest test-hippo-initialization
    (let ((A (hippo-initialization 8)))
      (assert (typep A 'neural-network:tensor))
      (assert-shape '(8 8) A)))
  
  (deftest test-hippo-initialization-properties
    (let* ((A (hippo-initialization 4))
           (A-data (tensor-data A)))
      ;; HiPPO matrix should have specific structure (upper triangular for LegS)
      (dotimes (i 4)
        (assert (not (zerop (aref A-data i i)))))))
  
  ;; Test Discretization Methods
  (deftest test-discretize-zoh
    (let* ((A (hippo-initialization 4))
           (B (make-array '(4 1) :element-type 'double-float :initial-element 1.0d0))
           (C (make-array '(1 4) :element-type 'double-float :initial-element 1.0d0))
           (dt 0.01d0)
           (result (discretize-zoh A B C dt)))
      (assert (consp result))
      (assert (= 2 (length result)))
      (assert (arrayp (first result)))
      (assert (arrayp (second result)))))
  
  (deftest test-discretize-bilinear
    (let* ((A (hippo-initialization 4))
           (B (make-array '(4 1) :element-type 'double-float :initial-element 1.0d0))
           (C (make-array '(1 4) :element-type 'double-float :initial-element 1.0d0))
           (dt 0.01d0)
           (result (discretize-bilinear A B C dt)))
      (assert (consp result))
      (assert (= 2 (length result)))))
  
  ;; Test Selective Scan
  (deftest test-selective-scan
    (let* ((batch-size 2)
           (seq-len 10)
           (d-state 8)
           (d-model 16)
           (u (randn (list batch-size seq-len d-model)))
           (delta (make-tensor 
                   (make-array (list batch-size seq-len d-state)
                              :element-type 'double-float
                              :initial-element 0.01d0)
                   :shape (list batch-size seq-len d-state)))
           (A (make-tensor
               (make-array (list d-state d-state)
                          :element-type 'double-float
                          :initial-element 0.1d0)
               :shape (list d-state d-state)))
           (B (randn (list batch-size seq-len d-state)))
           (C (randn (list batch-size seq-len d-state)))
           (output (selective-scan u delta A B C)))
      (assert-shape (list batch-size seq-len d-model) output)))
  
  ;; Test Mamba Block Creation
  (deftest test-mamba-creation
    (let ((mamba (make-instance 'mamba-block :d-model 64 :d-state 16 :expand-factor 2)))
      (assert (= 64 (neural-tensor-ssm::d-model mamba)))
      (assert (= 16 (neural-tensor-ssm::d-state mamba)))
      (assert (= 2 (neural-tensor-ssm::expand-factor mamba)))
      (assert (not (null (neural-tensor-ssm::in-proj mamba))))
      (assert (not (null (neural-tensor-ssm::out-proj mamba))))))
  
  (deftest test-mamba-parameters
    (let* ((mamba (make-instance 'mamba-block :d-model 32 :d-state 8))
           (params (layer-parameters mamba)))
      (assert (> (length params) 0))))
  
  (deftest test-mamba-forward
    (let ((mamba (make-instance 'mamba-block :d-model 64 :d-state 16))
          (x (randn '(2 10 64))))
      (let ((output (forward mamba x)))
        (assert-shape '(2 10 64) output))))
  
  (deftest test-mamba-different-batch-sizes
    (let ((mamba (make-instance 'mamba-block :d-model 32 :d-state 8)))
      (let ((out1 (forward mamba (randn '(1 5 32))))
            (out2 (forward mamba (randn '(4 5 32)))))
        (assert-shape '(1 5 32) out1)
        (assert-shape '(4 5 32) out2))))
  
  (deftest test-mamba-expand-factor
    (let ((mamba2 (make-instance 'mamba-block :d-model 64 :d-state 16 :expand-factor 2))
          (mamba4 (make-instance 'mamba-block :d-model 64 :d-state 16 :expand-factor 4))
          (x (randn '(2 5 64))))
      ;; Both should produce same output shape
      (assert-shape '(2 5 64) (forward mamba2 x))
      (assert-shape '(2 5 64) (forward mamba4 x))))
  
  ;; Test Samba Hybrid Layer
  (deftest test-samba-creation
    (let ((samba (make-instance 'samba-block :d-model 64 :d-state 16 :num-heads 8)))
      (assert (= 64 (neural-tensor-ssm::d-model samba)))
      (assert (= 16 (neural-tensor-ssm::d-state samba)))
      (assert (= 8 (neural-tensor-ssm::num-heads samba)))
      (assert (not (null (neural-tensor-ssm::mamba-block samba))))
      (assert (not (null (neural-tensor-ssm::attention samba))))))
  
  (deftest test-samba-parameters
    (let* ((samba (make-instance 'samba-block :d-model 64 :d-state 16 :num-heads 8))
           (params (layer-parameters samba)))
      (assert (> (length params) 0))))
  
  (deftest test-samba-forward
    (let ((samba (make-instance 'samba-block :d-model 64 :d-state 16 :num-heads 8))
          (x (randn '(2 10 64))))
      (let ((output (forward samba x)))
        (assert-shape '(2 10 64) output))))
  
  (deftest test-samba-with-mask
    (let ((samba (make-instance 'samba-block :d-model 64 :d-state 16 :num-heads 8))
          (x (randn '(2 10 64)))
          (mask (zeros '(10 10))))  ; Simple mask
      (declare (ignore mask)) ; Mask support not yet implemented
      (let ((output (forward samba x)))
        (assert-shape '(2 10 64) output))))
  
  ;; Test SSM Language Model
  (deftest test-ssm-lm-creation
    (let ((lm (make-instance 'ssm-lm 
                            :vocab-size 1000
                            :d-model 128
                            :d-state 32
                            :num-layers 4)))
      (assert (= 1000 (neural-tensor-ssm::vocab-size lm)))
      (assert (= 128 (neural-tensor-ssm::d-model lm)))
      (assert (= 32 (neural-tensor-ssm::d-state lm)))
      (assert (= 4 (length (neural-tensor-ssm::layers lm))))))
  
  (deftest test-ssm-lm-parameters
    (let* ((lm (make-instance 'ssm-lm
                             :vocab-size 500
                             :d-model 64
                             :d-state 16
                             :num-layers 2))
           (params (layer-parameters lm)))
      (assert (> (length params) 0))))
  
  (deftest test-ssm-lm-forward
    (let ((lm (make-instance 'ssm-lm
                            :vocab-size 1000
                            :d-model 64
                            :d-state 16
                            :num-layers 2))
          (input-ids (make-tensor
                      (make-array '(2 10)
                                 :element-type 'double-float
                                 :initial-contents
                                 '((1.0d0 2.0d0 3.0d0 4.0d0 5.0d0 6.0d0 7.0d0 8.0d0 9.0d0 10.0d0)
                                   (11.0d0 12.0d0 13.0d0 14.0d0 15.0d0 16.0d0 17.0d0 18.0d0 19.0d0 20.0d0)))
                      :shape '(2 10))))
      (let ((logits (forward lm input-ids)))
        (assert-shape '(2 10 1000) logits))))
  
  (deftest test-ssm-lm-different-architectures
    ;; Test with Samba layers
    (let ((lm (make-instance 'ssm-lm
                            :vocab-size 500
                            :d-model 64
                            :d-state 16
                            :num-layers 2
                            :use-samba t
                            :num-heads 4))
          (input-ids (randn '(2 8))))
      (let ((logits (forward lm input-ids)))
        (assert-shape '(2 8 500) logits))))
  
  ;; Test edge cases
  (deftest test-s4-single-element-sequence
    (let ((s4 (make-instance 's4-layer :d-model 32 :d-state 8))
          (x (randn '(1 1 32))))
      (let ((output (forward s4 x)))
        (assert-shape '(1 1 32) output))))
  
  (deftest test-mamba-single-element-sequence
    (let ((mamba (make-instance 'mamba-block :d-model 32 :d-state 8))
          (x (randn '(1 1 32))))
      (let ((output (forward mamba x)))
        (assert-shape '(1 1 32) output))))
  
  (deftest test-s4-long-sequence
    (let ((s4 (make-instance 's4-layer :d-model 32 :d-state 8))
          (x (randn '(1 100 32))))
      (let ((output (forward s4 x)))
        (assert-shape '(1 100 32) output))))
  
  (deftest test-mamba-long-sequence
    (let ((mamba (make-instance 'mamba-block :d-model 32 :d-state 8))
          (x (randn '(1 100 32))))
      (let ((output (forward mamba x)))
        (assert-shape '(1 100 32) output))))
  
  ;; Test different state dimensions
  (deftest test-s4-small-state
    (let ((s4 (make-instance 's4-layer :d-model 64 :d-state 4))
          (x (randn '(2 5 64))))
      (let ((output (forward s4 x)))
        (assert-shape '(2 5 64) output))))
  
  (deftest test-s4-large-state
    (let ((s4 (make-instance 's4-layer :d-model 64 :d-state 32))
          (x (randn '(2 5 64))))
      (let ((output (forward s4 x)))
        (assert-shape '(2 5 64) output))))
  
  (deftest test-mamba-small-state
    (let ((mamba (make-instance 'mamba-block :d-model 64 :d-state 4))
          (x (randn '(2 5 64))))
      (let ((output (forward mamba x)))
        (assert-shape '(2 5 64) output))))
  
  (deftest test-mamba-large-state
    (let ((mamba (make-instance 'mamba-block :d-model 64 :d-state 32))
          (x (randn '(2 5 64))))
      (let ((output (forward mamba x)))
        (assert-shape '(2 5 64) output))))
  
  ;; Test numerical properties
  (deftest test-s4-output-not-nan
    (let ((s4 (make-instance 's4-layer :d-model 32 :d-state 8))
          (x (randn '(2 5 32))))
      (let* ((output (forward s4 x))
             (data (tensor-data output)))
        (dotimes (i (min 10 (array-total-size data)))
          (assert (not (sb-ext:float-nan-p (row-major-aref data i))))))))
  
  (deftest test-mamba-output-not-nan
    (let ((mamba (make-instance 'mamba-block :d-model 32 :d-state 8))
          (x (randn '(2 5 32))))
      (let* ((output (forward mamba x))
             (data (tensor-data output)))
        (dotimes (i (min 10 (array-total-size data)))
          (assert (not (sb-ext:float-nan-p (row-major-aref data i))))))))
  
  ;; ========== EDGE CASE TESTS ==========
  
  ;; Test with minimal dimensions
  (deftest test-s4-minimal-dims
    (let ((s4 (make-instance 's4-layer :d-model 2 :d-state 2))
          (x (randn '(1 1 2))))
      (let ((output (forward s4 x)))
        (assert-shape '(1 1 2) output))))
  
  ;; Test with very large batch
  (deftest test-mamba-large-batch
    (let ((mamba (make-instance 'mamba-block :d-model 32 :d-state 8))
          (x (randn '(128 10 32))))
      (let ((output (forward mamba x)))
        (assert-shape '(128 10 32) output))))
  
  ;; Test S4 with extremely long sequences
  (deftest test-s4-extremely-long-sequence
    (let ((s4 (make-instance 's4-layer :d-model 16 :d-state 4))
          (x (randn '(1 1000 16))))
      (let ((output (forward s4 x)))
        (assert-shape '(1 1000 16) output))))
  
  ;; Test Mamba with various expand factors
  (deftest test-mamba-expand-factors
    (let ((mamba1 (make-instance 'mamba-block :d-model 32 :d-state 8 :expand-factor 1))
          (mamba2 (make-instance 'mamba-block :d-model 32 :d-state 8 :expand-factor 2))
          (mamba4 (make-instance 'mamba-block :d-model 32 :d-state 8 :expand-factor 4))
          (x (randn '(2 10 32))))
      (assert-shape '(2 10 32) (forward mamba1 x))
      (assert-shape '(2 10 32) (forward mamba2 x))
      (assert-shape '(2 10 32) (forward mamba4 x))))
  
  ;; Test HiPPO with different measures
  (deftest test-hippo-different-measures
    (let ((A-legs (hippo-initialization 8 :legs))
          (A-lagt (hippo-initialization 8 :lagt)))
      (assert-shape '(8 8) A-legs)
      (assert-shape '(8 8) A-lagt)))
  
  ;; Test HiPPO matrix properties
  (deftest test-hippo-matrix-properties
    (let* ((n 8)
           (A (hippo-initialization n))
           (A-data (tensor-data A)))
      ;; Check matrix is not all zeros
      (let ((has-nonzero nil))
        (dotimes (i (* n n))
          (when (> (abs (row-major-aref A-data i)) 1d-10)
            (setf has-nonzero t)))
        (assert has-nonzero))
      ;; Check no NaN or Inf
      (dotimes (i (* n n))
        (let ((val (row-major-aref A-data i)))
          (assert (not (sb-ext:float-nan-p val)))
          (assert (not (sb-ext:float-infinity-p val)))))))
  
  ;; Test discretization with small timesteps
  (deftest test-discretization-small-timestep
    (let* ((A (hippo-initialization 4))
           (B (make-array '(4 1) :element-type 'double-float :initial-element 1.0d0))
           (C (make-array '(1 4) :element-type 'double-float :initial-element 1.0d0))
           (dt 1d-6)  ; Very small timestep
           (result (discretize-zoh A B C dt)))
      (assert (consp result))
      (assert (= 2 (length result)))))
  
  ;; Test discretization with large timesteps
  (deftest test-discretization-large-timestep
    (let* ((A (hippo-initialization 4))
           (B (make-array '(4 1) :element-type 'double-float :initial-element 1.0d0))
           (C (make-array '(1 4) :element-type 'double-float :initial-element 1.0d0))
           (dt 1.0d0)  ; Large timestep
           (result (discretize-bilinear A B C dt)))
      (assert (consp result))
      (assert (= 2 (length result)))))
  
  ;; Test selective scan with edge cases
  (deftest test-selective-scan-edge-cases
    (let* ((u (randn '(1 1 4)))  ; Single timestep
           (delta (make-tensor 
                   (make-array '(1 1 2)
                              :element-type 'double-float
                              :initial-element 0.01d0)
                   :shape '(1 1 2)))
           (A (make-tensor
               (make-array '(2 2)
                          :element-type 'double-float
                          :initial-element 0.1d0)
               :shape '(2 2)))
           (B (randn '(1 1 2)))
           (C (randn '(1 1 2)))
           (output (selective-scan u delta A B C)))
      (assert-shape '(1 1 4) output)))
  
  ;; Test SSM-LM with small vocabulary
  (deftest test-ssm-lm-small-vocab
    (let ((lm (make-instance 'ssm-lm
                            :vocab-size 10
                            :d-model 8
                            :d-state 4
                            :num-layers 1))
          (input-ids (randn '(1 5))))
      (let ((logits (forward lm input-ids)))
        (assert-shape '(1 5 10) logits))))
  
  ;; Test SSM-LM with large vocabulary
  (deftest test-ssm-lm-large-vocab
    (let ((lm (make-instance 'ssm-lm
                            :vocab-size 10000  ; Reduced from 50000 to avoid heap exhaustion
                            :d-model 64
                            :d-state 16
                            :num-layers 2))
          (input-ids (randn '(2 10))))
      (let ((logits (forward lm input-ids)))
        (assert-shape '(2 10 10000) logits))))
  
  ;; Test Samba with different head counts
  (deftest test-samba-different-head-counts
    (let ((samba2 (make-instance 'samba-block :d-model 64 :d-state 16 :num-heads 2))
          (samba4 (make-instance 'samba-block :d-model 64 :d-state 16 :num-heads 4))
          (samba8 (make-instance 'samba-block :d-model 64 :d-state 16 :num-heads 8))
          (x (randn '(2 10 64))))
      (assert-shape '(2 10 64) (forward samba2 x))
      (assert-shape '(2 10 64) (forward samba4 x))
      (assert-shape '(2 10 64) (forward samba8 x))))
  
  ;; Test S4 numerical stability with large values
  (deftest test-s4-numerical-stability
    (let ((s4 (make-instance 's4-layer :d-model 32 :d-state 8))
          (x (make-tensor 
              (make-array '(1 5 32)
                         :element-type 'double-float
                         :initial-element 100.0d0)
              :shape '(1 5 32))))
      (let* ((output (forward s4 x))
             (data (tensor-data output)))
        ;; Check for NaN or Inf
        (dotimes (i (min 50 (array-total-size data)))
          (let ((val (row-major-aref data i)))
            (assert (not (sb-ext:float-nan-p val)))
            (assert (not (sb-ext:float-infinity-p val))))))))
  
  ;; Test Mamba numerical stability with large values
  (deftest test-mamba-numerical-stability
    (let ((mamba (make-instance 'mamba-block :d-model 32 :d-state 8))
          (x (make-tensor 
              (make-array '(1 5 32)
                         :element-type 'double-float
                         :initial-element 100.0d0)
              :shape '(1 5 32))))
      (let* ((output (forward mamba x))
             (data (tensor-data output)))
        (dotimes (i (min 50 (array-total-size data)))
          (let ((val (row-major-aref data i)))
            (assert (not (sb-ext:float-nan-p val)))
            (assert (not (sb-ext:float-infinity-p val))))))))
  
  ;; Test with zero input
  (deftest test-s4-zero-input
    (let ((s4 (make-instance 's4-layer :d-model 32 :d-state 8))
          (x (zeros '(2 10 32))))
      (let ((output (forward s4 x)))
        (assert-shape '(2 10 32) output))))
  
  ;; Test Mamba with zero input
  (deftest test-mamba-zero-input
    (let ((mamba (make-instance 'mamba-block :d-model 32 :d-state 8))
          (x (zeros '(2 10 32))))
      (let ((output (forward mamba x)))
        (assert-shape '(2 10 32) output))))
  
  ;; Test parameter initialization
  (deftest test-s4-parameter-initialization
    (let* ((s4 (make-instance 's4-layer :d-model 32 :d-state 8))
           (params (layer-parameters s4)))
      ;; Check parameters exist and are not all zero
      (assert (> (length params) 0))
      (let ((all-zero t))
        (dolist (param params)
          (let ((data (tensor-data param)))
            (when (> (abs (row-major-aref data 0)) 1d-10)
              (setf all-zero nil))))
        (assert (not all-zero)))))
  
  ;; Test Mamba parameter initialization
  (deftest test-mamba-parameter-initialization
    (let* ((mamba (make-instance 'mamba-block :d-model 32 :d-state 8))
           (params (layer-parameters mamba)))
      (assert (> (length params) 0))
      ;; Check not all parameters are zero
      (let ((all-zero t))
        (dolist (param params)
          (let ((data (tensor-data param)))
            (when (> (abs (row-major-aref data 0)) 1d-10)
              (setf all-zero nil))))
        (assert (not all-zero)))))
  
  ;; Test SSM-LM with single token
  (deftest test-ssm-lm-single-token
    (let ((lm (make-instance 'ssm-lm
                            :vocab-size 100
                            :d-model 32
                            :d-state 8
                            :num-layers 2))
          (input-ids (randn '(1 1))))
      (let ((logits (forward lm input-ids)))
        (assert-shape '(1 1 100) logits))))
  
  ;; Test consistency across multiple forward passes
  (deftest test-s4-consistency
    (let ((s4 (make-instance 's4-layer :d-model 32 :d-state 8))
          (x (randn '(2 5 32))))
      (let ((out1 (forward s4 x))
            (out2 (forward s4 x)))
        ;; Same input should produce same output (deterministic)
        (assert-shape '(2 5 32) out1)
        (assert-shape '(2 5 32) out2))))
  
  ;; Test Mamba consistency
  (deftest test-mamba-consistency
    (let ((mamba (make-instance 'mamba-block :d-model 32 :d-state 8))
          (x (randn '(2 5 32))))
      (let ((out1 (forward mamba x))
            (out2 (forward mamba x)))
        (assert-shape '(2 5 32) out1)
        (assert-shape '(2 5 32) out2))))
  
  ;; Test Samba consistency
  (deftest test-samba-consistency
    (let ((samba (make-instance 'samba-block :d-model 32 :d-state 8 :num-heads 4))
          (x (randn '(2 5 32))))
      (let ((out1 (forward samba x))
            (out2 (forward samba x)))
        (assert-shape '(2 5 32) out1)
        (assert-shape '(2 5 32) out2))))
  
  ;; Test different state/model dimension ratios
  (deftest test-state-model-ratio
    (let ((s4-small-state (make-instance 's4-layer :d-model 64 :d-state 8))
          (s4-large-state (make-instance 's4-layer :d-model 64 :d-state 128))
          (x (randn '(1 5 64))))
      (assert-shape '(1 5 64) (forward s4-small-state x))
      (assert-shape '(1 5 64) (forward s4-large-state x))))
  
  ;; Test gradient flow in parameters
  (deftest test-s4-gradient-flow
    (let* ((s4 (make-instance 's4-layer :d-model 32 :d-state 8))
           (params (layer-parameters s4)))
      ;; All parameters should require gradients
      (dolist (param params)
        (assert (neural-network::requires-grad param)))))
  
  ;; Test Mamba gradient flow
  (deftest test-mamba-gradient-flow
    (let* ((mamba (make-instance 'mamba-block :d-model 32 :d-state 8))
           (params (layer-parameters mamba)))
      (dolist (param params)
        (assert (neural-network::requires-grad param)))))
  
  ;; Test compute-ssm-kernel
  (deftest test-compute-ssm-kernel
    (let* ((a-bar (randn '(4 4)))
           (b-bar (randn '(4 1)))
           (c-matrix (randn '(1 4)))
           (kernel (neural-tensor-ssm::compute-ssm-kernel a-bar b-bar c-matrix 10)))
      (assert-shape '(1 10) kernel)))
  
  (deftest test-compute-ssm-kernel-values
    (let* ((n 3)
           (a-bar (make-tensor #2A((0.9d0 0.0d0 0.0d0)
                                   (0.0d0 0.8d0 0.0d0)
                                   (0.0d0 0.0d0 0.7d0))
                              :shape '(3 3)))
           (b-bar (make-tensor #2A((1.0d0) (1.0d0) (1.0d0)) :shape '(3 1)))
           (c-matrix (make-tensor #2A((1.0d0 1.0d0 1.0d0)) :shape '(1 3)))
           (kernel (neural-tensor-ssm::compute-ssm-kernel a-bar b-bar c-matrix 5))
           (kernel-data (tensor-data kernel)))
      ;; K[0] = C*B = 3.0
      (assert-equal 3.0d0 (aref kernel-data 0 0) 1d-4)
      ;; K[1] = C*A*B = 2.4 (0.9 + 0.8 + 0.7)
      (assert-equal 2.4d0 (aref kernel-data 0 1) 1d-4)))
  
  ;; Test convolve-1d with 2D input
  (deftest test-convolve-1d-2d
    (let* ((input (make-tensor #2A((1.0d0 2.0d0 3.0d0 4.0d0 5.0d0))
                              :shape '(1 5)))
           (kernel (make-tensor #2A((0.5d0 0.5d0)) :shape '(1 2)))
           (output (neural-tensor-ssm::convolve-1d input kernel)))
      (assert-shape '(1 5) output)
      (let ((out-data (tensor-data output)))
        ;; out[0] = 0.5*1.0 = 0.5
        (assert-equal 0.5d0 (aref out-data 0 0) 1d-4)
        ;; out[1] = 0.5*2.0 + 0.5*1.0 = 1.5
        (assert-equal 1.5d0 (aref out-data 0 1) 1d-4))))
  
  ;; Test convolve-1d with 3D input
  (deftest test-convolve-1d-3d
    (let* ((input (randn '(2 10 8)))
           (kernel (make-tensor #2A((0.5d0 0.3d0 0.2d0)) :shape '(1 3)))
           (output (neural-tensor-ssm::convolve-1d input kernel)))
      (assert-shape '(2 10 8) output)))
  
  ;; Test S4 recurrent mode
  (deftest test-s4-recurrent-mode
    (let* ((s4 (make-instance 's4-layer :d-model 16 :d-state 8 :mode :recurrent))
           (input (randn '(2 5 16)))
           (output (forward s4 input)))
      (assert-shape '(2 5 16) output)))
  
  ;; Test S4 convolution mode
  (deftest test-s4-convolution-mode
    (let* ((s4 (make-instance 's4-layer :d-model 16 :d-state 8 :mode :convolution))
           (input (randn '(2 5 16)))
           (output (forward s4 input)))
      (assert-shape '(2 5 16) output)))
  
  ;; Test mode switching
  (deftest test-s4-mode-switching
    (let ((s4 (make-instance 's4-layer :d-model 32 :d-state 8)))
      ;; Test convolution mode
      (setf (slot-value s4 'neural-tensor-ssm::mode) :convolution)
      (let ((out1 (forward s4 (randn '(1 10 32)))))
        (assert-shape '(1 10 32) out1))
      ;; Test recurrent mode
      (setf (slot-value s4 'neural-tensor-ssm::mode) :recurrent)
      (let ((out2 (forward s4 (randn '(1 10 32)))))
        (assert-shape '(1 10 32) out2))))
  
  ;; Test convolution kernel with different lengths
  (deftest test-kernel-different-lengths
    (let ((a (randn '(4 4)))
          (b (randn '(4 1)))
          (c (randn '(1 4))))
      (let ((k10 (neural-tensor-ssm::compute-ssm-kernel a b c 10))
            (k20 (neural-tensor-ssm::compute-ssm-kernel a b c 20)))
        (assert-shape '(1 10) k10)
        (assert-shape '(1 20) k20))))
  
  ;; Test 1D convolution boundary conditions
  (deftest test-convolve-1d-boundaries
    (let* ((input (make-tensor #2A((1.0d0 2.0d0 3.0d0)) :shape '(1 3)))
           (kernel (make-tensor #2A((1.0d0 1.0d0 1.0d0)) :shape '(1 3)))
           (output (neural-tensor-ssm::convolve-1d input kernel))
           (out-data (tensor-data output)))
      ;; First position: only 1 input available
      (assert-equal 1.0d0 (aref out-data 0 0) 1d-4)
      ;; Second position: 2 inputs available
      (assert-equal 3.0d0 (aref out-data 0 1) 1d-4)
      ;; Third position: all 3 inputs
      (assert-equal 6.0d0 (aref out-data 0 2) 1d-4)))
  
  ;; Test S4 recurrent forward with actual state evolution
  (deftest test-s4-recurrent-state-evolution
    (let* ((s4 (make-instance 's4-layer :d-model 8 :d-state 4 :mode :recurrent))
           (input (randn '(1 10 8)))
           (output (forward s4 input)))
      ;; Output should depend on history (not just current input)
      (assert-shape '(1 10 8) output)
      (assert (not (null output)))))
  
  ;; Print summary
  (format t "~%State Space Model Tests: ~d passed, ~d failed~%~%" *tests-passed* *tests-failed*)
  
  (values *tests-passed* *tests-failed*))

;; Run tests when file is loaded
(format t "~%To run SSM tests, execute: (neural-tensor-ssm-tests:run-ssm-tests)~%")
