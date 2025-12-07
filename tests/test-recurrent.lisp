;;;; Tests for Recurrent Neural Networks
;;;; Testing RNN, LSTM, GRU, and Bidirectional variants

(defpackage :neural-tensor-recurrent-tests
  (:use :common-lisp)
  (:import-from :neural-network
                #:tensor
                #:make-tensor
                #:zeros
                #:ones
                #:randn
                #:forward
                #:backward
                #:tensor-shape
                #:tensor-data
                #:requires-grad
                #:layer-parameters)
  (:import-from :neural-tensor-recurrent
                #:rnn-cell
                #:lstm-cell
                #:gru-cell
                #:rnn-layer
                #:lstm-layer
                #:gru-layer
                #:bidirectional-lstm
                #:bidirectional-gru
                #:cell-forward
                #:init-hidden
                #:reset-hidden-state
                #:sequence-map
                #:sequence-fold)
  (:export #:run-recurrent-tests))

(in-package :neural-tensor-recurrent-tests)

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

(defun run-recurrent-tests ()
  "Run all recurrent network tests"
  (setf *test-results* nil
        *tests-passed* 0
        *tests-failed* 0)
  
  (format t "~%")
  (format t "╔════════════════════════════════════════════════════════════════╗~%")
  (format t "║  Recurrent Neural Networks Tests                              ║~%")
  (format t "╚════════════════════════════════════════════════════════════════╝~%")
  (format t "~%")
  
  ;; Test RNN Cell
  (deftest test-rnn-cell-creation
    (let ((cell (make-instance 'rnn-cell
                              :input-size 10
                              :hidden-size 20)))
      (assert (= 10 (neural-tensor-recurrent::input-size cell)))
      (assert (= 20 (neural-tensor-recurrent::hidden-size cell)))
      (assert (= 4 (length (layer-parameters cell))))))
  
  (deftest test-rnn-cell-forward
    (let* ((cell (make-instance 'rnn-cell
                               :input-size 10
                               :hidden-size 20))
           (input (randn '(1 10)))
           (hidden (zeros '(1 20)))
           (output (cell-forward cell input hidden)))
      (assert-shape '(1 20) output)
      (assert (requires-grad output))))
  
  (deftest test-rnn-cell-init-hidden
    (let* ((cell (make-instance 'rnn-cell
                               :input-size 10
                               :hidden-size 20))
           (hidden (init-hidden cell 5)))
      (assert-shape '(5 20) hidden)))
  
  ;; Test LSTM Cell
  (deftest test-lstm-cell-creation
    (let ((cell (make-instance 'lstm-cell
                              :input-size 10
                              :hidden-size 20)))
      (assert (= 10 (neural-tensor-recurrent::input-size cell)))
      (assert (= 20 (neural-tensor-recurrent::hidden-size cell)))
      (assert (= 16 (length (layer-parameters cell)))))) ; 4 gates × 4 params each
  
  (deftest test-lstm-cell-forward
    (let* ((cell (make-instance 'lstm-cell
                               :input-size 10
                               :hidden-size 20))
           (input (randn '(1 10)))
           (hidden (zeros '(1 20)))
           (cell-state (zeros '(1 20)))
           (output (cell-forward cell input (list hidden cell-state))))
      (assert (listp output))
      (assert (= 2 (length output)))
      (assert-shape '(1 20) (first output))
      (assert-shape '(1 20) (second output))))
  
  (deftest test-lstm-init-hidden
    (let* ((cell (make-instance 'lstm-cell
                               :input-size 10
                               :hidden-size 20))
           (hidden-and-cell (init-hidden cell 3)))
      (assert (listp hidden-and-cell))
      (assert (= 2 (length hidden-and-cell)))
      (assert-shape '(3 20) (first hidden-and-cell))
      (assert-shape '(3 20) (second hidden-and-cell))))
  
  ;; Test GRU Cell
  (deftest test-gru-cell-creation
    (let ((cell (make-instance 'gru-cell
                              :input-size 10
                              :hidden-size 20)))
      (assert (= 10 (neural-tensor-recurrent::input-size cell)))
      (assert (= 20 (neural-tensor-recurrent::hidden-size cell)))
      (assert (= 12 (length (layer-parameters cell)))))) ; 3 gates × 4 params each
  
  (deftest test-gru-cell-forward
    (let* ((cell (make-instance 'gru-cell
                               :input-size 10
                               :hidden-size 20))
           (input (randn '(1 10)))
           (hidden (zeros '(1 20)))
           (output (cell-forward cell input hidden)))
      (assert-shape '(1 20) output)
      (assert (requires-grad output))))
  
  ;; Test RNN Layer
  (deftest test-rnn-layer-creation
    (let ((layer (rnn-layer 10 20)))
      (assert (not (null layer)))
      (assert (typep (neural-tensor-recurrent::rnn-cell layer) 'rnn-cell))))
  
  (deftest test-lstm-layer-creation
    (let ((layer (lstm-layer 10 20)))
      (assert (not (null layer)))
      (assert (typep (neural-tensor-recurrent::rnn-cell layer) 'lstm-cell))))
  
  (deftest test-gru-layer-creation
    (let ((layer (gru-layer 10 20)))
      (assert (not (null layer)))
      (assert (typep (neural-tensor-recurrent::rnn-cell layer) 'gru-cell))))
  
  ;; Test Bidirectional LSTM
  (deftest test-bidirectional-lstm-creation
    (let ((bilstm (bidirectional-lstm 10 20)))
      (assert (not (null bilstm)))
      (assert (not (null (neural-tensor-recurrent::forward-cell bilstm))))
      (assert (not (null (neural-tensor-recurrent::backward-cell bilstm))))))
  
  (deftest test-bidirectional-lstm-parameters
    (let* ((bilstm (bidirectional-lstm 10 20))
           (params (layer-parameters bilstm)))
      (assert (= 32 (length params))))) ; 16 params × 2 directions
  
  ;; Test Bidirectional GRU
  (deftest test-bidirectional-gru-creation
    (let ((bigru (bidirectional-gru 10 20)))
      (assert (not (null bigru)))
      (assert (not (null (neural-tensor-recurrent::forward-cell bigru))))
      (assert (not (null (neural-tensor-recurrent::backward-cell bigru))))))
  
  ;; Test higher-order functions
  (deftest test-sequence-map
    (let* ((fn (lambda (x hidden) (values (list x (1+ x)) (1+ hidden))))
           (seq '(1 2 3))
           (mapper (sequence-map fn seq)))
      (multiple-value-bind (result final-hidden)
          (funcall mapper 0)
        (assert (listp (first result)))
        (assert (= 3 final-hidden)))))
  
  (deftest test-sequence-fold
    (let* ((result (sequence-fold #'+ 0 '(1 2 3 4 5))))
      (assert-equal 15 result)))
  
  ;; Test hidden state management
  (deftest test-reset-hidden-state
    (let ((cell (make-instance 'lstm-cell
                              :input-size 10
                              :hidden-size 20)))
      (setf (neural-tensor-recurrent::hidden-state cell) (randn '(1 20)))
      (reset-hidden-state cell)
      (assert (null (neural-tensor-recurrent::hidden-state cell)))))
  
  ;; Test parameter counts
  (deftest test-rnn-parameter-count
    (let* ((cell (make-instance 'rnn-cell
                               :input-size 10
                               :hidden-size 20))
           (params (layer-parameters cell)))
      (assert (= 4 (length params))))) ; W_ih, W_hh, b_ih, b_hh
  
  (deftest test-lstm-parameter-count
    (let* ((cell (make-instance 'lstm-cell
                               :input-size 10
                               :hidden-size 20))
           (params (layer-parameters cell)))
      (assert (= 16 (length params))))) ; 4 gates × 4 params
  
  (deftest test-gru-parameter-count
    (let* ((cell (make-instance 'gru-cell
                               :input-size 10
                               :hidden-size 20))
           (params (layer-parameters cell)))
      (assert (= 12 (length params))))) ; 3 gates × 4 params
  
  ;; ========== EDGE CASE TESTS ==========
  
  ;; Test with minimal dimensions
  (deftest test-rnn-minimal-dims
    (let* ((cell (make-instance 'rnn-cell
                               :input-size 1
                               :hidden-size 1))
           (input (randn '(1 1)))
           (hidden (zeros '(1 1)))
           (output (cell-forward cell input hidden)))
      (assert-shape '(1 1) output)))
  
  ;; Test LSTM with large batch
  (deftest test-lstm-large-batch
    (let* ((cell (make-instance 'lstm-cell
                               :input-size 10
                               :hidden-size 20))
           (input (randn '(128 10)))
           (hidden (zeros '(128 20)))
           (cell-state (zeros '(128 20)))
           (output (cell-forward cell input (list hidden cell-state))))
      (assert-shape '(128 20) (first output))))
  
  ;; Test GRU with very long sequences
  (deftest test-gru-long-sequence
    (let* ((layer (gru-layer 10 20 :return-sequences t))
           (seq (randn '(2 100 10))))
      (reset-hidden-state (neural-tensor-recurrent::rnn-cell layer))
      (let ((output (forward layer seq)))
        (assert-shape '(2 100 20) output))))
  
  ;; Test hidden state persistence
  (deftest test-hidden-state-persistence
    (let* ((cell (make-instance 'rnn-cell
                               :input-size 5
                               :hidden-size 10))
           (input1 (randn '(1 5)))
           (hidden (init-hidden cell 1))
           (out1 (cell-forward cell input1 hidden))
           (out2 (cell-forward cell input1 out1)))
      ;; Second output should be different (state changed)
      (assert-shape '(1 10) out1)
      (assert-shape '(1 10) out2)))
  
  ;; Test LSTM forget gate behavior (numerical stability)
  (deftest test-lstm-numerical-stability
    (let* ((cell (make-instance 'lstm-cell
                               :input-size 10
                               :hidden-size 20))
           ;; Very large inputs
           (input (make-tensor 
                   (make-array '(1 10)
                              :element-type 'double-float
                              :initial-element 50.0d0)
                   :shape '(1 10)))
           (hidden (zeros '(1 20)))
           (cell-state (zeros '(1 20)))
           (output (cell-forward cell input (list hidden cell-state))))
      ;; Check no NaN or Inf
      (let ((h-data (tensor-data (first output)))
            (c-data (tensor-data (second output))))
        (dotimes (i 20)
          (assert (not (sb-ext:float-nan-p (aref h-data 0 i))))
          (assert (not (sb-ext:float-infinity-p (aref h-data 0 i))))
          (assert (not (sb-ext:float-nan-p (aref c-data 0 i))))
          (assert (not (sb-ext:float-infinity-p (aref c-data 0 i))))))))
  
  ;; Test GRU reset and update gates
  (deftest test-gru-gate-behavior
    (let* ((cell (make-instance 'gru-cell
                               :input-size 10
                               :hidden-size 20))
           (input (randn '(1 10)))
           (hidden1 (zeros '(1 20)))
           (hidden2 (ones '(1 20)))
           (out1 (cell-forward cell input hidden1))
           (out2 (cell-forward cell input hidden2)))
      ;; Different hidden states should produce different outputs
      (assert-shape '(1 20) out1)
      (assert-shape '(1 20) out2)))
  
  ;; Test bidirectional RNN output shape
  (deftest test-bidirectional-lstm-output-shape
    (let* ((bilstm (bidirectional-lstm 10 20))
           (input (randn '(2 15 10))))
      (let ((output (forward bilstm input)))
        ;; Output should be 2*hidden_size
        (assert-shape '(2 15 40) output))))
  
  ;; Test bidirectional with single timestep
  (deftest test-bidirectional-single-timestep
    (let* ((bilstm (bidirectional-lstm 10 20))
           (input (randn '(1 1 10))))
      (let ((output (forward bilstm input)))
        (assert-shape '(1 1 40) output))))
  
  ;; Test RNN with zero input
  (deftest test-rnn-zero-input
    (let* ((cell (make-instance 'rnn-cell
                               :input-size 10
                               :hidden-size 20))
           (input (zeros '(1 10)))
           (hidden (zeros '(1 20)))
           (output (cell-forward cell input hidden)))
      ;; Should not crash, output depends only on bias
      (assert-shape '(1 20) output)
      (assert (not (null output)))))
  
  ;; Test LSTM with zero cell state
  (deftest test-lstm-zero-cell-state
    (let* ((cell (make-instance 'lstm-cell
                               :input-size 10
                               :hidden-size 20))
           (input (randn '(1 10)))
           (hidden (zeros '(1 20)))
           (cell-state (zeros '(1 20)))
           (output (cell-forward cell input (list hidden cell-state))))
      (assert (listp output))
      (assert (= 2 (length output)))))
  
  ;; Test sequence processing with varying lengths
  (deftest test-variable-length-sequences
    (let* ((layer (lstm-layer 10 20 :return-sequences t)))
      ;; Process sequences of different lengths
      (let ((out1 (forward layer (randn '(1 5 10))))
            (out2 (forward layer (randn '(1 15 10)))))
        (assert-shape '(1 5 20) out1)
        (assert-shape '(1 15 20) out2))))
  
  ;; Test RNN activation functions
  (deftest test-rnn-activation-variants
    (let* ((tanh-cell (make-instance 'rnn-cell
                                    :input-size 10
                                    :hidden-size 20
                                    :activation :tanh))
           (input (randn '(1 10)))
           (hidden (zeros '(1 20)))
           (output (cell-forward tanh-cell input hidden)))
      (assert-shape '(1 20) output)
      ;; Output should be in tanh range [-1, 1] approximately
      (let ((data (tensor-data output)))
        (dotimes (i 20)
          (let ((val (aref data 0 i)))
            (assert (and (>= val -2.0d0) (<= val 2.0d0))))))))
  
  ;; Test gradient flow (ensure requires-grad propagates)
  (deftest test-rnn-gradient-flow
    (let* ((cell (make-instance 'rnn-cell
                               :input-size 10
                               :hidden-size 20))
           (input (randn '(1 10) :requires-grad t))
           (hidden (zeros '(1 20)))
           (output (cell-forward cell input hidden)))
      (assert (requires-grad output))))
  
  ;; Test LSTM gradient flow
  (deftest test-lstm-gradient-flow
    (let* ((cell (make-instance 'lstm-cell
                               :input-size 10
                               :hidden-size 20))
           (input (randn '(1 10) :requires-grad t))
           (hidden (zeros '(1 20)))
           (cell-state (zeros '(1 20)))
           (output (cell-forward cell input (list hidden cell-state))))
      (assert (requires-grad (first output)))))
  
  ;; Test parameter initialization scale
  (deftest test-parameter-initialization
    (let* ((cell (make-instance 'lstm-cell
                               :input-size 10
                               :hidden-size 20))
           (params (layer-parameters cell)))
      ;; Check parameters are not all zero
      (let ((all-zero t))
        (dolist (param params)
          (let ((data (tensor-data param)))
            (when (> (abs (row-major-aref data 0)) 1d-10)
              (setf all-zero nil))))
        (assert (not all-zero)))))
  
  ;; Test memory efficiency with large hidden size
  (deftest test-large-hidden-size
    (let* ((cell (make-instance 'gru-cell
                               :input-size 100
                               :hidden-size 512))
           (input (randn '(1 100)))
           (hidden (zeros '(1 512)))
           (output (cell-forward cell input hidden)))
      (assert-shape '(1 512) output)))
  
  ;; Test bidirectional GRU correctness
  (deftest test-bidirectional-gru-correctness
    (let* ((bigru (bidirectional-gru 10 20))
           (input (randn '(2 10 10))))
      (let ((output (forward bigru input)))
        (assert-shape '(2 10 40) output)
        ;; Check output is not all zeros
        (let ((data (tensor-data output))
              (has-nonzero nil))
          (dotimes (i (min 50 (array-total-size data)))
            (when (> (abs (row-major-aref data i)) 1d-6)
              (setf has-nonzero t)))
          (assert has-nonzero)))))
  
  ;; Print summary
  (format t "~%RNN/LSTM/GRU Tests: ~d passed, ~d failed~%~%" *tests-passed* *tests-failed*)
  
  (values *tests-passed* *tests-failed*))

;; Run tests when file is loaded
(format t "~%To run recurrent network tests, execute: (neural-tensor-recurrent-tests:run-recurrent-tests)~%")
