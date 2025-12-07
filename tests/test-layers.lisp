;;;; tests/test-layers.lisp - Neural Network Layer Tests

(in-package #:neural-lisp-tests)

(def-suite layer-tests
  :description "Tests for neural network layers"
  :in neural-lisp-tests)

(in-suite layer-tests)

(test linear-layer-creation
  "Test linear layer creation"
  (let ((layer (linear 3 5)))
    (is (not (null layer)))
    (is (equal '(5 3) (tensor-shape (layer-weight layer))))
    (is (equal '(5) (tensor-shape (layer-bias layer))))))

(test linear-layer-forward
  "Test linear layer forward pass"
  (let* ((layer (linear 2 3))
         (input (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (output (forward layer input)))
    (is (equal '(1 3) (tensor-shape output)))))

(test linear-layer-parameters
  "Test linear layer parameters"
  (let* ((layer (linear 2 3))
         (params (parameters layer)))
    (is (= 2 (length params)))
    (is (member (layer-weight layer) params))
    (is (member (layer-bias layer) params))))

(test relu-layer-forward
  "Test ReLU layer forward pass"
  (let* ((layer (relu-layer))
         (input (make-tensor #(-1.0 0.0 1.0 2.0) :shape '(4)))
         (output (forward layer input)))
    (is (= 0.0 (aref (tensor-data output) 0)))
    (is (= 0.0 (aref (tensor-data output) 1)))
    (is (= 1.0 (aref (tensor-data output) 2)))
    (is (= 2.0 (aref (tensor-data output) 3)))))

(test sigmoid-layer-forward
  "Test sigmoid layer forward pass"
  (let* ((layer (sigmoid-layer))
         (input (make-tensor #(0.0) :shape '(1)))
         (output (forward layer input)))
    (is (< 0.499 (aref (tensor-data output) 0) 0.501))))

(test sequential-creation
  "Test sequential model creation"
  (let ((model (sequential
                 (linear 10 5)
                 (relu-layer)
                 (linear 5 2))))
    (is (not (null model)))
    (is (= 3 (length (seq-layers model))))))

(test sequential-forward
  "Test sequential model forward pass"
  (let* ((model (sequential
                  (linear 3 5)
                  (relu-layer)
                  (linear 5 2)))
         (input (make-tensor #2A((1.0 2.0 3.0)) :shape '(1 3)))
         (output (forward model input)))
    (is (equal '(1 2) (tensor-shape output)))))

(test sequential-parameters
  "Test sequential model parameters"
  (let* ((model (sequential
                  (linear 3 5)
                  (linear 5 2)))
         (params (parameters model)))
    (is (= 4 (length params))))) ; 2 weights + 2 biases

(test nested-sequential
  "Test nested sequential models"
  (let* ((inner (sequential
                  (linear 5 3)
                  (relu-layer)))
         (outer (sequential
                  (linear 10 5)
                  inner
                  (linear 3 2)))
         (input (make-tensor #2A((1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)) :shape '(1 10)))
         (output (forward outer input)))
    (is (equal '(1 2) (tensor-shape output)))))

(test layer-training-mode
  "Test layer training mode flag"
  (let ((layer (linear 3 5)))
    (is (layer-training layer))
    (eval-mode layer)
    (is (not (layer-training layer)))
    (train-mode layer)
    (is (layer-training layer))))

;;; Edge Cases and Robustness Tests

(test linear-layer-single-input-output
  "Test linear layer with single input and output"
  (let ((layer (linear 1 1)))
    (is (equal '(1 1) (tensor-shape (layer-weight layer))))
    (is (equal '(1) (tensor-shape (layer-bias layer))))))

(test linear-layer-large-dimensions
  "Test linear layer with large dimensions"
  (let ((layer (linear 100 50)))
    (is (equal '(50 100) (tensor-shape (layer-weight layer))))
    (is (equal '(50) (tensor-shape (layer-bias layer))))))

(test linear-layer-batch-processing
  "Test linear layer with batched input"
  (let* ((layer (linear 3 5))
         (input (make-tensor #2A((1.0 2.0 3.0)
                                 (4.0 5.0 6.0)
                                 (7.0 8.0 9.0)) :shape '(3 3)))
         (output (forward layer input)))
    (is (equal '(3 5) (tensor-shape output)))))

(test linear-layer-single-sample
  "Test linear layer with single sample (batch size 1)"
  (let* ((layer (linear 4 2))
         (input (make-tensor #2A((1.0 2.0 3.0 4.0)) :shape '(1 4)))
         (output (forward layer input)))
    (is (equal '(1 2) (tensor-shape output)))))

(test linear-layer-weight-requires-grad
  "Test that linear layer weights require gradients"
  (let ((layer (linear 2 3)))
    (is (requires-grad (layer-weight layer)))
    (is (requires-grad (layer-bias layer)))))

(test linear-layer-parameters-count
  "Test that linear layer has correct number of parameters"
  (let* ((layer (linear 10 5))
         (params (parameters layer)))
    (is (= 2 (length params)))
    ;; Weight matrix should have 50 elements, bias should have 5
    (is (= 50 (array-total-size (tensor-data (layer-weight layer)))))
    (is (= 5 (array-total-size (tensor-data (layer-bias layer)))))))

(test linear-layer-zero-input
  "Test linear layer with zero input"
  (let* ((layer (linear 3 2))
         (input (zeros '(1 3)))
         (output (forward layer input)))
    ;; Output should only be bias
    (is (equal '(1 2) (tensor-shape output)))))

(test relu-layer-preserves-shape
  "Test ReLU layer preserves input shape"
  (let* ((layer (relu-layer))
         (input (make-tensor #2A((1.0 -2.0 3.0)
                                 (-4.0 5.0 -6.0)) :shape '(2 3)))
         (output (forward layer input)))
    (is (equal '(2 3) (tensor-shape output)))))

(test relu-layer-no-parameters
  "Test ReLU layer has no parameters"
  (let* ((layer (relu-layer))
         (params (parameters layer)))
    (is (= 0 (length params)))))

(test sigmoid-layer-preserves-shape
  "Test sigmoid layer preserves input shape"
  (let* ((layer (sigmoid-layer))
         (input (randn '(5 10)))
         (output (forward layer input)))
    (is (equal '(5 10) (tensor-shape output)))))

(test sigmoid-layer-output-range
  "Test sigmoid layer outputs are in (0, 1)"
  (let* ((layer (sigmoid-layer))
         (input (make-tensor #(-10.0 -1.0 0.0 1.0 10.0) :shape '(5)))
         (output (forward layer input)))
    (dotimes (i 5)
      (is (> (aref (tensor-data output) i) 0.0))
      (is (< (aref (tensor-data output) i) 1.0)))))

(test sequential-empty-error
  "Test that empty sequential model handles properly"
  ;; Note: This might fail if implementation doesn't handle empty models
  ;; In that case, it validates the edge case
  (let ((model (sequential)))
    (is (= 0 (length (seq-layers model))))))

(test sequential-single-layer
  "Test sequential with single layer"
  (let ((model (sequential (linear 5 3))))
    (is (= 1 (length (seq-layers model))))))

(test sequential-many-layers
  "Test sequential with many layers"
  (let ((model (sequential
                 (linear 10 8)
                 (relu-layer)
                 (linear 8 6)
                 (relu-layer)
                 (linear 6 4)
                 (relu-layer)
                 (linear 4 2))))
    (is (= 7 (length (seq-layers model))))))

(test sequential-forward-propagation
  "Test forward propagation through sequential model"
  (let* ((model (sequential
                  (linear 3 5)
                  (relu-layer)
                  (linear 5 2)))
         (input (randn '(10 3)))
         (output (forward model input)))
    (is (equal '(10 2) (tensor-shape output)))))

(test sequential-parameters-aggregation
  "Test sequential model aggregates all layer parameters"
  (let* ((l1 (linear 3 5))
         (l2 (linear 5 2))
         (model (sequential l1 (relu-layer) l2))
         (params (parameters model)))
    ;; Should have 4 parameters: 2 from l1, 2 from l2, 0 from relu
    (is (= 4 (length params)))))

(test sequential-train-eval-propagation
  "Test train/eval mode propagates to all layers"
  (let ((model (sequential
                 (linear 3 5)
                 (relu-layer)
                 (linear 5 2))))
    ;; Set to eval mode
    (eval-mode model)
    (dolist (layer (seq-layers model))
      (is (not (layer-training layer))))
    ;; Set back to train mode
    (train-mode model)
    (dolist (layer (seq-layers model))
      (is (layer-training layer)))))

(test nested-sequential-parameters
  "Test nested sequential models aggregate parameters correctly"
  (let* ((inner (sequential (linear 5 3) (relu-layer)))
         (outer (sequential (linear 10 5) inner (linear 3 2)))
         (params (parameters outer)))
    ;; Should have 6 parameters: 2 from first linear, 2 from inner linear, 2 from last linear
    (is (= 6 (length params)))))

(test layer-forward-deterministic
  "Test that layer forward pass is deterministic (with same input)"
  (let* ((layer (linear 3 2))
         (input (make-tensor #2A((1.0 2.0 3.0)) :shape '(1 3)))
         (output1 (forward layer input))
         (output2 (forward layer input)))
    ;; Outputs should be identical
    (is (= (aref (tensor-data output1) 0 0)
           (aref (tensor-data output2) 0 0)))))

(test linear-layer-bias-effect
  "Test that bias affects linear layer output"
  (let* ((layer (linear 2 2))
         (input (zeros '(1 2))))
    ;; Set weights to zero
    (let ((weight-data (tensor-data (layer-weight layer))))
      (dotimes (i (array-total-size weight-data))
        (setf (row-major-aref weight-data i) 0.0d0)))
    ;; Set bias to specific value
    (setf (aref (tensor-data (layer-bias layer)) 0) 5.0d0)
    (setf (aref (tensor-data (layer-bias layer)) 1) 3.0d0)
    (let ((output (forward layer input)))
      ;; Output should equal bias when input and weights are zero
      (is (= 5.0d0 (aref (tensor-data output) 0 0)))
      (is (= 3.0d0 (aref (tensor-data output) 0 1))))))

(test relu-layer-exact-boundary
  "Test ReLU layer behavior at exact zero"
  (let* ((layer (relu-layer))
         (input (make-tensor #(0.0 0.0 0.0) :shape '(3)))
         (output (forward layer input)))
    ;; ReLU(0) should be 0
    (is (= 0.0 (aref (tensor-data output) 0)))
    (is (= 0.0 (aref (tensor-data output) 1)))
    (is (= 0.0 (aref (tensor-data output) 2)))))

(test sequential-gradient-flow
  "Test that gradients can flow through sequential model"
  (let* ((model (sequential
                  (linear 2 3)
                  (relu-layer)
                  (linear 3 1)))
         (input (make-tensor #2A((1.0 2.0)) :shape '(1 2) :requires-grad t))
         (output (forward model input))
         (loss (tsum output)))
    (backward loss)
    ;; Check that input has gradient (gradient flowed back)
    (is (not (null (tensor-grad input))))))

(test layer-state-isolation
  "Test that different layer instances have independent state"
  (let ((layer1 (linear 2 2))
        (layer2 (linear 2 2)))
    ;; Modify layer1 weights
    (setf (aref (tensor-data (layer-weight layer1)) 0 0) 99.0d0)
    ;; layer2 should be unaffected
    (is (not (= 99.0 (aref (tensor-data (layer-weight layer2)) 0 0))))))

(test sequential-with-varying-batch-sizes
  "Test sequential model with different batch sizes"
  (let ((model (sequential (linear 3 2))))
    (let* ((input1 (randn '(1 3)))
           (output1 (forward model input1)))
      (is (equal '(1 2) (tensor-shape output1))))
    (let* ((input5 (randn '(5 3)))
           (output5 (forward model input5)))
      (is (equal '(5 2) (tensor-shape output5))))
    (let* ((input10 (randn '(10 3)))
           (output10 (forward model input10)))
      (is (equal '(10 2) (tensor-shape output10))))))
