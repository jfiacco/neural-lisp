;;;; Tests for Normalization Layers - Layer Norm and Batch Norm

(in-package :neural-lisp-tests)

;;;; ============================================================================
;;;; Layer Normalization Tests
;;;; ============================================================================

(defun test-layer-norm-basic ()
  "Test basic layer normalization forward pass"
  (format t "~%Testing Layer Norm - Basic Forward Pass...")
  
  ;; Create a simple 2D input: (2, 4) - 2 samples, 4 features
  (let* ((input-data (make-array '(2 4) 
                                 :element-type 'double-float
                                 :initial-contents '((1.0d0 2.0d0 3.0d0 4.0d0)
                                                    (5.0d0 6.0d0 7.0d0 8.0d0))))
         (input (make-tensor input-data :requires-grad t))
         (ln (layer-norm 4))
         (output (forward ln input)))
    
    ;; Output should have same shape as input
    (assert (equal (tensor-shape output) '(2 4))
            () "Layer norm output shape mismatch")
    
    ;; Check that each sample is normalized (mean ~0, std ~1)
    (let ((out-data (tensor-data output)))
      ;; First sample
      (let ((mean1 0.0d0))
        (dotimes (i 4)
          (incf mean1 (aref out-data 0 i)))
        (setf mean1 (/ mean1 4))
        (assert (< (abs mean1) 1d-6)
                () "Layer norm: first sample mean should be ~0"))
      
      ;; Second sample
      (let ((mean2 0.0d0))
        (dotimes (i 4)
          (incf mean2 (aref out-data 1 i)))
        (setf mean2 (/ mean2 4))
        (assert (< (abs mean2) 1d-6)
                () "Layer norm: second sample mean should be ~0")))
    
    (format t " PASSED~%")
    t))

(defun test-layer-norm-3d ()
  "Test layer normalization with 3D input (batch, seq-len, features)"
  (format t "Testing Layer Norm - 3D Input (Transformer-style)...")
  
  ;; Create 3D input: (2, 3, 4) - batch=2, seq_len=3, features=4
  (let* ((input (randn '(2 3 4) :requires-grad t))
         (ln (layer-norm 4))
         (output (forward ln input)))
    
    ;; Output should have same shape as input
    (assert (equal (tensor-shape output) '(2 3 4))
            () "Layer norm 3D output shape mismatch")
    
    ;; Check normalization for one position
    (let* ((out-data (tensor-data output))
           (mean 0.0d0))
      (dotimes (i 4)
        (incf mean (aref out-data 0 0 i)))
      (setf mean (/ mean 4))
      (assert (< (abs mean) 1d-5)
              () "Layer norm 3D: mean should be ~0"))
    
    (format t " PASSED~%")
    t))

(defun test-layer-norm-gradient ()
  "Test layer normalization backward pass"
  (format t "Testing Layer Norm - Gradient Computation...")
  
  ;; Create input with gradient tracking
  (let* ((input (make-tensor (make-array '(2 3) 
                                         :element-type 'double-float
                                         :initial-contents '((1.0d0 2.0d0 3.0d0)
                                                            (4.0d0 5.0d0 6.0d0)))
                             :requires-grad t))
         (ln (layer-norm 3))
         (output (forward ln input))
         (loss (tsum output)))
    
    ;; Backward pass
    (backward loss)
    
    ;; Input should have gradients
    (assert (not (null (tensor-grad input)))
            () "Input should have gradients after backward")
    
    ;; Gamma and beta should have gradients
    (let ((gamma (norm-gamma ln))
          (beta (norm-beta ln)))
      (assert (not (null (tensor-grad gamma)))
              () "Gamma should have gradients")
      (assert (not (null (tensor-grad beta)))
              () "Beta should have gradients"))
    
    (format t " PASSED~%")
    t))

(defun test-layer-norm-parameters ()
  "Test that layer norm has learnable parameters"
  (format t "Testing Layer Norm - Learnable Parameters...")
  
  (let* ((ln (layer-norm 5))
         (params (parameters ln)))
    
    ;; Should have 2 parameters: gamma and beta
    (assert (= (length params) 2)
            () "Layer norm should have 2 parameters")
    
    ;; Check shapes
    (let ((gamma (first params))
          (beta (second params)))
      (assert (equal (tensor-shape gamma) '(5))
              () "Gamma shape mismatch")
      (assert (equal (tensor-shape beta) '(5))
              () "Beta shape mismatch")
      
      ;; Check initialization: gamma=1, beta=0
      (assert (every (lambda (x) (= x 1.0d0))
                     (loop for i below 5 collect (aref (tensor-data gamma) i)))
              () "Gamma should be initialized to 1")
      (assert (every (lambda (x) (= x 0.0d0))
                     (loop for i below 5 collect (aref (tensor-data beta) i)))
              () "Beta should be initialized to 0"))
    
    (format t " PASSED~%")
    t))

;;;; ============================================================================
;;;; Batch Normalization Tests
;;;; ============================================================================

(defun test-batch-norm-basic ()
  "Test basic batch normalization forward pass"
  (format t "~%Testing Batch Norm - Basic Forward Pass...")
  
  ;; Create input: (3, 2) - batch=3, features=2
  (let* ((input-data (make-array '(3 2) 
                                 :element-type 'double-float
                                 :initial-contents '((1.0d0 2.0d0)
                                                    (3.0d0 4.0d0)
                                                    (5.0d0 6.0d0))))
         (input (make-tensor input-data :requires-grad t))
         (bn (batch-norm 2))
         (output (forward bn input)))
    
    ;; Output should have same shape as input
    (assert (equal (tensor-shape output) '(3 2))
            () "Batch norm output shape mismatch")
    
    ;; In training mode, each feature should be normalized across batch
    ;; Feature 0: [1, 3, 5] -> mean=3, var=8/3
    ;; Feature 1: [2, 4, 6] -> mean=4, var=8/3
    (let ((out-data (tensor-data output)))
      ;; Check that feature means are ~0
      (let ((mean-f0 (/ (+ (aref out-data 0 0)
                           (aref out-data 1 0)
                           (aref out-data 2 0))
                        3.0d0))
            (mean-f1 (/ (+ (aref out-data 0 1)
                           (aref out-data 1 1)
                           (aref out-data 2 1))
                        3.0d0)))
        (assert (< (abs mean-f0) 1d-6)
                () "Batch norm: feature 0 mean should be ~0")
        (assert (< (abs mean-f1) 1d-6)
                () "Batch norm: feature 1 mean should be ~0")))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-training-eval ()
  "Test batch norm behavior in training vs evaluation mode"
  (format t "Testing Batch Norm - Training vs Eval Mode...")
  
  (let* ((input (make-tensor (make-array '(4 3) 
                                         :element-type 'double-float
                                         :initial-contents '((1.0d0 2.0d0 3.0d0)
                                                            (4.0d0 5.0d0 6.0d0)
                                                            (7.0d0 8.0d0 9.0d0)
                                                            (10.0d0 11.0d0 12.0d0)))
                             :requires-grad nil))
         (bn (batch-norm 3)))
    
    ;; Forward in training mode
    (train-mode bn)
    (let ((output-train (forward bn input)))
      (assert (equal (tensor-shape output-train) '(4 3))
              () "Training output shape mismatch"))
    
    ;; Check running statistics were updated
    (let ((running-mean-data (tensor-data (running-mean bn)))
          (running-var-data (tensor-data (running-var bn))))
      ;; Running mean should be updated (not exactly equal to batch mean due to momentum)
      (assert (> (abs (aref running-mean-data 0)) 0.0d0)
              () "Running mean should be updated after training"))
    
    ;; Forward in eval mode
    (eval-mode bn)
    (let ((output-eval (forward bn input)))
      (assert (equal (tensor-shape output-eval) '(4 3))
              () "Eval output shape mismatch"))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-gradient ()
  "Test batch normalization backward pass"
  (format t "Testing Batch Norm - Gradient Computation...")
  
  (let* ((input (make-tensor (make-array '(3 2) 
                                         :element-type 'double-float
                                         :initial-contents '((1.0d0 2.0d0)
                                                            (3.0d0 4.0d0)
                                                            (5.0d0 6.0d0)))
                             :requires-grad t))
         (bn (batch-norm 2))
         (output (forward bn input))
         (loss (tsum output)))
    
    ;; Backward pass
    (backward loss)
    
    ;; Input should have gradients
    (assert (not (null (tensor-grad input)))
            () "Input should have gradients after backward")
    
    ;; Gamma and beta should have gradients
    (let ((gamma (norm-gamma bn))
          (beta (norm-beta bn)))
      (assert (not (null (tensor-grad gamma)))
              () "Gamma should have gradients")
      (assert (not (null (tensor-grad beta)))
              () "Beta should have gradients"))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-parameters ()
  "Test that batch norm has learnable parameters"
  (format t "Testing Batch Norm - Learnable Parameters...")
  
  (let* ((bn (batch-norm 4))
         (params (parameters bn)))
    
    ;; Should have 2 parameters: gamma and beta
    (assert (= (length params) 2)
            () "Batch norm should have 2 parameters")
    
    ;; Check shapes
    (let ((gamma (first params))
          (beta (second params)))
      (assert (equal (tensor-shape gamma) '(4))
              () "Gamma shape mismatch")
      (assert (equal (tensor-shape beta) '(4))
              () "Beta shape mismatch")
      
      ;; Check initialization: gamma=1, beta=0
      (assert (every (lambda (x) (= x 1.0d0))
                     (loop for i below 4 collect (aref (tensor-data gamma) i)))
              () "Gamma should be initialized to 1")
      (assert (every (lambda (x) (= x 0.0d0))
                     (loop for i below 4 collect (aref (tensor-data beta) i)))
              () "Beta should be initialized to 0"))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-running-stats ()
  "Test batch norm running statistics accumulation"
  (format t "Testing Batch Norm - Running Statistics...")
  
  (let ((bn (batch-norm 2 :momentum 0.1d0)))
    ;; Initially running mean=0, running var=1
    (let ((rm-before (tensor-data (running-mean bn)))
          (rv-before (tensor-data (running-var bn))))
      (assert (= (aref rm-before 0) 0.0d0)
              () "Initial running mean should be 0")
      (assert (= (aref rv-before 0) 1.0d0)
              () "Initial running var should be 1"))
    
    ;; Forward pass in training mode
    (train-mode bn)
    (let ((input (make-tensor (make-array '(3 2) 
                                          :element-type 'double-float
                                          :initial-contents '((1.0d0 2.0d0)
                                                             (3.0d0 4.0d0)
                                                             (5.0d0 6.0d0)))
                              :requires-grad nil)))
      (forward bn input))
    
    ;; Running statistics should be updated
    (let ((rm-after (tensor-data (running-mean bn)))
          (rv-after (tensor-data (running-var bn))))
      (assert (not (= (aref rm-after 0) 0.0d0))
              () "Running mean should be updated")
      (assert (not (= (aref rv-after 0) 1.0d0))
              () "Running var should be updated"))
    
    (format t " PASSED~%")
    t))

;;;; ============================================================================
;;;; Edge Cases and Numerical Stability Tests
;;;; ============================================================================

(defun test-layer-norm-eps ()
  "Test layer norm epsilon for numerical stability"
  (format t "Testing Layer Norm - Numerical Stability (eps)...")
  
  ;; Create input with constant values (zero variance)
  (let* ((input (make-tensor (make-array '(2 3) 
                                         :element-type 'double-float
                                         :initial-element 5.0d0)
                             :requires-grad t))
         (ln (layer-norm 3 :eps 1d-5)))
    
    ;; This should not crash due to division by zero
    (let ((output (forward ln input)))
      (assert (not (null output))
              () "Layer norm should handle zero variance"))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-eps ()
  "Test batch norm epsilon for numerical stability"
  (format t "Testing Batch Norm - Numerical Stability (eps)...")
  
  ;; Create input with constant values (zero variance)
  (let* ((input (make-tensor (make-array '(4 2) 
                                         :element-type 'double-float
                                         :initial-element 3.0d0)
                             :requires-grad t))
         (bn (batch-norm 2 :eps 1d-5)))
    
    ;; This should not crash due to division by zero
    (let ((output (forward bn input)))
      (assert (not (null output))
              () "Batch norm should handle zero variance"))
    
    (format t " PASSED~%")
    t))

(defun test-layer-norm-single-sample ()
  "Test layer norm with single sample"
  (format t "Testing Layer Norm - Single Sample...")
  
  (let* ((input (make-tensor (make-array '(1 5) 
                                         :element-type 'double-float
                                         :initial-contents '((1.0d0 2.0d0 3.0d0 4.0d0 5.0d0)))
                             :requires-grad t))
         (ln (layer-norm 5))
         (output (forward ln input)))
    
    (assert (equal (tensor-shape output) '(1 5))
            () "Single sample output shape mismatch")
    
    ;; Check normalization
    (let* ((out-data (tensor-data output))
           (mean 0.0d0))
      (dotimes (i 5)
        (incf mean (aref out-data 0 i)))
      (setf mean (/ mean 5))
      (assert (< (abs mean) 1d-6)
              () "Single sample mean should be ~0"))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-single-sample ()
  "Test batch norm with single sample (should still work)"
  (format t "Testing Batch Norm - Single Sample...")
  
  (let* ((input (make-tensor (make-array '(1 3) 
                                         :element-type 'double-float
                                         :initial-contents '((1.0d0 2.0d0 3.0d0)))
                             :requires-grad t))
         (bn (batch-norm 3))
         (output (forward bn input)))
    
    (assert (equal (tensor-shape output) '(1 3))
            () "Single sample output shape mismatch")
    
    (format t " PASSED~%")
    t))

;;;; ============================================================================
;;;; Integration Tests
;;;; ============================================================================

(defun test-layer-norm-in-network ()
  "Test layer norm integrated in a sequential network"
  (format t "Testing Layer Norm - Integration in Network...")
  
  (let* ((model (sequential
                  (linear 4 8)
                  (layer-norm 8)
                  (relu-layer)
                  (linear 8 2)))
         (input (randn '(3 4) :requires-grad t))
         (output (forward model input)))
    
    (assert (equal (tensor-shape output) '(3 2))
            () "Network with layer norm output shape mismatch")
    
    ;; Test backward pass
    (let ((loss (tsum output)))
      (backward loss)
      (assert (not (null (tensor-grad input)))
              () "Gradients should flow through layer norm"))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-in-network ()
  "Test batch norm integrated in a sequential network"
  (format t "Testing Batch Norm - Integration in Network...")
  
  (let* ((model (sequential
                  (linear 4 8)
                  (batch-norm 8)
                  (relu-layer)
                  (linear 8 2)))
         (input (randn '(3 4) :requires-grad t)))
    
    ;; Training mode
    (train-mode model)
    (let ((output-train (forward model input)))
      (assert (equal (tensor-shape output-train) '(3 2))
              () "Network with batch norm training output shape mismatch"))
    
    ;; Eval mode
    (eval-mode model)
    (let ((output-eval (forward model input)))
      (assert (equal (tensor-shape output-eval) '(3 2))
              () "Network with batch norm eval output shape mismatch"))
    
    (format t " PASSED~%")
    t))

;;;; ============================================================================
;;;; 4D Batch Normalization Tests (for CNNs)
;;;; ============================================================================

(defun test-batch-norm-4d-basic ()
  "Test batch normalization with 4D input (batch, channels, height, width)"
  (format t "Testing Batch Norm - 4D Input (CNN-style)...")
  
  ;; Create 4D input: (2, 3, 4, 4) - 2 samples, 3 channels, 4x4 spatial
  (let* ((input (randn '(2 3 4 4) :requires-grad t))
         (bn (batch-norm 3))
         (output (forward bn input)))
    
    ;; Output should have same shape as input
    (assert (equal (tensor-shape output) '(2 3 4 4))
            () "Batch norm 4D output shape mismatch")
    
    ;; Check that normalization happened (output values should be different from input)
    (let ((in-data (tensor-data input))
          (out-data (tensor-data output)))
      (assert (not (= (aref in-data 0 0 0 0) (aref out-data 0 0 0 0)))
              () "Batch norm should transform input values"))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-4d-channel-independence ()
  "Test that batch norm normalizes each channel independently"
  (format t "Testing Batch Norm - 4D Channel Independence...")
  
  ;; Create input with different scales per channel
  (let* ((input-data (make-array '(2 2 3 3) :element-type 'double-float))
         ;; Channel 0: values around 10
         ;; Channel 1: values around 100
         (bn (batch-norm 2)))
    
    ;; Fill channel 0 with values around 10
    (dotimes (b 2)
      (dotimes (h 3)
        (dotimes (w 3)
          (setf (aref input-data b 0 h w) (+ 10.0d0 (random 2.0d0))))))
    
    ;; Fill channel 1 with values around 100
    (dotimes (b 2)
      (dotimes (h 3)
        (dotimes (w 3)
          (setf (aref input-data b 1 h w) (+ 100.0d0 (random 2.0d0))))))
    
    (let* ((input (make-tensor input-data :requires-grad t))
           (output (forward bn input))
           (out-data (tensor-data output)))
      
      ;; After normalization, both channels should have similar scale
      ;; Check that mean per channel is close to 0
      (dotimes (c 2)
        (let ((sum 0.0d0)
              (count 0))
          (dotimes (b 2)
            (dotimes (h 3)
              (dotimes (w 3)
                (incf sum (aref out-data b c h w))
                (incf count))))
          (let ((mean (/ sum count)))
            (assert (< (abs mean) 0.5d0)
                    () "Batch norm 4D: channel ~d mean should be close to 0, got ~a" c mean)))))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-4d-training-vs-eval ()
  "Test batch norm 4D behavior in training vs evaluation mode"
  (format t "Testing Batch Norm - 4D Training vs Eval Mode...")
  
  (let* ((bn (batch-norm 3))
         (input1 (randn '(4 3 8 8) :requires-grad t))
         (input2 (randn '(4 3 8 8) :requires-grad t)))
    
    ;; Training mode - compute statistics from input
    (train-mode bn)
    (let ((out1 (forward bn input1))
          (out2 (forward bn input2)))
      (assert (equal (tensor-shape out1) '(4 3 8 8))
              () "Batch norm 4D training output shape mismatch")
      (assert (equal (tensor-shape out2) '(4 3 8 8))
              () "Batch norm 4D training output shape mismatch"))
    
    ;; Eval mode - use running statistics
    (eval-mode bn)
    (let* ((out3 (forward bn input1))
           (out4 (forward bn input1))
           (data3 (tensor-data out3))
           (data4 (tensor-data out4)))
      (assert (equal (tensor-shape out3) '(4 3 8 8))
              () "Batch norm 4D eval output shape mismatch")
      
      ;; In eval mode, same input should give same output (deterministic)
      (dotimes (b 4)
        (dotimes (c 3)
          (dotimes (h 8)
            (dotimes (w 8)
              (assert (< (abs (- (aref data3 b c h w) (aref data4 b c h w))) 1d-9)
                      () "Batch norm 4D eval should be deterministic"))))))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-4d-gradient ()
  "Test batch norm 4D gradient computation"
  (format t "Testing Batch Norm - 4D Gradient Computation...")
  
  (let* ((bn (batch-norm 2))
         (input (randn '(2 2 4 4) :requires-grad t))
         (output (forward bn input))
         ;; Create upstream gradient as a tensor
         (grad (ones '(2 2 4 4))))
    
    (setf (tensor-grad output) grad)
    
    ;; Backward pass
    (backward output)
    
    ;; Input should have gradient
    (let ((input-grad (tensor-grad input)))
      (assert (not (null input-grad))
              () "Batch norm 4D should compute input gradient")
      
      ;; Gradient should be an array or tensor with matching dimensions
      (let ((grad-dims (if (typep input-grad 'tensor)
                          (tensor-shape input-grad)
                          (array-dimensions input-grad))))
        (assert (equal grad-dims '(2 2 4 4))
                () "Batch norm 4D gradient shape mismatch, got ~a" grad-dims)))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-4d-single-batch ()
  "Test batch norm 4D with single batch element"
  (format t "Testing Batch Norm - 4D Single Batch Element...")
  
  (let* ((bn (batch-norm 3))
         (input (randn '(1 3 5 5) :requires-grad t)))
    
    ;; Should work even with batch size 1
    (train-mode bn)
    (let ((output (forward bn input)))
      (assert (equal (tensor-shape output) '(1 3 5 5))
              () "Batch norm 4D single batch output shape mismatch"))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-4d-large-spatial ()
  "Test batch norm 4D with large spatial dimensions"
  (format t "Testing Batch Norm - 4D Large Spatial Dimensions...")
  
  (let* ((bn (batch-norm 8))
         ;; Small batch, many channels, large spatial
         (input (randn '(2 8 32 32) :requires-grad t)))
    
    (train-mode bn)
    (let ((output (forward bn input)))
      (assert (equal (tensor-shape output) '(2 8 32 32))
              () "Batch norm 4D large spatial output shape mismatch")
      
      ;; Verify normalization per channel
      (let ((out-data (tensor-data output)))
        (dotimes (c 8)
          (let ((sum 0.0d0)
                (count 0))
            (dotimes (b 2)
              (dotimes (h 32)
                (dotimes (w 32)
                  (incf sum (aref out-data b c h w))
                  (incf count))))
            (let ((mean (/ sum count)))
              (assert (< (abs mean) 0.1d0)
                      () "Batch norm 4D large spatial: channel mean should be close to 0"))))))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-2d-vs-4d-consistency ()
  "Test that 2D batch norm is consistent with reshaped 4D"
  (format t "Testing Batch Norm - 2D vs 4D Consistency...")
  
  ;; When spatial dimensions are 1x1, 4D should behave like 2D
  (let* ((bn-2d (batch-norm 4))
         (bn-4d (batch-norm 4))
         (input-2d-data (make-array '(3 4) :element-type 'double-float))
         (input-4d-data (make-array '(3 4 1 1) :element-type 'double-float)))
    
    ;; Fill with same data
    (dotimes (b 3)
      (dotimes (c 4)
        (let ((val (random 10.0d0)))
          (setf (aref input-2d-data b c) val)
          (setf (aref input-4d-data b c 0 0) val))))
    
    (let* ((input-2d (make-tensor input-2d-data :requires-grad t))
           (input-4d (make-tensor input-4d-data :requires-grad t))
           (output-2d nil)
           (output-4d nil))
      
      (train-mode bn-2d)
      (train-mode bn-4d)
      
      (setf output-2d (forward bn-2d input-2d))
      (setf output-4d (forward bn-4d input-4d))
      
      (let ((data-2d (tensor-data output-2d))
            (data-4d (tensor-data output-4d)))
        
        ;; Results should be very similar
        (dotimes (b 3)
          (dotimes (c 4)
            (let ((val-2d (aref data-2d b c))
                  (val-4d (aref data-4d b c 0 0)))
              (assert (< (abs (- val-2d val-4d)) 1d-6)
                      () "Batch norm 2D and 4D should produce similar results for 1x1 spatial"))))))
    
    (format t " PASSED~%")
    t))

;;;; ============================================================================
;;;; 3D Batch Normalization Tests (for 1D CNNs)
;;;; ============================================================================

(defun test-batch-norm-3d-basic ()
  "Test batch normalization with 3D input (batch, channels, length)"
  (format t "Testing Batch Norm - 3D Input (1D CNN-style)...")
  
  ;; Create 3D input: (2, 4, 16) - 2 samples, 4 channels, 16 temporal steps
  (let* ((input (randn '(2 4 16) :requires-grad t))
         (bn (batch-norm 4))
         (output (forward bn input)))
    
    ;; Output should have same shape as input
    (assert (equal (tensor-shape output) '(2 4 16))
            () "Batch norm 3D output shape mismatch")
    
    ;; Check that normalization happened
    (let ((in-data (tensor-data input))
          (out-data (tensor-data output)))
      (assert (not (= (aref in-data 0 0 0) (aref out-data 0 0 0)))
              () "Batch norm 3D should transform input values"))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-3d-channel-independence ()
  "Test that 3D batch norm normalizes each channel independently"
  (format t "Testing Batch Norm - 3D Channel Independence...")
  
  ;; Create input with different scales per channel
  (let* ((input-data (make-array '(2 3 10) :element-type 'double-float))
         (bn (batch-norm 3)))
    
    ;; Fill channel 0 with values around 5
    (dotimes (b 2)
      (dotimes (l 10)
        (setf (aref input-data b 0 l) (+ 5.0d0 (random 1.0d0)))))
    
    ;; Fill channel 1 with values around 50
    (dotimes (b 2)
      (dotimes (l 10)
        (setf (aref input-data b 1 l) (+ 50.0d0 (random 1.0d0)))))
    
    ;; Fill channel 2 with values around 500
    (dotimes (b 2)
      (dotimes (l 10)
        (setf (aref input-data b 2 l) (+ 500.0d0 (random 1.0d0)))))
    
    (let* ((input (make-tensor input-data :requires-grad t))
           (output (forward bn input))
           (out-data (tensor-data output)))
      
      ;; After normalization, all channels should have similar scale
      ;; Check that mean per channel is close to 0
      (dotimes (c 3)
        (let ((sum 0.0d0)
              (count 0))
          (dotimes (b 2)
            (dotimes (l 10)
              (incf sum (aref out-data b c l))
              (incf count)))
          (let ((mean (/ sum count)))
            (assert (< (abs mean) 0.5d0)
                    () "Batch norm 3D: channel ~d mean should be close to 0, got ~a" c mean)))))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-3d-training-vs-eval ()
  "Test 3D batch norm behavior in training vs evaluation mode"
  (format t "Testing Batch Norm - 3D Training vs Eval Mode...")
  
  (let* ((bn (batch-norm 3))
         (input1 (randn '(4 3 20) :requires-grad t))
         (input2 (randn '(4 3 20) :requires-grad t)))
    
    ;; Training mode - compute statistics from input
    (train-mode bn)
    (let ((out1 (forward bn input1))
          (out2 (forward bn input2)))
      (assert (equal (tensor-shape out1) '(4 3 20))
              () "Batch norm 3D training output shape mismatch")
      (assert (equal (tensor-shape out2) '(4 3 20))
              () "Batch norm 3D training output shape mismatch"))
    
    ;; Eval mode - use running statistics (should be deterministic)
    (eval-mode bn)
    (let* ((out3 (forward bn input1))
           (out4 (forward bn input1))
           (data3 (tensor-data out3))
           (data4 (tensor-data out4)))
      (assert (equal (tensor-shape out3) '(4 3 20))
              () "Batch norm 3D eval output shape mismatch")
      
      ;; In eval mode, same input should give same output
      (dotimes (b 4)
        (dotimes (c 3)
          (dotimes (l 20)
            (assert (< (abs (- (aref data3 b c l) (aref data4 b c l))) 1d-9)
                    () "Batch norm 3D eval should be deterministic")))))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-3d-gradient ()
  "Test 3D batch norm gradient computation"
  (format t "Testing Batch Norm - 3D Gradient Computation...")
  
  (let* ((bn (batch-norm 2))
         (input (randn '(3 2 12) :requires-grad t))
         (output (forward bn input))
         (grad (ones '(3 2 12))))
    
    (setf (tensor-grad output) grad)
    (backward output)
    
    (let ((input-grad (tensor-grad input)))
      (assert (not (null input-grad))
              () "Batch norm 3D should compute input gradient")
      
      (let ((grad-dims (if (typep input-grad 'tensor)
                          (tensor-shape input-grad)
                          (array-dimensions input-grad))))
        (assert (equal grad-dims '(3 2 12))
                () "Batch norm 3D gradient shape mismatch, got ~a" grad-dims)))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-3d-long-sequence ()
  "Test 3D batch norm with long temporal sequences"
  (format t "Testing Batch Norm - 3D Long Sequences...")
  
  (let* ((bn (batch-norm 8))
         ;; Small batch, multiple channels, long sequence
         (input (randn '(2 8 128) :requires-grad t)))
    
    (train-mode bn)
    (let ((output (forward bn input)))
      (assert (equal (tensor-shape output) '(2 8 128))
              () "Batch norm 3D long sequence output shape mismatch")
      
      ;; Verify normalization per channel
      (let ((out-data (tensor-data output)))
        (dotimes (c 8)
          (let ((sum 0.0d0)
                (count 0))
            (dotimes (b 2)
              (dotimes (l 128)
                (incf sum (aref out-data b c l))
                (incf count)))
            (let ((mean (/ sum count)))
              (assert (< (abs mean) 0.1d0)
                      () "Batch norm 3D long sequence: channel mean should be close to 0"))))))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-3d-single-timestep ()
  "Test 3D batch norm with single timestep (edge case)"
  (format t "Testing Batch Norm - 3D Single Timestep...")
  
  (let* ((bn (batch-norm 4))
         ;; Multiple samples, multiple channels, single time step
         (input (randn '(8 4 1) :requires-grad t)))
    
    (train-mode bn)
    (let ((output (forward bn input)))
      (assert (equal (tensor-shape output) '(8 4 1))
              () "Batch norm 3D single timestep output shape mismatch"))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-2d-vs-3d-consistency ()
  "Test that 2D and 3D batch norm are consistent when length=1"
  (format t "Testing Batch Norm - 2D vs 3D Consistency...")
  
  ;; When temporal dimension is 1, 3D should behave similarly to 2D
  (let* ((bn-2d (batch-norm 3))
         (bn-3d (batch-norm 3))
         (input-2d-data (make-array '(4 3) :element-type 'double-float))
         (input-3d-data (make-array '(4 3 1) :element-type 'double-float)))
    
    ;; Fill with same data
    (dotimes (b 4)
      (dotimes (c 3)
        (let ((val (random 10.0d0)))
          (setf (aref input-2d-data b c) val)
          (setf (aref input-3d-data b c 0) val))))
    
    (let* ((input-2d (make-tensor input-2d-data :requires-grad t))
           (input-3d (make-tensor input-3d-data :requires-grad t))
           (output-2d nil)
           (output-3d nil))
      
      (train-mode bn-2d)
      (train-mode bn-3d)
      
      (setf output-2d (forward bn-2d input-2d))
      (setf output-3d (forward bn-3d input-3d))
      
      (let ((data-2d (tensor-data output-2d))
            (data-3d (tensor-data output-3d)))
        
        ;; Results should be very similar
        (dotimes (b 4)
          (dotimes (c 3)
            (let ((val-2d (aref data-2d b c))
                  (val-3d (aref data-3d b c 0)))
              (assert (< (abs (- val-2d val-3d)) 1d-6)
                      () "Batch norm 2D and 3D should produce similar results for length=1"))))))
    
    (format t " PASSED~%")
    t))

;;;; ============================================================================
;;;; 5D Batch Normalization Tests (for 3D CNNs)
;;;; ============================================================================

(defun test-batch-norm-5d-basic ()
  "Test batch normalization with 5D input (batch, channels, depth, height, width)"
  (format t "Testing Batch Norm - 5D Input (3D CNN-style)...")
  
  ;; Create 5D input: (2, 3, 4, 4, 4) - 2 samples, 3 channels, 4x4x4 volume
  (let* ((input (randn '(2 3 4 4 4) :requires-grad t))
         (bn (batch-norm 3))
         (output (forward bn input)))
    
    ;; Output should have same shape as input
    (assert (equal (tensor-shape output) '(2 3 4 4 4))
            () "Batch norm 5D output shape mismatch")
    
    ;; Check that normalization happened
    (let ((in-data (tensor-data input))
          (out-data (tensor-data output)))
      (assert (not (= (aref in-data 0 0 0 0 0) (aref out-data 0 0 0 0 0)))
              () "Batch norm 5D should transform input values"))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-5d-channel-normalization ()
  "Test that 5D batch norm normalizes per channel across all spatial dimensions"
  (format t "Testing Batch Norm - 5D Channel Normalization...")
  
  (let* ((bn (batch-norm 2))
         (input (randn '(3 2 5 5 5) :requires-grad t)))
    
    (train-mode bn)
    (let* ((output (forward bn input))
           (out-data (tensor-data output)))
      
      ;; Check that each channel is normalized (mean close to 0)
      (dotimes (c 2)
        (let ((sum 0.0d0)
              (count 0))
          (dotimes (b 3)
            (dotimes (d 5)
              (dotimes (h 5)
                (dotimes (w 5)
                  (incf sum (aref out-data b c d h w))
                  (incf count)))))
          (let ((mean (/ sum count)))
            (assert (< (abs mean) 0.2d0)
                    () "Batch norm 5D: channel ~d mean should be close to 0, got ~a" c mean)))))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-5d-training-vs-eval ()
  "Test 5D batch norm in training vs evaluation mode"
  (format t "Testing Batch Norm - 5D Training vs Eval...")
  
  (let* ((bn (batch-norm 2))
         (input1 (randn '(2 2 3 3 3) :requires-grad t))
         (input2 (randn '(2 2 3 3 3) :requires-grad t)))
    
    ;; Training mode
    (train-mode bn)
    (let ((out1 (forward bn input1))
          (out2 (forward bn input2)))
      (assert (equal (tensor-shape out1) '(2 2 3 3 3))
              () "Batch norm 5D training shape mismatch"))
    
    ;; Eval mode - should be deterministic
    (eval-mode bn)
    (let* ((out3 (forward bn input1))
           (out4 (forward bn input1))
           (data3 (tensor-data out3))
           (data4 (tensor-data out4)))
      (dotimes (b 2)
        (dotimes (c 2)
          (dotimes (d 3)
            (dotimes (h 3)
              (dotimes (w 3)
                (assert (< (abs (- (aref data3 b c d h w) (aref data4 b c d h w))) 1d-9)
                        () "Batch norm 5D eval should be deterministic")))))))
    
    (format t " PASSED~%")
    t))

(defun test-batch-norm-5d-gradient ()
  "Test 5D batch norm gradient computation"
  (format t "Testing Batch Norm - 5D Gradient...")
  
  (let* ((bn (batch-norm 2))
         (input (randn '(2 2 3 3 3) :requires-grad t))
         (output (forward bn input))
         (grad (ones '(2 2 3 3 3))))
    
    (setf (tensor-grad output) grad)
    (backward output)
    
    (let ((input-grad (tensor-grad input)))
      (assert (not (null input-grad))
              () "Batch norm 5D should compute gradient")
      
      (let ((grad-dims (if (typep input-grad 'tensor)
                          (tensor-shape input-grad)
                          (array-dimensions input-grad))))
        (assert (equal grad-dims '(2 2 3 3 3))
                () "Batch norm 5D gradient shape mismatch")))
    
    (format t " PASSED~%")
    t))

;;;; ============================================================================
;;;; Generalized Layer Normalization Tests
;;;; ============================================================================

(defun test-layer-norm-4d ()
  "Test generalized layer norm with 4D input"
  (format t "Testing Layer Norm - 4D Input (General ND)...")
  
  ;; (batch, seq1, seq2, features)
  (let* ((input (randn '(2 3 4 8) :requires-grad t))
         (ln (layer-norm 8))
         (output (forward ln input)))
    
    (assert (equal (tensor-shape output) '(2 3 4 8))
            () "Layer norm 4D output shape mismatch")
    
    ;; Check normalization: each position should have mean ~0
    (let ((out-data (tensor-data output)))
      (dotimes (b 2)
        (dotimes (s1 3)
          (dotimes (s2 4)
            (let ((sum 0.0d0))
              (dotimes (f 8)
                (incf sum (aref out-data b s1 s2 f)))
              (let ((mean (/ sum 8)))
                (assert (< (abs mean) 0.01d0)
                        () "Layer norm 4D: position mean should be close to 0")))))))
    
    (format t " PASSED~%")
    t))

(defun test-layer-norm-5d ()
  "Test generalized layer norm with 5D input"
  (format t "Testing Layer Norm - 5D Input (General ND)...")
  
  ;; (batch, dim1, dim2, dim3, features)
  (let* ((input (randn '(2 2 2 3 6) :requires-grad t))
         (ln (layer-norm 6))
         (output (forward ln input)))
    
    (assert (equal (tensor-shape output) '(2 2 2 3 6))
            () "Layer norm 5D output shape mismatch")
    
    ;; Spot check a few positions for normalization
    (let ((out-data (tensor-data output)))
      (dotimes (b 2)
        (dotimes (d1 2)
          (let ((sum 0.0d0))
            (dotimes (f 6)
              (incf sum (aref out-data b d1 0 0 f)))
            (let ((mean (/ sum 6)))
              (assert (< (abs mean) 0.1d0)
                      () "Layer norm 5D: position mean should be close to 0"))))))
    
    (format t " PASSED~%")
    t))

(defun test-layer-norm-nd-consistency ()
  "Test that specialized and generalized layer norm implementations are consistent"
  (format t "Testing Layer Norm - ND Consistency...")
  
  ;; For 3D input, both the specialized and general code paths should work
  (let* ((ln (layer-norm 4))
         (input-data (make-array '(2 3 4) :element-type 'double-float)))
    
    ;; Fill with random data
    (dotimes (b 2)
      (dotimes (s 3)
        (dotimes (f 4)
          (setf (aref input-data b s f) (random 10.0d0)))))
    
    (let* ((input (make-tensor input-data :requires-grad t))
           (output (forward ln input)))
      
      ;; Should produce normalized output
      (assert (equal (tensor-shape output) '(2 3 4))
              () "Layer norm ND consistency: output shape mismatch")
      
      ;; Check that normalization worked
      (let ((out-data (tensor-data output)))
        (dotimes (b 2)
          (dotimes (s 3)
            (let ((sum 0.0d0))
              (dotimes (f 4)
                (incf sum (aref out-data b s f)))
              (let ((mean (/ sum 4)))
                (assert (< (abs mean) 1d-6)
                        () "Layer norm ND: mean should be close to 0")))))))
    
    (format t " PASSED~%")
    t))

;;;; ============================================================================
;;;; Test Runner
;;;; ============================================================================

(defun run-normalization-tests ()
  "Run all normalization tests"
  (format t "~%")
  (format t "╔════════════════════════════════════════════════════════════════╗~%")
  (format t "║  Normalization Layer Tests                                    ║~%")
  (format t "╚════════════════════════════════════════════════════════════════╝~%")
  (format t "~%")
  
  (let ((tests '(test-layer-norm-basic
                 test-layer-norm-3d
                 test-layer-norm-gradient
                 test-layer-norm-parameters
                 test-batch-norm-basic
                 test-batch-norm-training-eval
                 test-batch-norm-gradient
                 test-batch-norm-parameters
                 test-batch-norm-running-stats
                 test-layer-norm-eps
                 test-batch-norm-eps
                 test-layer-norm-single-sample
                 test-batch-norm-single-sample
                 test-layer-norm-in-network
                 test-batch-norm-in-network
                 ;; 3D batch norm tests (1D CNNs)
                 test-batch-norm-3d-basic
                 test-batch-norm-3d-channel-independence
                 test-batch-norm-3d-training-vs-eval
                 test-batch-norm-3d-gradient
                 test-batch-norm-3d-long-sequence
                 test-batch-norm-3d-single-timestep
                 test-batch-norm-2d-vs-3d-consistency
                 ;; 4D batch norm tests (2D CNNs)
                 test-batch-norm-4d-basic
                 test-batch-norm-4d-channel-independence
                 test-batch-norm-4d-training-vs-eval
                 test-batch-norm-4d-gradient
                 test-batch-norm-4d-single-batch
                 test-batch-norm-4d-large-spatial
                 test-batch-norm-2d-vs-4d-consistency
                 ;; 5D batch norm tests (3D CNNs)
                 test-batch-norm-5d-basic
                 test-batch-norm-5d-channel-normalization
                 test-batch-norm-5d-training-vs-eval
                 test-batch-norm-5d-gradient
                 ;; Generalized layer norm tests
                 test-layer-norm-4d
                 test-layer-norm-5d
                 test-layer-norm-nd-consistency))
        (passed 0)
        (failed 0))
    
    (dolist (test tests)
      (handler-case
          (progn
            (funcall test)
            (incf passed))
        (error (e)
          (format t "✗ FAILED: ~a~%" e)
          (incf failed))))
    
    (format t "~%Normalization Tests: ~d passed, ~d failed~%~%" passed failed)
    
    (values passed failed)))

