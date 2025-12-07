;;;; tests/test-residual.lisp - Tests for Residual Blocks

(in-package #:neural-lisp-tests)

(def-suite residual-tests
  :description "Test suite for residual blocks (ResNet, EfficientNet, ConvNeXt)"
  :in neural-lisp-tests)

(in-suite residual-tests)

;;;; Helper function for numerical comparisons
(defun assert-equal-residual (expected actual &optional (tolerance 1d-6))
  "Assert two values are equal within tolerance"
  (unless (< (abs (- expected actual)) tolerance)
    (error "Expected ~A but got ~A (difference: ~A, tolerance: ~A)"
           expected actual (abs (- expected actual)) tolerance)))

;;;; ============================================================================
;;;; ResNet Block Tests
;;;; ============================================================================

(test resnet-basic-block-no-downsample
  "Test ResNet basic block without downsampling (same channels, stride=1)"
  (let* ((batch-size 2)
         (channels 64)
         (height 8)
         (width 8)
         
         (block (make-instance 'neural-tensor-residual:resnet-basic-block
                              :in-channels channels
                              :out-channels channels
                              :stride 1))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward block input)))
    
    ;; Output shape should be unchanged
    (is (equal (tensor-shape output) (list batch-size channels height width)))
    
    ;; Output should be non-zero (conv + skip)
    (is (not (null output)))))

(test resnet-basic-block-with-downsample
  "Test ResNet basic block with downsampling (channel change and stride=2)"
  (let* ((batch-size 1)
         (in-channels 64)
         (out-channels 128)
         (height 16)
         (width 16)
         (stride 2)
         
         (block (make-instance 'neural-tensor-residual:resnet-basic-block
                              :in-channels in-channels
                              :out-channels out-channels
                              :stride stride))
         
         (input (make-tensor (make-array (list batch-size in-channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels height width)))
         
         (output (forward block input)))
    
    ;; Output should be halved in spatial dimensions and doubled in channels
    (is (equal (tensor-shape output) (list batch-size out-channels 8 8)))))

(test resnet-bottleneck-block-basic
  "Test ResNet bottleneck block"
  (let* ((batch-size 1)
         (in-channels 256)
         (out-channels 64)  ; Bottleneck channels (will be expanded by 4x)
         (height 14)
         (width 14)
         
         (block (make-instance 'neural-tensor-residual:resnet-bottleneck-block
                              :in-channels in-channels
                              :out-channels out-channels
                              :stride 1))
         
         (input (make-tensor (make-array (list batch-size in-channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels height width)))
         
         (output (forward block input)))
    
    ;; Output channels = out-channels * expansion (4)
    (is (equal (tensor-shape output) (list batch-size 256 height width)))))

(test resnet-bottleneck-block-expansion
  "Test ResNet bottleneck block channel expansion"
  (let* ((batch-size 2)
         (in-channels 64)
         (out-channels 64)
         (height 7)
         (width 7)
         
         (block (make-instance 'neural-tensor-residual:resnet-bottleneck-block
                              :in-channels in-channels
                              :out-channels out-channels
                              :stride 1))
         
         (input (make-tensor (make-array (list batch-size in-channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels height width)))
         
         (output (forward block input)))
    
    ;; Output channels should be 64 * 4 = 256 (with downsample to match)
    (is (equal (tensor-shape output) (list batch-size 256 height width)))))

(test resnet-downsample-layer
  "Test ResNet downsampling layer"
  (let* ((batch-size 1)
         (in-channels 64)
         (out-channels 128)
         (height 16)
         (width 16)
         (stride 2)
         
         (downsample (make-instance 'neural-tensor-residual:resnet-downsample
                                   :in-channels in-channels
                                   :out-channels out-channels
                                   :stride stride))
         
         (input (make-tensor (make-array (list batch-size in-channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels height width)))
         
         (output (forward downsample input)))
    
    (is (equal (tensor-shape output) (list batch-size out-channels 8 8)))))

;;;; ============================================================================
;;;; Squeeze-and-Excitation Tests
;;;; ============================================================================

(test squeeze-excitation-basic
  "Test squeeze-and-excitation module"
  (let* ((batch-size 2)
         (channels 64)
         (height 8)
         (width 8)
         
         (se (make-instance 'neural-tensor-residual:squeeze-excitation
                           :channels channels
                           :reduction 4))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward se input)))
    
    ;; Output shape should be unchanged (channel attention scaling)
    (is (equal (tensor-shape output) (list batch-size channels height width)))))

(test squeeze-excitation-different-reductions
  "Test SE with different reduction ratios"
  (let* ((input (make-tensor (make-array '(1 128 16 16)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(1 128 16 16))))
    
    ;; Test with reduction=8
    (let* ((se8 (make-instance 'neural-tensor-residual:squeeze-excitation
                              :channels 128
                              :reduction 8))
           (output8 (forward se8 input)))
      (is (equal (tensor-shape output8) '(1 128 16 16))))
    
    ;; Test with reduction=16
    (let* ((se16 (make-instance 'neural-tensor-residual:squeeze-excitation
                               :channels 128
                               :reduction 16))
           (output16 (forward se16 input)))
      (is (equal (tensor-shape output16) '(1 128 16 16))))))

;;;; ============================================================================
;;;; EfficientNet MBConv Block Tests
;;;; ============================================================================

(test mbconv-block-no-expansion
  "Test MBConv block with expand-ratio=1 (no expansion phase)"
  (let* ((batch-size 1)
         (channels 32)
         (height 14)
         (width 14)
         
         (block (make-instance 'neural-tensor-residual:mbconv-block
                              :in-channels channels
                              :out-channels channels
                              :expand-ratio 1
                              :kernel-size 3
                              :stride 1))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward block input)))
    
    (is (equal (tensor-shape output) (list batch-size channels height width)))))

(test mbconv-block-with-expansion
  "Test MBConv block with expansion"
  (let* ((batch-size 2)
         (in-channels 24)
         (out-channels 40)
         (height 28)
         (width 28)
         
         (block (make-instance 'neural-tensor-residual:mbconv-block
                              :in-channels in-channels
                              :out-channels out-channels
                              :expand-ratio 6
                              :kernel-size 3
                              :stride 1))
         
         (input (make-tensor (make-array (list batch-size in-channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels height width)))
         
         (output (forward block input)))
    
    (is (equal (tensor-shape output) (list batch-size out-channels height width)))))

(test mbconv-block-with-stride
  "Test MBConv block with stride=2 (downsampling)"
  (let* ((batch-size 1)
         (channels 32)
         (height 14)
         (width 14)
         
         (block (make-instance 'neural-tensor-residual:mbconv-block
                              :in-channels channels
                              :out-channels channels
                              :expand-ratio 6
                              :kernel-size 3
                              :stride 2))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward block input)))
    
    ;; Spatial dimensions should be halved
    (is (equal (tensor-shape output) (list batch-size channels 7 7)))))

(test mbconv-block-skip-connection
  "Test MBConv skip connection behavior"
  (let* ((batch-size 1)
         (channels 32)
         (height 8)
         (width 8)
         
         ;; Block with skip connection (stride=1, same channels)
         (block-skip (make-instance 'neural-tensor-residual:mbconv-block
                                   :in-channels channels
                                   :out-channels channels
                                   :expand-ratio 4
                                   :stride 1))
         
         ;; Block without skip (stride=2)
         (block-no-skip (make-instance 'neural-tensor-residual:mbconv-block
                                      :in-channels channels
                                      :out-channels channels
                                      :expand-ratio 4
                                      :stride 2))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width))))
    
    ;; With skip: same dimensions
    (let ((output-skip (forward block-skip input)))
      (is (equal (tensor-shape output-skip) (list batch-size channels height width))))
    
    ;; Without skip: downsampled
    (let ((output-no-skip (forward block-no-skip input)))
      (is (equal (tensor-shape output-no-skip) (list batch-size channels 4 4))))))

(test mbconv-block-with-se
  "Test MBConv block with squeeze-excitation"
  (let* ((batch-size 1)
         (channels 24)
         (height 16)
         (width 16)
         
         (block (make-instance 'neural-tensor-residual:mbconv-block
                              :in-channels channels
                              :out-channels channels
                              :expand-ratio 6
                              :kernel-size 5
                              :stride 1
                              :use-se t))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward block input)))
    
    (is (equal (tensor-shape output) (list batch-size channels height width)))))

(test mbconv-block-without-se
  "Test MBConv block without squeeze-excitation"
  (let* ((batch-size 1)
         (channels 24)
         (height 16)
         (width 16)
         
         (block (make-instance 'neural-tensor-residual:mbconv-block
                              :in-channels channels
                              :out-channels channels
                              :expand-ratio 6
                              :kernel-size 3
                              :stride 1
                              :use-se nil))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward block input)))
    
    (is (equal (tensor-shape output) (list batch-size channels height width)))))

;;;; ============================================================================
;;;; ConvNeXt Block Tests
;;;; ============================================================================

(test convnext-block-basic
  "Test ConvNeXt block basic functionality"
  (let* ((batch-size 1)
         (dim 96)
         (height 56)
         (width 56)
         
         (block (make-instance 'neural-tensor-residual:convnext-block
                              :dim dim))
         
         (input (make-tensor (make-array (list batch-size dim height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size dim height width)))
         
         (output (forward block input)))
    
    ;; Output shape should be unchanged (residual block)
    (is (equal (tensor-shape output) (list batch-size dim height width)))))

(test convnext-block-different-dims
  "Test ConvNeXt blocks with different channel dimensions"
  (let ((input-shapes '((1 96 56 56)
                        (2 192 28 28)
                        (1 384 14 14)
                        (1 768 7 7))))
    
    (dolist (shape input-shapes)
      (let* ((batch-size (first shape))
             (dim (second shape))
             (height (third shape))
             (width (fourth shape))
             
             (block (make-instance 'neural-tensor-residual:convnext-block
                                  :dim dim))
             
             (input (make-tensor (make-array shape
                                            :element-type 'double-float
                                            :initial-element 1.0d0)
                                :shape shape))
             
             (output (forward block input)))
        
        (is (equal (tensor-shape output) shape))))))

(test convnext-block-layer-scale
  "Test ConvNeXt block with custom layer scale initialization"
  (let* ((batch-size 1)
         (dim 128)
         (height 14)
         (width 14)
         
         (block (make-instance 'neural-tensor-residual:convnext-block
                              :dim dim
                              :layer-scale-init 1.0d-4))
         
         (input (make-tensor (make-array (list batch-size dim height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size dim height width)))
         
         (output (forward block input)))
    
    (is (equal (tensor-shape output) (list batch-size dim height width)))))

(test convnext-stage
  "Test ConvNeXt stage with multiple blocks"
  (let* ((batch-size 1)
         (dim 96)
         (depth 3)
         (height 56)
         (width 56)
         
         (stage (make-instance 'neural-tensor-residual:convnext-stage
                              :dim dim
                              :depth depth))
         
         (input (make-tensor (make-array (list batch-size dim height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size dim height width)))
         
         (output (forward stage input)))
    
    ;; After 3 blocks, shape should still be the same
    (is (equal (tensor-shape output) (list batch-size dim height width)))))

;;;; ============================================================================
;;;; Integration Tests
;;;; ============================================================================

(test residual-block-batch-processing
  "Test residual blocks with batch size > 1"
  (let* ((batch-size 8)
         (channels 64)
         (height 32)
         (width 32)
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width))))
    
    ;; Test ResNet basic block
    (let* ((resnet-block (make-instance 'neural-tensor-residual:resnet-basic-block
                                       :in-channels channels
                                       :out-channels channels
                                       :stride 1))
           (output (forward resnet-block input)))
      (is (equal (tensor-shape output) (list batch-size channels height width))))
    
    ;; Test MBConv block
    (let* ((mbconv (make-instance 'neural-tensor-residual:mbconv-block
                                 :in-channels channels
                                 :out-channels channels
                                 :expand-ratio 4
                                 :stride 1))
           (output (forward mbconv input)))
      (is (equal (tensor-shape output) (list batch-size channels height width))))
    
    ;; Test ConvNeXt block
    (let* ((convnext (make-instance 'neural-tensor-residual:convnext-block
                                   :dim channels))
           (output (forward convnext input)))
      (is (equal (tensor-shape output) (list batch-size channels height width))))))

(test residual-blocks-composition
  "Test composing multiple residual blocks"
  (let* ((batch-size 1)
         (channels 64)
         (height 32)
         (width 32)
         
         ;; Create a sequence of blocks
         (block1 (make-instance 'neural-tensor-residual:resnet-basic-block
                               :in-channels channels
                               :out-channels channels
                               :stride 1))
         (block2 (make-instance 'neural-tensor-residual:resnet-basic-block
                               :in-channels channels
                               :out-channels channels
                               :stride 1))
         (block3 (make-instance 'neural-tensor-residual:resnet-basic-block
                               :in-channels channels
                               :out-channels channels
                               :stride 1))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width))))
    
    ;; Forward through sequence
    (let ((out input))
      (setf out (forward block1 out))
      (setf out (forward block2 out))
      (setf out (forward block3 out))
      
      ;; Final output should still have same shape
      (is (equal (tensor-shape out) (list batch-size channels height width))))))

(test mixed-architecture-blocks
  "Test mixing different residual block types"
  (let* ((batch-size 1)
         (in-channels 64)
         (mid-channels 128)
         (height 32)
         (width 32)
         
         (input (make-tensor (make-array (list batch-size in-channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels height width))))
    
    ;; ResNet block first
    (let* ((resnet (make-instance 'neural-tensor-residual:resnet-basic-block
                                 :in-channels in-channels
                                 :out-channels mid-channels
                                 :stride 2))
           (out1 (forward resnet input)))
      
      ;; Then MBConv block
      (let* ((mbconv (make-instance 'neural-tensor-residual:mbconv-block
                                   :in-channels mid-channels
                                   :out-channels mid-channels
                                   :expand-ratio 4
                                   :stride 1))
             (out2 (forward mbconv out1)))
        
        ;; Final output shape
        (is (equal (tensor-shape out2) (list batch-size mid-channels 16 16)))))))

;;;; ============================================================================
;;;; Stochastic Depth Tests
;;;; ============================================================================

(test stochastic-depth-creation
  "Test basic creation of stochastic depth layer"
  (let ((sd (make-instance 'neural-tensor-residual:stochastic-depth
                          :drop-prob 0.2)))
    (is (not (null sd)))))

(test stochastic-depth-default-drop-prob
  "Test stochastic depth with default drop probability"
  (let ((sd (make-instance 'neural-tensor-residual:stochastic-depth)))
    (is (not (null sd)))))

(test stochastic-depth-zero-drop-prob
  "Test stochastic depth with zero drop probability (no dropout)"
  (let* ((sd (make-instance 'neural-tensor-residual:stochastic-depth
                           :drop-prob 0.0))
         (input (make-tensor (make-array '(4 16 8 8)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(4 16 8 8)))
         (output (forward sd input)))
    
    ;; With drop-prob=0, output should equal input
    (is (equal (tensor-shape output) (tensor-shape input)))
    (let ((input-data (tensor-data input))
          (output-data (tensor-data output)))
      ;; Check several elements are equal
      (is (= (row-major-aref input-data 0)
             (row-major-aref output-data 0)))
      (is (= (row-major-aref input-data 100)
             (row-major-aref output-data 100)))
      (is (= (row-major-aref input-data 1000)
             (row-major-aref output-data 1000))))))

(test stochastic-depth-eval-mode
  "Test stochastic depth in eval mode (deterministic, no dropout)"
  (let* ((sd (make-instance 'neural-tensor-residual:stochastic-depth
                           :drop-prob 0.5))
         (input (make-tensor (make-array '(2 8)
                                        :element-type 'double-float
                                        :initial-element 2.0d0)
                            :shape '(2 8))))
    
    ;; Set to eval mode
    (setf (neural-network::layer-training sd) nil)
    
    (let ((output (forward sd input)))
      ;; In eval mode, output should match input shape
      (is (equal (tensor-shape output) (tensor-shape input))))))

(test stochastic-depth-training-mode-shape
  "Test stochastic depth in training mode preserves shape"
  (let* ((sd (make-instance 'neural-tensor-residual:stochastic-depth
                           :drop-prob 0.3))
         (input (make-tensor (make-array '(8 32 16 16)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(8 32 16 16))))
    
    ;; Set to training mode
    (setf (neural-network::layer-training sd) t)
    
    (let ((output (forward sd input)))
      ;; Output shape should match input shape
      (is (equal (tensor-shape output) (tensor-shape input))))))

(test stochastic-depth-training-mode-stochasticity
  "Test stochastic depth in training mode produces variable outputs"
  (let* ((sd (make-instance 'neural-tensor-residual:stochastic-depth
                           :drop-prob 0.5))
         (input (make-tensor (make-array '(16 8)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(16 8)))
         (outputs nil))
    
    ;; Set to training mode
    (setf (neural-network::layer-training sd) t)
    
    ;; Use deterministic seed and run multiple times with different seeds
    (dotimes (i 20)
      (variational:set-random-seed (+ 1000 i))
      (let* ((output (forward sd input))
             (data (tensor-data output))
             (sum 0.0d0))
        ;; Sum all elements
        (dotimes (j (array-total-size data))
          (incf sum (row-major-aref data j)))
        (push sum outputs)))
    
    ;; With drop-prob=0.5, we should see at least 2 different output sums
    ;; (some samples dropped, some kept and scaled)
    (is (> (length (remove-duplicates outputs :test (lambda (a b) 
                                                      (< (abs (- a b)) 1.0d0))))
           1))))

(test stochastic-depth-training-has-zeros
  "Test stochastic depth drops samples (produces zeros) in training mode"
  (let* ((sd (make-instance 'neural-tensor-residual:stochastic-depth
                           :drop-prob 0.5))
         ;; Use 8 samples - with seed 42 this deterministically produces both kept and dropped
         (input (make-tensor (make-array '(8 4)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(8 4))))
    
    ;; Set to training mode and use fixed seed for deterministic behavior
    (setf (neural-network::layer-training sd) t)
    (variational:set-random-seed 42)
    
    (let* ((output (forward sd input))
           (data (tensor-data output))
           (batch-size 8)
           (elements-per-sample 4)
           (found-zeros nil))
      
      ;; Check each batch sample - at least one should be all zeros (dropped)
      (dotimes (b batch-size)
        (let ((sample-start (* b elements-per-sample))
              (sample-is-zero t))
          (dotimes (i elements-per-sample)
            (when (/= (row-major-aref data (+ sample-start i)) 0.0d0)
              (setf sample-is-zero nil)
              (return)))
          (when sample-is-zero
            (setf found-zeros t)
            (return))))
      
      ;; With seed 42, at least one sample should be dropped (all zeros)
      (is (not (null found-zeros))))))

(test stochastic-depth-training-has-scaled-values
  "Test stochastic depth scales kept samples by 1/(1-drop_prob)"
  (let* ((drop-prob 0.5d0)
         (sd (make-instance 'neural-tensor-residual:stochastic-depth
                           :drop-prob drop-prob))
         ;; Use 8 samples - with seed 123 this deterministically produces both kept and dropped
         (input (make-tensor (make-array '(8 4)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(8 4)))
         (found-scaled nil))
    
    ;; Set to training mode and use fixed seed for deterministic behavior
    (setf (neural-network::layer-training sd) t)
    (variational:set-random-seed 123)
    
    (let* ((output (forward sd input))
           (data (tensor-data output))
           (batch-size 8)
           (elements-per-sample 4))
      
      ;; Check each batch sample - at least one should be scaled (kept)
      (dotimes (b batch-size)
        (let ((sample-start (* b elements-per-sample))
              (sample-is-scaled t))
          (dotimes (i elements-per-sample)
            (let ((val (row-major-aref data (+ sample-start i))))
              ;; Scaled values should be approximately 2.0 (1.0 * scale where scale=2.0)
              (when (< val 1.5d0)
                (setf sample-is-scaled nil)
                (return))))
          (when sample-is-scaled
            (setf found-scaled t)
            (return))))
      
      ;; With seed 123, at least one sample should be kept and scaled
      (is (not (null found-scaled))))))

(test stochastic-depth-batch-independence
  "Test stochastic depth drops entire batch samples, not individual elements"
  (let* ((batch-size 8)
         (channels 16)
         (height 4)
         (width 4)
         (sd (make-instance 'neural-tensor-residual:stochastic-depth
                           :drop-prob 0.5))
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width))))
    
    ;; Set to training mode
    (setf (neural-network::layer-training sd) t)
    
    ;; Set seed for reproducibility
    (variational:set-random-seed 999)
    
    (let* ((output (forward sd input))
           (data (tensor-data output))
           (elements-per-sample (* channels height width)))
      
      ;; Check that for each batch element, either all zeros or all scaled
      (dotimes (b batch-size)
        (let ((first-val (row-major-aref data (* b elements-per-sample)))
              (all-same t))
          (loop for i from (* b elements-per-sample) 
                below (* (1+ b) elements-per-sample)
                do (unless (= (row-major-aref data i) first-val)
                     (setf all-same nil)
                     (return)))
          ;; Each batch element should have uniform values (all 0 or all scaled)
          (is (not (null all-same))))))))

(test stochastic-depth-high-drop-prob
  "Test stochastic depth with high drop probability"
  (let* ((sd (make-instance 'neural-tensor-residual:stochastic-depth
                           :drop-prob 0.9))
         (input (make-tensor (make-array '(100 8)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(100 8))))
    
    ;; Set to training mode and seed for determinism
    (setf (neural-network::layer-training sd) t)
    (variational:set-random-seed 500)
    
    (let* ((output (forward sd input))
           (data (tensor-data output))
           (zero-count 0)
           (nonzero-count 0))
      ;; Count zeros vs non-zeros
      (dotimes (j (array-total-size data))
        (if (= (row-major-aref data j) 0.0d0)
            (incf zero-count)
            (incf nonzero-count)))
      
      ;; With 0.9 drop prob and 100 samples, should see significantly more zeros
      (is (> zero-count nonzero-count)))))

(test stochastic-depth-in-residual-block
  "Test stochastic depth integrated in a residual block"
  (let* ((batch-size 2)
         (channels 32)
         (height 8)
         (width 8)
         
         ;; Create a simple residual path with stochastic depth
         (sd (make-instance 'neural-tensor-residual:stochastic-depth
                           :drop-prob 0.3))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width))))
    
    ;; Test in training mode
    (setf (neural-network::layer-training sd) t)
    (let ((output-train (forward sd input)))
      (is (equal (tensor-shape output-train) (tensor-shape input))))
    
    ;; Test in eval mode - shape should match
    (setf (neural-network::layer-training sd) nil)
    (let ((output-eval (forward sd input)))
      (is (equal (tensor-shape output-eval) (tensor-shape input))))))

(test stochastic-depth-gradient-tracking
  "Test stochastic depth preserves gradient tracking"
  (let* ((sd (make-instance 'neural-tensor-residual:stochastic-depth
                           :drop-prob 0.2))
         (input (make-tensor (make-array '(4 8)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(4 8)
                            :requires-grad t)))
    
    (setf (neural-network::layer-training sd) nil)
    
    (let ((output (forward sd input)))
      ;; Output should preserve requires-grad
      (is (neural-network::requires-grad output)))))

(test stochastic-depth-deterministic-with-seed
  "Test stochastic depth produces deterministic results with random seed"
  (let* ((sd (make-instance 'neural-tensor-residual:stochastic-depth
                           :drop-prob 0.5))
         (input (make-tensor (make-array '(8 4)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(8 4))))
    
    (setf (neural-network::layer-training sd) t)
    
    ;; Run with seed 42
    (variational:set-random-seed 42)
    (let* ((output1 (forward sd input))
           (data1 (tensor-data output1))
           (sum1 (loop for i below (array-total-size data1)
                      sum (row-major-aref data1 i))))
      
      ;; Run again with same seed
      (variational:set-random-seed 42)
      (let* ((output2 (forward sd input))
             (data2 (tensor-data output2))
             (sum2 (loop for i below (array-total-size data2)
                        sum (row-major-aref data2 i)))
             (all-equal t))
        
        ;; Check all values match exactly (not just sums)
        (dotimes (i (array-total-size data2))
          (unless (= (row-major-aref data1 i) (row-major-aref data2 i))
            (setf all-equal nil)
            (return)))
        
        ;; Results should be identical
        (is (not (null all-equal)))
        ;; Also check sums match (redundant but useful for debugging)
        (is (= sum1 sum2))))))
