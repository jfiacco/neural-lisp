;;;; tests/test-convolution.lisp - Tests for Convolutional Neural Networks

(in-package #:neural-lisp-tests)

(def-suite convolution-tests
  :description "Test suite for convolutional layers"
  :in neural-lisp-tests)

(in-suite convolution-tests)

;;;; Helper function for numerical comparisons
(defun assert-equal (expected actual &optional (tolerance 1d-6))
  "Assert two values are equal within tolerance"
  (unless (< (abs (- expected actual)) tolerance)
    (error "Expected ~A but got ~A (difference: ~A, tolerance: ~A)"
           expected actual (abs (- expected actual)) tolerance)))

;;;; ============================================================================
;;;; Conv1D Tests
;;;; ============================================================================

(test conv1d-basic
  "Test basic 1D convolution"
  (let* ((batch-size 2)
         (in-channels 3)
         (out-channels 5)
         (length 10)
         (kernel-size 3)
         
         ;; Create layer
         (conv (neural-tensor-convolution:conv1d in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding 1))
         
         ;; Create input
         (input (make-tensor (make-array (list batch-size in-channels length)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels length)
                            :requires-grad t))
         
         ;; Forward pass
         (output (forward conv input)))
    
    ;; Check output shape
    (is (equal (tensor-shape output) (list batch-size out-channels length)))
    
    ;; Check that output contains valid values
    (is (every (lambda (x) (and (numberp x) 
                                 (not (sb-ext:float-nan-p x)) 
                                 (not (sb-ext:float-infinity-p x))))
               (loop for i below (array-total-size (tensor-data output))
                     collect (row-major-aref (tensor-data output) i))))))

(test conv1d-stride
  "Test 1D convolution with stride"
  (let* ((batch-size 1)
         (in-channels 2)
         (out-channels 4)
         (length 10)
         (kernel-size 3)
         (stride 2)
         
         (conv (neural-tensor-convolution:conv1d in-channels out-channels kernel-size
                                                  :stride stride
                                                  :padding 0))
         
         (input (make-tensor (make-array (list batch-size in-channels length)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels length)))
         
         (output (forward conv input))
         
         ;; Expected output length
         (expected-length (neural-tensor-convolution:calculate-output-size 
                          length kernel-size stride 0 1)))
    
    ;; Check output shape
    (is (equal (tensor-shape output) (list batch-size out-channels expected-length)))
    (is (= expected-length 4))))

(test conv1d-padding
  "Test 1D convolution with padding"
  (let* ((batch-size 1)
         (in-channels 2)
         (out-channels 3)
         (length 8)
         (kernel-size 5)
         (padding 2)
         
         (conv (neural-tensor-convolution:conv1d in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding padding))
         
         (input (make-tensor (make-array (list batch-size in-channels length)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels length)))
         
         (output (forward conv input)))
    
    ;; With padding=2, kernel=5, stride=1, output length should equal input length
    (is (equal (tensor-shape output) (list batch-size out-channels length)))))

;;;; ============================================================================
;;;; Conv2D Tests
;;;; ============================================================================

(test conv2d-basic
  "Test basic 2D convolution"
  (let* ((batch-size 2)
         (in-channels 3)
         (out-channels 6)
         (height 8)
         (width 8)
         (kernel-size 3)
         
         (conv (neural-tensor-convolution:conv2d in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding 1))
         
         (input (make-tensor (make-array (list batch-size in-channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels height width)
                            :requires-grad t))
         
         (output (forward conv input)))
    
    ;; Check output shape (with padding=1, output size should match input)
    (is (equal (tensor-shape output) (list batch-size out-channels height width)))
    
    ;; Check that layer has parameters
    (is (not (null (layer-parameters conv))))
    (is (= (length (layer-parameters conv)) 2)))) ; kernel and bias

(test conv2d-stride-padding
  "Test 2D convolution with stride and padding"
  (let* ((batch-size 1)
         (in-channels 3)
         (out-channels 8)
         (height 28)
         (width 28)
         (kernel-size '(5 5))
         (stride '(2 2))
         (padding '(2 2))
         
         (conv (neural-tensor-convolution:conv2d in-channels out-channels kernel-size
                                                  :stride stride
                                                  :padding padding))
         
         (input (make-tensor (make-array (list batch-size in-channels height width)
                                        :element-type 'double-float
                                        :initial-element 0.5d0)
                            :shape (list batch-size in-channels height width)))
         
         (output (forward conv input))
         
         (expected-h (neural-tensor-convolution:calculate-output-size 
                     height (first kernel-size) (first stride) (first padding) 1))
         (expected-w (neural-tensor-convolution:calculate-output-size 
                     width (second kernel-size) (second stride) (second padding) 1)))
    
    (is (equal (tensor-shape output) (list batch-size out-channels expected-h expected-w)))))

(test conv2d-no-bias
  "Test 2D convolution without bias"
  (let* ((conv (neural-tensor-convolution:conv2d 3 6 3
                                                  :stride 1
                                                  :padding 1
                                                  :use-bias nil))
         
         (input (make-tensor (make-array '(1 3 8 8)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(1 3 8 8)))
         
         (output (forward conv input)))
    
    ;; Check that only kernel parameter exists (no bias)
    (is (= (length (layer-parameters conv)) 1))
    (is (equal (tensor-shape output) '(1 6 8 8)))))

;;;; ============================================================================
;;;; Conv3D Tests
;;;; ============================================================================

(test conv3d-basic
  "Test basic 3D convolution"
  (let* ((batch-size 1)
         (in-channels 2)
         (out-channels 4)
         (depth 8)
         (height 8)
         (width 8)
         (kernel-size 3)
         
         (conv (neural-tensor-convolution:conv3d in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding 1))
         
         (input (make-tensor (make-array (list batch-size in-channels depth height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels depth height width)))
         
         (output (forward conv input)))
    
    ;; Check output shape
    (is (equal (tensor-shape output) (list batch-size out-channels depth height width)))))

(test conv3d-asymmetric-kernel
  "Test 3D convolution with asymmetric kernel size"
  (let* ((batch-size 1)
         (in-channels 2)
         (out-channels 3)
         (depth 10)
         (height 8)
         (width 8)
         (kernel-size '(3 5 5))
         (padding '(1 2 2))
         
         (conv (neural-tensor-convolution:conv3d in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding padding))
         
         (input (make-tensor (make-array (list batch-size in-channels depth height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels depth height width)))
         
         (output (forward conv input)))
    
    ;; Check output shape (with proper padding, dimensions should be preserved)
    (is (equal (tensor-shape output) (list batch-size out-channels depth height width)))))

;;;; ============================================================================
;;;; ConvND Tests
;;;; ============================================================================

(test convnd-2d
  "Test N-D convolution with 2 dimensions"
  (let* ((batch-size 1)
         (in-channels 2)
         (out-channels 4)
         (h 6)
         (w 6)
         (kernel-size 3)
         
         (conv (neural-tensor-convolution:convnd 2 in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding 1))
         
         (input (make-tensor (make-array (list batch-size in-channels h w)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels h w)))
         
         (output (forward conv input)))
    
    (is (equal (tensor-shape output) (list batch-size out-channels h w)))))

(test convnd-4d
  "Test N-D convolution with 4 dimensions"
  (let* ((batch-size 1)
         (in-channels 2)
         (out-channels 3)
         (dims '(4 4 4 4))
         (kernel-size 3)
         
         (conv (neural-tensor-convolution:convnd 4 in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding 1))
         
         (input (make-tensor (make-array (append (list batch-size in-channels) dims)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (append (list batch-size in-channels) dims)))
         
         (output (forward conv input)))
    
    (is (equal (tensor-shape output) (append (list batch-size out-channels) dims)))))

(test convnd-3d-with-stride
  "Test N-D convolution (3D) with stride > 1"
  (let* ((batch-size 2)
         (in-channels 3)
         (out-channels 5)
         (dims '(8 8 8))
         (kernel-size 3)
         (stride 2)
         
         (conv (neural-tensor-convolution:convnd 3 in-channels out-channels kernel-size
                                                  :stride stride
                                                  :padding 0))
         
         (input (make-tensor (make-array (append (list batch-size in-channels) dims)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (append (list batch-size in-channels) dims)))
         
         (output (forward conv input)))
    
    ;; With stride=2, kernel=3, padding=0: (8 - 3)/2 + 1 = 3
    (is (equal (tensor-shape output) (list batch-size out-channels 3 3 3)))))

(test convnd-2d-with-dilation
  "Test N-D convolution (2D) with dilation"
  (let* ((batch-size 1)
         (in-channels 1)
         (out-channels 2)
         (h 10)
         (w 10)
         (kernel-size 3)
         (dilation 2)
         
         (conv (neural-tensor-convolution:convnd 2 in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding 0
                                                  :dilation dilation))
         
         (input (make-tensor (make-array (list batch-size in-channels h w)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels h w)))
         
         (output (forward conv input)))
    
    ;; With dilation=2, kernel=3: effective kernel = 2*(3-1)+1 = 5
    ;; Output size: (10 - 5)/1 + 1 = 6
    (is (equal (tensor-shape output) (list batch-size out-channels 6 6)))))

(test convnd-2d-asymmetric-params
  "Test N-D convolution with asymmetric kernel, stride, padding"
  (let* ((batch-size 1)
         (in-channels 2)
         (out-channels 3)
         (h 12)
         (w 16)
         
         (conv (neural-tensor-convolution:convnd 2 in-channels out-channels
                                                  '(3 5)  ; asymmetric kernel
                                                  :stride '(2 1)
                                                  :padding '(1 2)))
         
         (input (make-tensor (make-array (list batch-size in-channels h w)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels h w)))
         
         (output (forward conv input)))
    
    ;; H: (12 + 2*1 - 3)/2 + 1 = 6
    ;; W: (16 + 2*2 - 5)/1 + 1 = 16
    (is (equal (tensor-shape output) (list batch-size out-channels 6 16)))))

(test convnd-1d-sequence
  "Test N-D convolution for 1D sequence (like temporal/audio data)"
  (let* ((batch-size 4)
         (in-channels 8)
         (out-channels 16)
         (seq-length 100)
         (kernel-size 5)
         
         (conv (neural-tensor-convolution:convnd 1 in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding 2))
         
         (input (make-tensor (make-array (list batch-size in-channels seq-length)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels seq-length)))
         
         (output (forward conv input)))
    
    ;; Same length due to padding=2 (total padding = 4, kernel=5)
    (is (equal (tensor-shape output) (list batch-size out-channels seq-length)))))

(test convnd-5d-hyperdimensional
  "Test N-D convolution with 5 spatial dimensions"
  (let* ((batch-size 1)
         (in-channels 1)
         (out-channels 2)
         (dims '(3 3 3 3 3))  ; 5D spatial
         (kernel-size 3)
         
         (conv (neural-tensor-convolution:convnd 5 in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding 1))
         
         (input (make-tensor (make-array (append (list batch-size in-channels) dims)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (append (list batch-size in-channels) dims)))
         
         (output (forward conv input)))
    
    ;; All dimensions preserved with padding=1
    (is (equal (tensor-shape output) (append (list batch-size out-channels) dims)))))

(test convnd-2d-with-groups
  "Test N-D convolution with grouped convolution"
  (let* ((batch-size 2)
         (in-channels 4)
         (out-channels 8)
         (groups 2)
         (h 6)
         (w 6)
         (kernel-size 3)
         
         (conv (neural-tensor-convolution:convnd 2 in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding 1
                                                  :groups groups))
         
         (input (make-tensor (make-array (list batch-size in-channels h w)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels h w)))
         
         (output (forward conv input)))
    
    (is (equal (tensor-shape output) (list batch-size out-channels h w)))))

(test convnd-2d-no-padding-output-size
  "Test N-D convolution output size calculation without padding"
  (let* ((batch-size 1)
         (in-channels 3)
         (out-channels 6)
         (h 28)
         (w 28)
         (kernel-size 5)
         (stride 2)
         
         (conv (neural-tensor-convolution:convnd 2 in-channels out-channels kernel-size
                                                  :stride stride
                                                  :padding 0))
         
         (input (make-tensor (make-array (list batch-size in-channels h w)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels h w)))
         
         (output (forward conv input)))
    
    ;; (28 - 5)/2 + 1 = 12
    (is (equal (tensor-shape output) (list batch-size out-channels 12 12)))))

(test convnd-3d-numerical-correctness
  "Test N-D convolution numerical correctness with known values"
  (let* ((batch-size 1)
         (in-channels 1)
         (out-channels 1)
         (dims '(3 3 3))
         (kernel-size 2)
         
         (conv (neural-tensor-convolution:convnd 3 in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding 0
                                                  :use-bias nil))
         
         ;; Set kernel to all 1s for predictable output
         (kernel-data (tensor-data (neural-tensor-convolution:conv-kernel conv)))
         
         (input (make-tensor (make-array (append (list batch-size in-channels) dims)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (append (list batch-size in-channels) dims))))
    
    ;; Set all kernel weights to 1.0
    (dotimes (i (array-total-size kernel-data))
      (setf (row-major-aref kernel-data i) 1.0d0))
    
    (let ((output (forward conv input)))
      ;; Output should be 2x2x2 (from 3x3x3 with kernel 2x2x2)
      (is (equal (tensor-shape output) (list batch-size out-channels 2 2 2)))
      
      ;; Each output element is sum of 2^3=8 input elements, all 1.0
      ;; So each output should be 8.0
      (let ((output-data (tensor-data output)))
        (dotimes (i (array-total-size output-data))
          (is (< (abs (- (row-major-aref output-data i) 8.0d0)) 1e-6)))))))

(test convnd-edge-single-element
  "Test N-D convolution with single spatial element (edge case)"
  (let* ((batch-size 1)
         (in-channels 2)
         (out-channels 3)
         (dims '(1 1))  ; Single element
         (kernel-size 1)
         
         (conv (neural-tensor-convolution:convnd 2 in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding 0))
         
         (input (make-tensor (make-array (append (list batch-size in-channels) dims)
                                        :element-type 'double-float
                                        :initial-element 2.0d0)
                            :shape (append (list batch-size in-channels) dims)))
         
         (output (forward conv input)))
    
    (is (equal (tensor-shape output) (list batch-size out-channels 1 1)))
    (is (not (null output)))))

(test convnd-large-batch
  "Test N-D convolution with large batch size"
  (let* ((batch-size 32)
         (in-channels 3)
         (out-channels 8)
         (h 16)
         (w 16)
         (kernel-size 3)
         
         (conv (neural-tensor-convolution:convnd 2 in-channels out-channels kernel-size
                                                  :stride 1
                                                  :padding 1))
         
         (input (make-tensor (make-array (list batch-size in-channels h w)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels h w)))
         
         (output (forward conv input)))
    
    (is (equal (tensor-shape output) (list batch-size out-channels h w)))))

;;;; ============================================================================
;;;; Pooling Tests
;;;; ============================================================================

(test max-pool2d-basic
  "Test basic 2D max pooling"
  (let* ((batch-size 1)
         (channels 2)
         (height 8)
         (width 8)
         (kernel-size 2)
         
         (pool (neural-tensor-convolution:max-pool2d kernel-size :stride 2))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward pool input)))
    
    ;; Output should be half the size
    (is (equal (tensor-shape output) (list batch-size channels 4 4)))))

(test avg-pool2d-basic
  "Test basic 2D average pooling"
  (let* ((batch-size 1)
         (channels 3)
         (height 6)
         (width 6)
         (kernel-size 3)
         
         (pool (neural-tensor-convolution:avg-pool2d kernel-size :stride 3))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 2.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward pool input)))
    
    ;; Check output shape
    (is (equal (tensor-shape output) (list batch-size channels 2 2)))
    
    ;; Check that average is computed correctly (all values are 2.0)
    (is (every (lambda (x) (< (abs (- x 2.0d0)) 1e-6))
               (loop for i below (array-total-size (tensor-data output))
                     collect (row-major-aref (tensor-data output) i))))))

(test max-pool2d-with-padding
  "Test 2D max pooling with padding"
  (let* ((pool (neural-tensor-convolution:max-pool2d 2 :stride 2 :padding 1))
         
         (input (make-tensor (make-array '(1 1 3 3)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(1 1 3 3)))
         
         (output (forward pool input)))
    
    ;; With padding, output size should be larger
    (is (equal (tensor-shape output) '(1 1 2 2)))))

;;;; ============================================================================
;;;; Global Pooling Tests
;;;; ============================================================================

(test global-avg-pool1d-basic
  "Test global average pooling for 1D data"
  (let* ((batch-size 2)
         (channels 3)
         (length 10)
         
         (pool (make-instance 'neural-tensor-convolution:global-avg-pool1d))
         
         (input (make-tensor (make-array (list batch-size channels length)
                                        :element-type 'double-float
                                        :initial-element 2.0d0)
                            :shape (list batch-size channels length)))
         
         (output (forward pool input)))
    
    ;; Output should be (batch, channels, 1)
    (is (equal (tensor-shape output) (list batch-size channels 1)))
    
    ;; Average of all 2.0s should be 2.0
    (let ((output-data (tensor-data output)))
      (assert-equal 2.0d0 (aref output-data 0 0 0)))))

(test global-avg-pool2d-basic
  "Test global average pooling for 2D data"
  (let* ((batch-size 2)
         (channels 4)
         (height 8)
         (width 8)
         
         (pool (make-instance 'neural-tensor-convolution:global-avg-pool2d))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 3.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward pool input)))
    
    ;; Output should be (batch, channels, 1, 1)
    (is (equal (tensor-shape output) (list batch-size channels 1 1)))
    
    ;; Average of all 3.0s should be 3.0
    (let ((output-data (tensor-data output)))
      (assert-equal 3.0d0 (aref output-data 0 0 0 0)))))

(test global-avg-pool3d-basic
  "Test global average pooling for 3D data"
  (let* ((batch-size 1)
         (channels 2)
         (depth 4)
         (height 4)
         (width 4)
         
         (pool (make-instance 'neural-tensor-convolution:global-avg-pool3d))
         
         (input (make-tensor (make-array (list batch-size channels depth height width)
                                        :element-type 'double-float
                                        :initial-element 5.0d0)
                            :shape (list batch-size channels depth height width)))
         
         (output (forward pool input)))
    
    ;; Output should be (batch, channels, 1, 1, 1)
    (is (equal (tensor-shape output) (list batch-size channels 1 1 1)))
    
    ;; Average of all 5.0s should be 5.0
    (let ((output-data (tensor-data output)))
      (assert-equal 5.0d0 (aref output-data 0 0 0 0 0)))))

(test global-max-pool1d-basic
  "Test global max pooling for 1D data"
  (let* ((batch-size 1)
         (channels 2)
         (length 10)
         
         (pool (make-instance 'neural-tensor-convolution:global-max-pool1d))
         
         (input-data (make-array (list batch-size channels length)
                                :element-type 'double-float
                                :initial-element 1.0d0)))
    
    ;; Set one value higher BEFORE creating tensor
    (setf (aref input-data 0 0 5) 10.0d0)
    
    (let* ((input (make-tensor input-data :shape (list batch-size channels length)))
           (output (forward pool input)))
      ;; Output should be (batch, channels, 1)
      (is (equal (tensor-shape output) (list batch-size channels 1)))
      
      ;; Max should be 10.0
      (let ((output-data (tensor-data output)))
        (assert-equal 10.0d0 (aref output-data 0 0 0))))))

(test global-max-pool2d-basic
  "Test global max pooling for 2D data"
  (let* ((batch-size 1)
         (channels 3)
         (height 6)
         (width 6)
         
         (pool (make-instance 'neural-tensor-convolution:global-max-pool2d))
         
         (input-data (make-array (list batch-size channels height width)
                                :element-type 'double-float
                                :initial-element 1.0d0)))
    
    ;; Set one value higher BEFORE creating tensor
    (setf (aref input-data 0 1 3 2) 15.0d0)
    
    (let* ((input (make-tensor input-data :shape (list batch-size channels height width)))
           (output (forward pool input)))
      ;; Output should be (batch, channels, 1, 1)
      (is (equal (tensor-shape output) (list batch-size channels 1 1)))
      
      ;; Max in channel 1 should be 15.0
      (let ((output-data (tensor-data output)))
        (assert-equal 15.0d0 (aref output-data 0 1 0 0))))))

(test global-max-pool3d-basic
  "Test global max pooling for 3D data"
  (let* ((batch-size 1)
         (channels 2)
         (depth 3)
         (height 3)
         (width 3)
         
         (pool (make-instance 'neural-tensor-convolution:global-max-pool3d))
         
         (input-data (make-array (list batch-size channels depth height width)
                                :element-type 'double-float
                                :initial-element 2.0d0)))
    
    ;; Set one value higher BEFORE creating tensor
    (setf (aref input-data 0 0 1 1 1) 20.0d0)
    
    (let* ((input (make-tensor input-data :shape (list batch-size channels depth height width)))
           (output (forward pool input)))
      ;; Output should be (batch, channels, 1, 1, 1)
      (is (equal (tensor-shape output) (list batch-size channels 1 1 1)))
      
      ;; Max in channel 0 should be 20.0
      (let ((output-data (tensor-data output)))
        (assert-equal 20.0d0 (aref output-data 0 0 0 0 0))))))

(test global-avg-pool2d-numerical-correctness
  "Test GAP2D computes correct average"
  (let* ((batch-size 1)
         (channels 1)
         (height 4)
         (width 4)
         
         (pool (make-instance 'neural-tensor-convolution:global-avg-pool2d))
         
         (input-data (make-array (list batch-size channels height width)
                                :element-type 'double-float)))
    
    ;; Fill with values 1 to 16 BEFORE creating tensor
    (let ((val 1.0d0))
      (loop for h from 0 below height do
        (loop for w from 0 below width do
          (setf (aref input-data 0 0 h w) val)
          (incf val))))
    
    (let* ((input (make-tensor input-data :shape (list batch-size channels height width)))
           (output (forward pool input)))
      ;; Average of 1..16 is 8.5
      (let ((output-data (tensor-data output)))
        (assert-equal 8.5d0 (aref output-data 0 0 0 0))))))

(test global-pooling-variable-size
  "Test that global pooling works with different input sizes"
  (let ((pool (make-instance 'neural-tensor-convolution:global-avg-pool2d)))
    
    ;; Test with 4x4 input
    (let* ((input1 (make-tensor (make-array '(1 2 4 4)
                                           :element-type 'double-float
                                           :initial-element 1.0d0)
                               :shape '(1 2 4 4)))
           (output1 (forward pool input1)))
      (is (equal (tensor-shape output1) '(1 2 1 1))))
    
    ;; Test with 10x10 input (different size)
    (let* ((input2 (make-tensor (make-array '(1 2 10 10)
                                           :element-type 'double-float
                                           :initial-element 1.0d0)
                               :shape '(1 2 10 10)))
           (output2 (forward pool input2)))
      (is (equal (tensor-shape output2) '(1 2 1 1))))))

;;;; ============================================================================
;;;; Spatial Pyramid Pooling Tests
;;;; ============================================================================

(test spp-basic-single-level
  "Test SPP with single pyramid level"
  (let* ((batch-size 1)
         (channels 2)
         (height 8)
         (width 8)
         
         (spp (make-instance 'neural-tensor-convolution:spatial-pyramid-pool2d
                            :pyramid-levels '(1)))  ; Just 1x1 grid
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward spp input)))
    
    ;; 1x1 grid = 1 bin per channel, so output is (batch, channels * 1)
    (is (equal (tensor-shape output) (list batch-size (* channels 1))))))

(test spp-multi-level
  "Test SPP with multiple pyramid levels"
  (let* ((batch-size 1)
         (channels 3)
         (height 16)
         (width 16)
         
         (spp (make-instance 'neural-tensor-convolution:spatial-pyramid-pool2d
                            :pyramid-levels '(1 2 4)))  ; 1x1, 2x2, 4x4 grids
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward spp input)))
    
    ;; Total bins: 1 + 4 + 16 = 21 per channel
    (is (equal (tensor-shape output) (list batch-size (* channels 21))))))

(test spp-variable-input-size
  "Test that SPP produces fixed output size for variable input sizes"
  (let ((spp (make-instance 'neural-tensor-convolution:spatial-pyramid-pool2d
                           :pyramid-levels '(1 2 4))))
    
    ;; Test with 8x8 input
    (let* ((input1 (make-tensor (make-array '(1 2 8 8)
                                           :element-type 'double-float
                                           :initial-element 1.0d0)
                               :shape '(1 2 8 8)))
           (output1 (forward spp input1)))
      ;; 21 bins per channel, 2 channels = 42
      (is (equal (tensor-shape output1) '(1 42))))
    
    ;; Test with 16x16 input (different size)
    (let* ((input2 (make-tensor (make-array '(1 2 16 16)
                                           :element-type 'double-float
                                           :initial-element 1.0d0)
                               :shape '(1 2 16 16)))
           (output2 (forward spp input2)))
      ;; Same output size!
      (is (equal (tensor-shape output2) '(1 42))))
    
    ;; Test with 32x24 input (non-square, different size)
    (let* ((input3 (make-tensor (make-array '(1 2 32 24)
                                           :element-type 'double-float
                                           :initial-element 1.0d0)
                               :shape '(1 2 32 24)))
           (output3 (forward spp input3)))
      ;; Still same output size!
      (is (equal (tensor-shape output3) '(1 42))))))

(test spp-with-max-pooling
  "Test SPP with max pooling"
  (let* ((batch-size 1)
         (channels 1)
         (height 4)
         (width 4)
         
         (spp (make-instance 'neural-tensor-convolution:spatial-pyramid-pool2d
                            :pyramid-levels '(2)
                            :pool-type :max))
         
         (input-data (make-array (list batch-size channels height width)
                                :element-type 'double-float
                                :initial-element 1.0d0)))
    
    ;; Set specific max values in each quadrant BEFORE creating tensor
    (setf (aref input-data 0 0 0 0) 10.0d0)  ; Top-left
    (setf (aref input-data 0 0 0 3) 20.0d0)  ; Top-right
    (setf (aref input-data 0 0 3 0) 30.0d0)  ; Bottom-left
    (setf (aref input-data 0 0 3 3) 40.0d0)  ; Bottom-right
    
    (let* ((input (make-tensor input-data :shape (list batch-size channels height width)))
           (output (forward spp input)))
      ;; 2x2 grid = 4 bins
      (is (equal (tensor-shape output) '(1 4)))
      
      ;; Check that each bin captured its max
      (let ((output-data (tensor-data output)))
        (assert-equal 10.0d0 (aref output-data 0 0))
        (assert-equal 20.0d0 (aref output-data 0 1))
        (assert-equal 30.0d0 (aref output-data 0 2))
        (assert-equal 40.0d0 (aref output-data 0 3))))))

(test spp-with-avg-pooling
  "Test SPP with average pooling"
  (let* ((batch-size 1)
         (channels 1)
         (height 4)
         (width 4)
         
         (spp (make-instance 'neural-tensor-convolution:spatial-pyramid-pool2d
                            :pyramid-levels '(1)
                            :pool-type :avg))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 2.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward spp input)))
    
    ;; Average of all 2.0s should be 2.0
    (let ((output-data (tensor-data output)))
      (assert-equal 2.0d0 (aref output-data 0 0)))))

(test spp-batch-processing
  "Test SPP with batch size > 1"
  (let* ((batch-size 4)
         (channels 2)
         (height 8)
         (width 8)
         
         (spp (make-instance 'neural-tensor-convolution:spatial-pyramid-pool2d
                            :pyramid-levels '(1 2)))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward spp input)))
    
    ;; 1 + 4 = 5 bins per channel, 2 channels = 10 features per batch
    (is (equal (tensor-shape output) (list batch-size 10)))))

(test spp-large-pyramid
  "Test SPP with large pyramid levels"
  (let* ((batch-size 1)
         (channels 1)
         (height 32)
         (width 32)
         
         (spp (make-instance 'neural-tensor-convolution:spatial-pyramid-pool2d
                            :pyramid-levels '(1 2 4 8)))  ; 4-level pyramid
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward spp input)))
    
    ;; Total bins: 1 + 4 + 16 + 64 = 85
    (is (equal (tensor-shape output) (list batch-size 85)))))

;;;; ============================================================================
;;;; Transposed Convolution Tests
;;;; ============================================================================

(test conv-transpose2d-basic
  "Test basic 2D transposed convolution"
  (let* ((batch-size 1)
         (in-channels 4)
         (out-channels 2)
         (height 4)
         (width 4)
         (kernel-size 4)
         (stride 2)
         (padding 1)
         
         (conv-transpose (neural-tensor-convolution:conv-transpose2d 
                         in-channels out-channels kernel-size
                         :stride stride
                         :padding padding))
         
         (input (make-tensor (make-array (list batch-size in-channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels height width)))
         
         (output (forward conv-transpose input)))
    
    ;; Transposed convolution should upsample
    (is (= (third (tensor-shape output)) 8))
    (is (= (fourth (tensor-shape output)) 8))))

(test conv-transpose2d-upsampling
  "Test that transposed convolution upsamples correctly"
  (let* ((conv-transpose (neural-tensor-convolution:conv-transpose2d 
                         2 2 3
                         :stride 2
                         :padding 1))
         
         (input (make-tensor (make-array '(1 2 4 4)
                                        :element-type 'double-float
                                        :initial-element 0.5d0)
                            :shape '(1 2 4 4)))
         
         (output (forward conv-transpose input)))
    
    ;; Should roughly double the spatial dimensions
    (is (>= (third (tensor-shape output)) 6))
    (is (>= (fourth (tensor-shape output)) 6))))

;;;; ============================================================================
;;;; Separable Convolution Tests
;;;; ============================================================================

(test separable-conv2d-basic
  "Test separable 2D convolution"
  (let* ((batch-size 1)
         (in-channels 3)
         (out-channels 6)
         (height 8)
         (width 8)
         (kernel-size 3)
         
         (sep-conv (neural-tensor-convolution:separable-conv2d 
                   in-channels out-channels kernel-size
                   :stride 1
                   :padding 1))
         
         (input (make-tensor (make-array (list batch-size in-channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels height width)))
         
         (output (forward sep-conv input)))
    
    ;; Check output shape
    (is (equal (tensor-shape output) (list batch-size out-channels height width)))
    
    ;; Check that separable conv has both depthwise and pointwise parameters
    (is (not (null (layer-parameters sep-conv))))))

(test depthwise-conv2d-basic
  "Test depthwise 2D convolution"
  (let* ((batch-size 1)
         (channels 4)
         (height 6)
         (width 6)
         (kernel-size 3)
         
         (dw-conv (neural-tensor-convolution:depthwise-conv2d 
                  channels kernel-size
                  :stride 1
                  :padding 1))
         
         (input (make-tensor (make-array (list batch-size channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size channels height width)))
         
         (output (forward dw-conv input)))
    
    ;; Depthwise convolution preserves number of channels
    (is (equal (tensor-shape output) (list batch-size channels height width)))))

;;;; ============================================================================
;;;; Dilated Convolution Tests
;;;; ============================================================================

(test dilated-conv2d-basic
  "Test dilated 2D convolution"
  (let* ((batch-size 1)
         (in-channels 3)
         (out-channels 5)
         (height 10)
         (width 10)
         (kernel-size 3)
         (dilation 2)
         
         (conv (neural-tensor-convolution:dilated-conv2d 
               in-channels out-channels kernel-size dilation
               :stride 1
               :padding 2))
         
         (input (make-tensor (make-array (list batch-size in-channels height width)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (list batch-size in-channels height width)))
         
         (output (forward conv input)))
    
    ;; With proper padding, output size should match input
    (is (equal (tensor-shape output) (list batch-size out-channels height width)))))

;;;; ============================================================================
;;;; Utility Function Tests
;;;; ============================================================================

(test calculate-output-size
  "Test output size calculation"
  (is (= (neural-tensor-convolution:calculate-output-size 10 3 1 1 1) 10))
  (is (= (neural-tensor-convolution:calculate-output-size 10 3 2 0 1) 4))
  (is (= (neural-tensor-convolution:calculate-output-size 28 5 1 2 1) 28))
  (is (= (neural-tensor-convolution:calculate-output-size 32 3 2 1 1) 16)))

(test pad-tensor-basic
  "Test tensor padding"
  (let* ((tensor (make-tensor (make-array '(2 2)
                                         :element-type 'double-float
                                         :initial-element 1.0d0)
                             :shape '(2 2)))
         
         (padded (neural-tensor-convolution:pad-tensor tensor '((1 1) (1 1)) 0.0d0)))
    
    ;; Check padded shape
    (is (equal (tensor-shape padded) '(4 4)))
    
    ;; Check that center values are preserved
    (is (= (aref (tensor-data padded) 1 1) 1.0d0))
    
    ;; Check that padding is zero
    (is (= (aref (tensor-data padded) 0 0) 0.0d0))))

;;;; ============================================================================
;;;; Integration Tests
;;;; ============================================================================

(test conv-layers-sequential
  "Test sequential composition of convolutional layers"
  (let* ((conv1 (neural-tensor-convolution:conv2d 3 6 3 :stride 1 :padding 1))
         (pool1 (neural-tensor-convolution:max-pool2d 2 :stride 2))
         (conv2 (neural-tensor-convolution:conv2d 6 12 3 :stride 1 :padding 1))
         (pool2 (neural-tensor-convolution:max-pool2d 2 :stride 2))
         
         (input (make-tensor (make-array '(1 3 16 16)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(1 3 16 16)))
         
         (x1 (forward conv1 input))
         (x2 (forward pool1 x1))
         (x3 (forward conv2 x2))
         (x4 (forward pool2 x3)))
    
    ;; Check progressive downsampling
    (is (equal (tensor-shape x1) '(1 6 16 16)))
    (is (equal (tensor-shape x2) '(1 6 8 8)))
    (is (equal (tensor-shape x3) '(1 12 8 8)))
    (is (equal (tensor-shape x4) '(1 12 4 4)))))

(test conv-with-activation
  "Test convolution with activation function"
  (let* ((conv (neural-tensor-convolution:conv2d 3 6 3 :stride 1 :padding 1))
         
         (input (make-tensor (make-array '(1 3 8 8)
                                        :element-type 'double-float
                                        :initial-element 0.5d0)
                            :shape '(1 3 8 8)))
         
         (conv-out (forward conv input))
         (activated (relu conv-out)))
    
    ;; Check that activation was applied
    (is (equal (tensor-shape activated) (tensor-shape conv-out)))
    
    ;; All values should be non-negative after ReLU
    (is (every (lambda (x) (>= x 0.0d0))
               (loop for i below (array-total-size (tensor-data activated))
                     collect (row-major-aref (tensor-data activated) i))))))

;;;; ============================================================================
;;;; Performance and Edge Cases
;;;; ============================================================================

(test conv2d-single-pixel
  "Test convolution on single-pixel input"
  (let* ((conv (neural-tensor-convolution:conv2d 3 6 1 :stride 1 :padding 0))
         
         (input (make-tensor (make-array '(1 3 1 1)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(1 3 1 1)))
         
         (output (forward conv input)))
    
    (is (equal (tensor-shape output) '(1 6 1 1)))))

(test conv2d-large-kernel
  "Test convolution with large kernel"
  (let* ((conv (neural-tensor-convolution:conv2d 2 4 7 :stride 1 :padding 3))
         
         (input (make-tensor (make-array '(1 2 8 8)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape '(1 2 8 8)))
         
         (output (forward conv input)))
    
    (is (equal (tensor-shape output) '(1 4 8 8)))))
