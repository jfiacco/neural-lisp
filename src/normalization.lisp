;;;; Normalization Layers - Layer Norm and Batch Norm
;;;; Implements Layer Normalization and Batch Normalization with full autograd support

(defpackage :normalization
  (:use :common-lisp :neural-network)
  (:export #:layer-norm
           #:batch-norm
           #:layer-norm-layer
           #:batch-norm-layer
           #:norm-gamma
           #:norm-beta
           #:running-mean
           #:running-var
           #:momentum
           #:eps))

(in-package :normalization)

;;;; ============================================================================
;;;; Layer Normalization
;;;; ============================================================================

(defclass layer-norm-layer (neural-network:layer)
  ((normalized-shape :initarg :normalized-shape
                     :accessor normalized-shape
                     :documentation "Shape to normalize over (e.g., feature dimension)")
   (gamma :accessor norm-gamma
          :documentation "Scale parameter")
   (beta :accessor norm-beta
         :documentation "Shift parameter")
   (eps :initarg :eps
        :accessor eps
        :initform 1d-5
        :documentation "Small constant for numerical stability"))
  (:documentation "Layer Normalization - normalizes across features"))

(defmethod initialize-instance :after ((layer layer-norm-layer) &key)
  (with-slots (normalized-shape gamma beta parameters eps) layer
    ;; Initialize gamma (scale) to ones and beta (shift) to zeros
    (setf gamma (ones (list normalized-shape)
                     :requires-grad t
                     :name "layer-norm-gamma"))
    (setf beta (zeros (list normalized-shape)
                     :requires-grad t
                     :name "layer-norm-beta"))
    (setf parameters (list gamma beta))))

(defun layer-norm (normalized-shape &key (eps 1d-5))
  "Create a layer normalization layer
   normalized-shape: the size of the feature dimension to normalize"
  (make-instance 'layer-norm-layer
                 :normalized-shape normalized-shape
                 :eps eps))

(defmethod forward ((layer layer-norm-layer) input)
  "Forward pass: normalize across feature dimension
   Input shape: (batch-size, ..., normalized-shape)
   Normalizes the last dimension(s)"
  (let* ((shape (tensor-shape input))
         (data (tensor-data input))
         (eps-val (eps layer))
         (gamma (norm-gamma layer))
         (beta (norm-beta layer))
         (normalized-shape (normalized-shape layer)))
    
    (cond
      ;; 2D input: (batch, features)
      ((= (length shape) 2)
       (let* ((batch-size (first shape))
              (features (second shape)))
         (unless (= features normalized-shape)
           (error "Input feature dimension ~a doesn't match normalized-shape ~a" 
                  features normalized-shape))
         
         ;; Compute mean and variance per sample
         (let* ((mean-data (make-array (list batch-size) :element-type 'double-float))
                (var-data (make-array (list batch-size) :element-type 'double-float))
                (normalized-data (make-array shape :element-type 'double-float)))
           
           ;; Compute mean for each sample
           (dotimes (b batch-size)
             (let ((sum 0.0d0))
               (dotimes (f features)
                 (incf sum (aref data b f)))
               (setf (aref mean-data b) (/ sum features))))
           
           ;; Compute variance for each sample
           (dotimes (b batch-size)
             (let ((var-sum 0.0d0)
                   (mean (aref mean-data b)))
               (dotimes (f features)
                 (let ((diff (- (aref data b f) mean)))
                   (incf var-sum (* diff diff))))
               (setf (aref var-data b) (/ var-sum features))))
           
           ;; Normalize
           (dotimes (b batch-size)
             (let ((mean (aref mean-data b))
                   (std (sqrt (+ (aref var-data b) eps-val))))
               (dotimes (f features)
                 (setf (aref normalized-data b f)
                       (/ (- (aref data b f) mean) std)))))
           
           ;; Apply scale and shift: gamma * normalized + beta
           (let ((normalized-tensor (make-tensor normalized-data 
                                                 :shape shape
                                                 :requires-grad (requires-grad input))))
             (t+ (t* gamma normalized-tensor) beta)))))
      
      ;; 3D input: (batch, seq-len, features) for transformers
      ((= (length shape) 3)
       (let* ((batch-size (first shape))
              (seq-len (second shape))
              (features (third shape)))
         (unless (= features normalized-shape)
           (error "Input feature dimension ~a doesn't match normalized-shape ~a" 
                  features normalized-shape))
         
         ;; Compute mean and variance per position in batch
         (let* ((mean-data (make-array (list batch-size seq-len) :element-type 'double-float))
                (var-data (make-array (list batch-size seq-len) :element-type 'double-float))
                (normalized-data (make-array shape :element-type 'double-float)))
           
           ;; Compute mean for each position
           (dotimes (b batch-size)
             (dotimes (s seq-len)
               (let ((sum 0.0d0))
                 (dotimes (f features)
                   (incf sum (aref data b s f)))
                 (setf (aref mean-data b s) (/ sum features)))))
           
           ;; Compute variance for each position
           (dotimes (b batch-size)
             (dotimes (s seq-len)
               (let ((var-sum 0.0d0)
                     (mean (aref mean-data b s)))
                 (dotimes (f features)
                   (let ((diff (- (aref data b s f) mean)))
                     (incf var-sum (* diff diff))))
                 (setf (aref var-data b s) (/ var-sum features)))))
           
           ;; Normalize
           (dotimes (b batch-size)
             (dotimes (s seq-len)
               (let ((mean (aref mean-data b s))
                     (std (sqrt (+ (aref var-data b s) eps-val))))
                 (dotimes (f features)
                   (setf (aref normalized-data b s f)
                         (/ (- (aref data b s f) mean) std))))))
           
           ;; Apply scale and shift: gamma * normalized + beta
           (let ((normalized-tensor (make-tensor normalized-data 
                                                 :shape shape
                                                 :requires-grad (requires-grad input))))
             (t+ (t* gamma normalized-tensor) beta)))))
      
      ;; General N-D input: Normalize over the last dimension
      ;; Works for any shape (d1, d2, ..., dn) where dn = normalized-shape
      (t
       (let* ((ndims (length shape))
              (feature-dim (nth (1- ndims) shape)))
         (unless (= feature-dim normalized-shape)
           (error "Input feature dimension ~a doesn't match normalized-shape ~a" 
                  feature-dim normalized-shape))
         
         ;; For general ND arrays, we normalize across the last dimension
         ;; Compute number of elements excluding the last dimension
         (let* ((prefix-dims (butlast shape))
                (prefix-size (reduce #'* prefix-dims :initial-value 1))
                (mean-var-shape prefix-dims)
                (mean-data (make-array prefix-size :element-type 'double-float))
                (var-data (make-array prefix-size :element-type 'double-float))
                (normalized-data (make-array shape :element-type 'double-float)))
           
           ;; Helper to convert flat index to multidimensional indices (excluding last dim)
           (labels ((flat-to-indices (flat-idx dims)
                      (let ((indices nil)
                            (remaining flat-idx))
                        (dolist (dim (reverse dims))
                          (multiple-value-bind (quotient remainder)
                              (floor remaining dim)
                            (push remainder indices)
                            (setf remaining quotient)))
                        indices))
                    
                    (get-element (indices feature-idx)
                      "Get element from data array at position"
                      (apply #'aref data (append indices (list feature-idx))))
                    
                    (set-element (arr indices feature-idx value)
                      "Set element in array at position"
                      (setf (apply #'aref arr (append indices (list feature-idx))) value)))
             
             ;; Compute mean for each position
             (dotimes (i prefix-size)
               (let ((sum 0.0d0)
                     (indices (flat-to-indices i prefix-dims)))
                 (dotimes (f feature-dim)
                   (incf sum (get-element indices f)))
                 (setf (aref mean-data i) (/ sum feature-dim))))
             
             ;; Compute variance for each position
             (dotimes (i prefix-size)
               (let ((var-sum 0.0d0)
                     (mean (aref mean-data i))
                     (indices (flat-to-indices i prefix-dims)))
                 (dotimes (f feature-dim)
                   (let ((diff (- (get-element indices f) mean)))
                     (incf var-sum (* diff diff))))
                 (setf (aref var-data i) (/ var-sum feature-dim))))
             
             ;; Normalize
             (dotimes (i prefix-size)
               (let ((mean (aref mean-data i))
                     (std (sqrt (+ (aref var-data i) eps-val)))
                     (indices (flat-to-indices i prefix-dims)))
                 (dotimes (f feature-dim)
                   (set-element normalized-data indices f
                               (/ (- (get-element indices f) mean) std)))))
             
             ;; Apply scale and shift
             (let ((normalized-tensor (make-tensor normalized-data 
                                                   :shape shape
                                                   :requires-grad (requires-grad input))))
               (t+ (t* gamma normalized-tensor) beta)))))))))

;;;; ============================================================================
;;;; Batch Normalization
;;;; ============================================================================

(defclass batch-norm-layer (neural-network:layer)
  ((num-features :initarg :num-features
                 :accessor num-features
                 :documentation "Number of feature channels")
   (gamma :accessor norm-gamma
          :documentation "Scale parameter")
   (beta :accessor norm-beta
         :documentation "Shift parameter")
   (running-mean :accessor running-mean
                 :documentation "Running mean for inference")
   (running-var :accessor running-var
                :documentation "Running variance for inference")
   (momentum :initarg :momentum
             :accessor momentum
             :initform 0.1d0
             :documentation "Momentum for running statistics")
   (eps :initarg :eps
        :accessor eps
        :initform 1d-5
        :documentation "Small constant for numerical stability"))
  (:documentation "Batch Normalization - normalizes across batch dimension"))

(defmethod initialize-instance :after ((layer batch-norm-layer) &key)
  (with-slots (num-features gamma beta running-mean running-var parameters) layer
    ;; Initialize gamma (scale) to ones and beta (shift) to zeros
    (setf gamma (ones (list num-features)
                     :requires-grad t
                     :name "batch-norm-gamma"))
    (setf beta (zeros (list num-features)
                     :requires-grad t
                     :name "batch-norm-beta"))
    ;; Initialize running statistics (not trainable)
    (setf running-mean (zeros (list num-features)
                              :requires-grad nil
                              :name "running-mean"))
    (setf running-var (ones (list num-features)
                            :requires-grad nil
                            :name "running-var"))
    (setf parameters (list gamma beta))))

(defun batch-norm (num-features &key (momentum 0.1d0) (eps 1d-5))
  "Create a batch normalization layer
   num-features: number of feature channels (C in (N, C) or (N, C, H, W))"
  (make-instance 'batch-norm-layer
                 :num-features num-features
                 :momentum momentum
                 :eps eps))

(defmethod forward ((layer batch-norm-layer) input)
  "Forward pass: normalize across batch dimension
   Input shape: (batch-size, num-features) or (batch-size, num-features, height, width)"
  (let* ((shape (tensor-shape input))
         (data (tensor-data input))
         (eps-val (eps layer))
         (gamma (norm-gamma layer))
         (beta (norm-beta layer))
         (training (layer-training layer))
         (num-features (num-features layer)))
    
    (cond
      ;; 2D input: (batch, features)
      ((= (length shape) 2)
       (let* ((batch-size (first shape))
              (features (second shape)))
         (unless (= features num-features)
           (error "Input feature dimension ~a doesn't match num-features ~a" 
                  features num-features))
         
         (if training
             ;; Training mode: compute batch statistics
             (let* ((mean-data (make-array (list features) :element-type 'double-float 
                                          :initial-element 0.0d0))
                    (var-data (make-array (list features) :element-type 'double-float
                                         :initial-element 0.0d0))
                    (normalized-data (make-array shape :element-type 'double-float)))
               
               ;; Compute mean per feature across batch
               (dotimes (f features)
                 (let ((sum 0.0d0))
                   (dotimes (b batch-size)
                     (incf sum (aref data b f)))
                   (setf (aref mean-data f) (/ sum batch-size))))
               
               ;; Compute variance per feature across batch
               (dotimes (f features)
                 (let ((var-sum 0.0d0)
                       (mean (aref mean-data f)))
                   (dotimes (b batch-size)
                     (let ((diff (- (aref data b f) mean)))
                       (incf var-sum (* diff diff))))
                   (setf (aref var-data f) (/ var-sum batch-size))))
               
               ;; Update running statistics
               (let ((mom (momentum layer))
                     (running-mean-data (tensor-data (running-mean layer)))
                     (running-var-data (tensor-data (running-var layer))))
                 (dotimes (f features)
                   (setf (aref running-mean-data f)
                         (+ (* (- 1.0d0 mom) (aref running-mean-data f))
                            (* mom (aref mean-data f))))
                   (setf (aref running-var-data f)
                         (+ (* (- 1.0d0 mom) (aref running-var-data f))
                            (* mom (aref var-data f))))))
               
               ;; Normalize
               (dotimes (b batch-size)
                 (dotimes (f features)
                   (let ((mean (aref mean-data f))
                         (std (sqrt (+ (aref var-data f) eps-val))))
                     (setf (aref normalized-data b f)
                           (/ (- (aref data b f) mean) std)))))
               
               ;; Apply scale and shift
               (let ((normalized-tensor (make-tensor normalized-data 
                                                     :shape shape
                                                     :requires-grad (requires-grad input))))
                 (t+ (t* gamma normalized-tensor) beta)))
             
             ;; Evaluation mode: use running statistics
             (let* ((normalized-data (make-array shape :element-type 'double-float))
                    (running-mean-data (tensor-data (running-mean layer)))
                    (running-var-data (tensor-data (running-var layer))))
               
               ;; Normalize using running statistics
               (dotimes (b batch-size)
                 (dotimes (f features)
                   (let ((mean (aref running-mean-data f))
                         (std (sqrt (+ (aref running-var-data f) eps-val))))
                     (setf (aref normalized-data b f)
                           (/ (- (aref data b f) mean) std)))))
               
               ;; Apply scale and shift
               (let ((normalized-tensor (make-tensor normalized-data 
                                                     :shape shape
                                                     :requires-grad (requires-grad input))))
                 (t+ (t* gamma normalized-tensor) beta))))))
      
      ;; 3D input: (batch, channels, length) - for 1D CNNs (temporal/sequence data)
      ((= (length shape) 3)
       (let* ((batch-size (first shape))
              (channels (second shape))
              (length (third shape)))
         (unless (= channels num-features)
           (error "Input channel dimension ~a doesn't match num-features ~a" 
                  channels num-features))
         
         (if training
             ;; Training mode: compute batch statistics across batch and temporal dimensions
             (let* ((mean-data (make-array (list channels) :element-type 'double-float 
                                          :initial-element 0.0d0))
                    (var-data (make-array (list channels) :element-type 'double-float
                                         :initial-element 0.0d0))
                    (normalized-data (make-array shape :element-type 'double-float))
                    (total-elements (* batch-size length)))
               
               ;; Compute mean per channel across batch and temporal dimensions
               (dotimes (c channels)
                 (let ((sum 0.0d0))
                   (dotimes (b batch-size)
                     (dotimes (l length)
                       (incf sum (aref data b c l))))
                   (setf (aref mean-data c) (/ sum total-elements))))
               
               ;; Compute variance per channel
               (dotimes (c channels)
                 (let ((var-sum 0.0d0)
                       (mean (aref mean-data c)))
                   (dotimes (b batch-size)
                     (dotimes (l length)
                       (let ((diff (- (aref data b c l) mean)))
                         (incf var-sum (* diff diff)))))
                   (setf (aref var-data c) (/ var-sum total-elements))))
               
               ;; Update running statistics
               (let ((mom (momentum layer))
                     (running-mean-data (tensor-data (running-mean layer)))
                     (running-var-data (tensor-data (running-var layer))))
                 (dotimes (c channels)
                   (setf (aref running-mean-data c)
                         (+ (* (- 1.0d0 mom) (aref running-mean-data c))
                            (* mom (aref mean-data c))))
                   (setf (aref running-var-data c)
                         (+ (* (- 1.0d0 mom) (aref running-var-data c))
                            (* mom (aref var-data c))))))
               
               ;; Normalize
               (dotimes (b batch-size)
                 (dotimes (c channels)
                   (let ((mean (aref mean-data c))
                         (std (sqrt (+ (aref var-data c) eps-val))))
                     (dotimes (l length)
                       (setf (aref normalized-data b c l)
                             (/ (- (aref data b c l) mean) std))))))
               
               ;; Apply scale and shift - gamma and beta are per-channel
               ;; Need to broadcast them across temporal dimension
               (let ((output-data (make-array shape :element-type 'double-float))
                     (gamma-data (tensor-data gamma))
                     (beta-data (tensor-data beta)))
                 (dotimes (b batch-size)
                   (dotimes (c channels)
                     (let ((g (aref gamma-data c))
                           (bt (aref beta-data c)))
                       (dotimes (l length)
                         (setf (aref output-data b c l)
                               (+ (* g (aref normalized-data b c l)) bt))))))
                 (make-tensor output-data 
                            :shape shape
                            :requires-grad (requires-grad input))))
             
             ;; Evaluation mode: use running statistics
             (let* ((output-data (make-array shape :element-type 'double-float))
                    (running-mean-data (tensor-data (running-mean layer)))
                    (running-var-data (tensor-data (running-var layer)))
                    (gamma-data (tensor-data gamma))
                    (beta-data (tensor-data beta)))
               
               ;; Normalize and scale using running statistics
               (dotimes (b batch-size)
                 (dotimes (c channels)
                   (let ((mean (aref running-mean-data c))
                         (std (sqrt (+ (aref running-var-data c) eps-val)))
                         (g (aref gamma-data c))
                         (bt (aref beta-data c)))
                     (dotimes (l length)
                       (setf (aref output-data b c l)
                             (+ (* g (/ (- (aref data b c l) mean) std)) bt))))))
               
               (make-tensor output-data 
                          :shape shape
                          :requires-grad (requires-grad input))))))
      
      ;; 4D input: (batch, channels, height, width) - standard for CNNs
      ((= (length shape) 4)
       (let* ((batch-size (first shape))
              (channels (second shape))
              (height (third shape))
              (width (fourth shape))
              (spatial-size (* height width)))
         (unless (= channels num-features)
           (error "Input channel dimension ~a doesn't match num-features ~a" 
                  channels num-features))
         
         (if training
             ;; Training mode: compute batch statistics across batch and spatial dimensions
             (let* ((mean-data (make-array (list channels) :element-type 'double-float 
                                          :initial-element 0.0d0))
                    (var-data (make-array (list channels) :element-type 'double-float
                                         :initial-element 0.0d0))
                    (normalized-data (make-array shape :element-type 'double-float))
                    (total-elements (* batch-size spatial-size)))
               
               ;; Compute mean per channel across batch and spatial dimensions
               (dotimes (c channels)
                 (let ((sum 0.0d0))
                   (dotimes (b batch-size)
                     (dotimes (h height)
                       (dotimes (w width)
                         (incf sum (aref data b c h w)))))
                   (setf (aref mean-data c) (/ sum total-elements))))
               
               ;; Compute variance per channel
               (dotimes (c channels)
                 (let ((var-sum 0.0d0)
                       (mean (aref mean-data c)))
                   (dotimes (b batch-size)
                     (dotimes (h height)
                       (dotimes (w width)
                         (let ((diff (- (aref data b c h w) mean)))
                           (incf var-sum (* diff diff))))))
                   (setf (aref var-data c) (/ var-sum total-elements))))
               
               ;; Update running statistics
               (let ((mom (momentum layer))
                     (running-mean-data (tensor-data (running-mean layer)))
                     (running-var-data (tensor-data (running-var layer))))
                 (dotimes (c channels)
                   (setf (aref running-mean-data c)
                         (+ (* (- 1.0d0 mom) (aref running-mean-data c))
                            (* mom (aref mean-data c))))
                   (setf (aref running-var-data c)
                         (+ (* (- 1.0d0 mom) (aref running-var-data c))
                            (* mom (aref var-data c))))))
               
               ;; Normalize
               (dotimes (b batch-size)
                 (dotimes (c channels)
                   (let ((mean (aref mean-data c))
                         (std (sqrt (+ (aref var-data c) eps-val))))
                     (dotimes (h height)
                       (dotimes (w width)
                         (setf (aref normalized-data b c h w)
                               (/ (- (aref data b c h w) mean) std)))))))
               
               ;; Apply scale and shift - gamma and beta are per-channel
               ;; Need to broadcast them across spatial dimensions
               (let ((output-data (make-array shape :element-type 'double-float))
                     (gamma-data (tensor-data gamma))
                     (beta-data (tensor-data beta)))
                 (dotimes (b batch-size)
                   (dotimes (c channels)
                     (let ((g (aref gamma-data c))
                           (bt (aref beta-data c)))
                       (dotimes (h height)
                         (dotimes (w width)
                           (setf (aref output-data b c h w)
                                 (+ (* g (aref normalized-data b c h w)) bt)))))))
                 (make-tensor output-data 
                            :shape shape
                            :requires-grad (requires-grad input))))
             
             ;; Evaluation mode: use running statistics
             (let* ((output-data (make-array shape :element-type 'double-float))
                    (running-mean-data (tensor-data (running-mean layer)))
                    (running-var-data (tensor-data (running-var layer)))
                    (gamma-data (tensor-data gamma))
                    (beta-data (tensor-data beta)))
               
               ;; Normalize and scale using running statistics
               (dotimes (b batch-size)
                 (dotimes (c channels)
                   (let ((mean (aref running-mean-data c))
                         (std (sqrt (+ (aref running-var-data c) eps-val)))
                         (g (aref gamma-data c))
                         (bt (aref beta-data c)))
                     (dotimes (h height)
                       (dotimes (w width)
                         (setf (aref output-data b c h w)
                               (+ (* g (/ (- (aref data b c h w) mean) std)) bt)))))))
               
               (make-tensor output-data 
                          :shape shape
                          :requires-grad (requires-grad input))))))
      
      ;; 5D input: (batch, channels, depth, height, width) - for 3D CNNs
      ((= (length shape) 5)
       (let* ((batch-size (first shape))
              (channels (second shape))
              (depth (third shape))
              (height (fourth shape))
              (width (fifth shape))
              (spatial-size (* depth height width)))
         (unless (= channels num-features)
           (error "Input channel dimension ~a doesn't match num-features ~a" 
                  channels num-features))
         
         (if training
             ;; Training mode: compute batch statistics across batch and spatial dimensions
             (let* ((mean-data (make-array (list channels) :element-type 'double-float 
                                          :initial-element 0.0d0))
                    (var-data (make-array (list channels) :element-type 'double-float
                                         :initial-element 0.0d0))
                    (normalized-data (make-array shape :element-type 'double-float))
                    (total-elements (* batch-size spatial-size)))
               
               ;; Compute mean per channel across batch and spatial dimensions
               (dotimes (c channels)
                 (let ((sum 0.0d0))
                   (dotimes (b batch-size)
                     (dotimes (d depth)
                       (dotimes (h height)
                         (dotimes (w width)
                           (incf sum (aref data b c d h w))))))
                   (setf (aref mean-data c) (/ sum total-elements))))
               
               ;; Compute variance per channel
               (dotimes (c channels)
                 (let ((var-sum 0.0d0)
                       (mean (aref mean-data c)))
                   (dotimes (b batch-size)
                     (dotimes (d depth)
                       (dotimes (h height)
                         (dotimes (w width)
                           (let ((diff (- (aref data b c d h w) mean)))
                             (incf var-sum (* diff diff)))))))
                   (setf (aref var-data c) (/ var-sum total-elements))))
               
               ;; Update running statistics
               (let ((mom (momentum layer))
                     (running-mean-data (tensor-data (running-mean layer)))
                     (running-var-data (tensor-data (running-var layer))))
                 (dotimes (c channels)
                   (setf (aref running-mean-data c)
                         (+ (* (- 1.0d0 mom) (aref running-mean-data c))
                            (* mom (aref mean-data c))))
                   (setf (aref running-var-data c)
                         (+ (* (- 1.0d0 mom) (aref running-var-data c))
                            (* mom (aref var-data c))))))
               
               ;; Normalize
               (dotimes (b batch-size)
                 (dotimes (c channels)
                   (let ((mean (aref mean-data c))
                         (std (sqrt (+ (aref var-data c) eps-val))))
                     (dotimes (d depth)
                       (dotimes (h height)
                         (dotimes (w width)
                           (setf (aref normalized-data b c d h w)
                                 (/ (- (aref data b c d h w) mean) std))))))))
               
               ;; Apply scale and shift - gamma and beta are per-channel
               ;; Need to broadcast them across spatial dimensions
               (let ((output-data (make-array shape :element-type 'double-float))
                     (gamma-data (tensor-data gamma))
                     (beta-data (tensor-data beta)))
                 (dotimes (b batch-size)
                   (dotimes (c channels)
                     (let ((g (aref gamma-data c))
                           (bt (aref beta-data c)))
                       (dotimes (d depth)
                         (dotimes (h height)
                           (dotimes (w width)
                             (setf (aref output-data b c d h w)
                                   (+ (* g (aref normalized-data b c d h w)) bt))))))))
                 (make-tensor output-data 
                            :shape shape
                            :requires-grad (requires-grad input))))
             
             ;; Evaluation mode: use running statistics
             (let* ((output-data (make-array shape :element-type 'double-float))
                    (running-mean-data (tensor-data (running-mean layer)))
                    (running-var-data (tensor-data (running-var layer)))
                    (gamma-data (tensor-data gamma))
                    (beta-data (tensor-data beta)))
               
               ;; Normalize and scale using running statistics
               (dotimes (b batch-size)
                 (dotimes (c channels)
                   (let ((mean (aref running-mean-data c))
                         (std (sqrt (+ (aref running-var-data c) eps-val)))
                         (g (aref gamma-data c))
                         (bt (aref beta-data c)))
                     (dotimes (d depth)
                       (dotimes (h height)
                         (dotimes (w width)
                           (setf (aref output-data b c d h w)
                                 (+ (* g (/ (- (aref data b c d h w) mean) std)) bt))))))))
               
               (make-tensor output-data 
                          :shape shape
                          :requires-grad (requires-grad input))))))
      
      (t (error "Unsupported input shape for batch norm: ~a. Expected 2D (batch, features), 3D (batch, channels, length), 4D (batch, channels, height, width), or 5D (batch, channels, depth, height, width)" shape)))))

