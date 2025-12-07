;;;; Neural Tensor Library - Convolutional Neural Networks
;;;; Supporting 1D, 2D, 3D, and N-dimensional convolutions
;;;; Demonstrating Lisp's ability to handle complex tensor operations

(defpackage :neural-tensor-convolution
  (:use :common-lisp)
  (:import-from :neural-network
                #:tensor
                #:tensor-data
                #:tensor-shape
                #:tensor-grad
                #:tensor-name
                #:requires-grad
                #:make-tensor
                #:zeros
                #:randn
                #:forward
                #:layer
                #:layer-parameters
                #:layer-training
                #:t+
                #:t*)
  (:import-from :neural-tensor-activations
                #:relu)
  (:export ;; Base convolution classes
           #:conv-layer
           #:conv1d
           #:conv2d
           #:conv3d
           #:convnd
           ;; Pooling layers
           #:max-pool1d
           #:max-pool2d
           #:max-pool3d
           #:avg-pool1d
           #:avg-pool2d
           #:avg-pool3d
           ;; Global pooling
           #:global-avg-pool1d
           #:global-avg-pool2d
           #:global-avg-pool3d
           #:global-max-pool1d
           #:global-max-pool2d
           #:global-max-pool3d
           ;; Spatial Pyramid Pooling
           #:spatial-pyramid-pool2d
           ;; Transposed convolutions (deconvolutions)
           #:conv-transpose1d
           #:conv-transpose2d
           #:conv-transpose3d
           ;; Separable convolutions
           #:separable-conv2d
           #:depthwise-conv2d
           ;; Dilated/Atrous convolutions
           #:dilated-conv2d
           ;; Utility functions
           #:calculate-output-size
           #:im2col
           #:col2im
           #:pad-tensor
           ;; Convolution parameters
           #:conv-kernel
           #:conv-bias
           #:kernel-size
           #:stride
           #:padding
           #:dilation
           #:groups))

(in-package :neural-tensor-convolution)

;;;; ============================================================================
;;;; Base Convolution Layer Class
;;;; ============================================================================

(defclass conv-layer (layer)
  ((kernel :accessor conv-kernel
           :documentation "Convolution kernel weights")
   (bias :accessor conv-bias
         :documentation "Bias terms")
   (in-channels :initarg :in-channels
                :reader in-channels
                :type integer
                :documentation "Number of input channels")
   (out-channels :initarg :out-channels
                 :reader out-channels
                 :type integer
                 :documentation "Number of output channels")
   (kernel-size :initarg :kernel-size
                :reader kernel-size
                :documentation "Size of convolution kernel (single int or list)")
   (stride :initarg :stride
           :initform 1
           :reader stride
           :documentation "Stride for convolution")
   (padding :initarg :padding
            :initform 0
            :reader padding
            :documentation "Padding to apply")
   (dilation :initarg :dilation
             :initform 1
             :reader dilation
             :documentation "Dilation rate for dilated convolution")
   (groups :initarg :groups
           :initform 1
           :reader groups
           :documentation "Number of groups for grouped convolution")
   (use-bias :initarg :use-bias
             :initform t
             :reader use-bias
             :documentation "Whether to use bias"))
  (:documentation "Base class for convolutional layers"))

;;;; ============================================================================
;;;; Utility Functions
;;;; ============================================================================

(defun ensure-list (x length)
  "Ensure X is a list of LENGTH elements"
  (cond
    ((listp x) x)
    (t (make-list length :initial-element x))))

(defun calculate-output-size (input-size kernel-size stride padding dilation)
  "Calculate output size for convolution operation"
  (1+ (floor (- (+ input-size (* 2 padding))
                (* dilation (- kernel-size 1))
                1)
             stride)))

(defun calculate-padding-for-same (input-size kernel-size stride dilation)
  "Calculate padding needed for 'same' output size"
  (let* ((effective-kernel-size (+ (* dilation (- kernel-size 1)) 1))
         (total-padding (max 0 (- (* stride (- input-size 1))
                                 input-size
                                 (- effective-kernel-size)))))
    (floor total-padding 2)))

(defun pad-tensor (tensor padding-spec &optional (pad-value 0.0d0))
  "Pad a tensor according to padding specification
   PADDING-SPEC: list of (before after) pairs for each dimension
   Example: ((1 1) (2 2)) pads first dim by 1 on each side, second by 2"
  (let* ((shape (tensor-shape tensor))
         (data (tensor-data tensor))
         (ndims (length shape))
         (new-shape (loop for dim in shape
                         for (before after) in padding-spec
                         collect (+ dim before after)))
         (new-data (make-array new-shape 
                              :element-type 'double-float 
                              :initial-element pad-value)))
    
    ;; Copy data to padded array
    (labels ((copy-recursive (src-idx dst-idx dim)
               (if (= dim ndims)
                   (setf (apply #'aref new-data dst-idx)
                         (apply #'aref data src-idx))
                   (let ((pad-before (car (nth dim padding-spec))))
                     (dotimes (i (nth dim shape))
                       (copy-recursive (append src-idx (list i))
                                     (append dst-idx (list (+ i pad-before)))
                                     (1+ dim)))))))
      (copy-recursive nil nil 0))
    
    (make-tensor new-data
                :shape new-shape
                :requires-grad (requires-grad tensor))))

;;;; ============================================================================
;;;; Im2Col and Col2Im - Core Convolution Algorithms
;;;; ============================================================================

(defun im2col-2d (input kernel-h kernel-w stride-h stride-w padding-h padding-w dilation-h dilation-w)
  "Transform image to column matrix for efficient convolution (2D)
   This is the classic im2col algorithm used in many deep learning frameworks"
  (let* ((shape (tensor-shape input))
         (batch-size (first shape))
         (in-channels (second shape))
         (in-h (third shape))
         (in-w (fourth shape))
         
         ;; Apply padding
         (padded-input (if (or (> padding-h 0) (> padding-w 0))
                          (pad-tensor input 
                                    (list '(0 0) '(0 0) 
                                          (list padding-h padding-h)
                                          (list padding-w padding-w)))
                          input))
         (padded-shape (tensor-shape padded-input))
         (padded-h (third padded-shape))
         (padded-w (fourth padded-shape))
         (padded-data (tensor-data padded-input))
         
         ;; Calculate output dimensions
         (out-h (calculate-output-size in-h kernel-h stride-h padding-h dilation-h))
         (out-w (calculate-output-size in-w kernel-w stride-w padding-w dilation-w))
         
         ;; Create column matrix
         (col-height (* in-channels kernel-h kernel-w))
         (col-width (* batch-size out-h out-w))
         (col-matrix (make-array (list col-height col-width)
                                :element-type 'double-float
                                :initial-element 0.0d0)))
    
    ;; Fill column matrix
    (loop for b from 0 below batch-size do
      (loop for out-y from 0 below out-h do
        (loop for out-x from 0 below out-w do
          (let ((col-idx (+ (* b out-h out-w)
                           (* out-y out-w)
                           out-x)))
            (loop for c from 0 below in-channels do
              (loop for ky from 0 below kernel-h do
                (loop for kx from 0 below kernel-w do
                  (let* ((in-y (+ (* out-y stride-h) (* ky dilation-h)))
                         (in-x (+ (* out-x stride-w) (* kx dilation-w)))
                         (row-idx (+ (* c kernel-h kernel-w)
                                   (* ky kernel-w)
                                   kx)))
                    (when (and (< in-y padded-h) (< in-x padded-w))
                      (setf (aref col-matrix row-idx col-idx)
                            (aref padded-data b c in-y in-x)))))))))))
    
    (make-tensor col-matrix
                :shape (list col-height col-width)
                :requires-grad (requires-grad input))))

(defun col2im-2d (col-tensor output-shape kernel-h kernel-w stride-h stride-w 
                  padding-h padding-w dilation-h dilation-w)
  "Transform column matrix back to image (inverse of im2col)"
  (let* ((batch-size (first output-shape))
         (in-channels (second output-shape))
         (out-h (third output-shape))
         (out-w (fourth output-shape))
         (padded-h (+ out-h (* 2 padding-h)))
         (padded-w (+ out-w (* 2 padding-w)))
         (padded-data (make-array (list batch-size in-channels padded-h padded-w)
                                 :element-type 'double-float
                                 :initial-element 0.0d0))
         (col-data (tensor-data col-tensor))
         (col-shape (tensor-shape col-tensor)))
    
    (declare (ignore col-shape))
    
    ;; Reverse the im2col operation
    (loop for b from 0 below batch-size do
      (loop for out-y from 0 below (calculate-output-size out-h kernel-h stride-h padding-h dilation-h) do
        (loop for out-x from 0 below (calculate-output-size out-w kernel-w stride-w padding-w dilation-w) do
          (let ((col-idx (+ (* b out-h out-w)
                           (* out-y out-w)
                           out-x)))
            (loop for c from 0 below in-channels do
              (loop for ky from 0 below kernel-h do
                (loop for kx from 0 below kernel-w do
                  (let* ((in-y (+ (* out-y stride-h) (* ky dilation-h)))
                         (in-x (+ (* out-x stride-w) (* kx dilation-w)))
                         (row-idx (+ (* c kernel-h kernel-w)
                                   (* ky kernel-w)
                                   kx)))
                    (when (and (< in-y padded-h) (< in-x padded-w))
                      (incf (aref padded-data b c in-y in-x)
                            (aref col-data row-idx col-idx)))))))))))
    
    ;; Remove padding if necessary
    (if (or (> padding-h 0) (> padding-w 0))
        (let ((unpadded-data (make-array output-shape :element-type 'double-float)))
          (loop for b from 0 below batch-size do
            (loop for c from 0 below in-channels do
              (loop for h from 0 below out-h do
                (loop for w from 0 below out-w do
                  (setf (aref unpadded-data b c h w)
                        (aref padded-data b c (+ h padding-h) (+ w padding-w)))))))
          (make-tensor unpadded-data :shape output-shape))
        (make-tensor padded-data :shape output-shape))))

;;;; ============================================================================
;;;; 1D Convolution
;;;; ============================================================================

(defclass conv1d (conv-layer)
  ()
  (:documentation "1D Convolutional layer for sequential data"))

(defmethod initialize-instance :after ((layer conv1d) &key)
  (with-slots (kernel bias in-channels out-channels kernel-size stride padding 
               dilation groups use-bias) layer
    
    ;; Ensure kernel-size is an integer
    (when (listp kernel-size)
      (setf kernel-size (first kernel-size)))
    
    ;; Initialize weights with He initialization
    (let* ((fan-in (* in-channels kernel-size))
           (std (sqrt (/ 2.0d0 fan-in))))
      (setf kernel (randn (list out-channels (/ in-channels groups) kernel-size)
                         :requires-grad t
                         :scale std
                         :name "conv1d-kernel")))
    
    ;; Initialize bias
    (when use-bias
      (setf bias (zeros (list out-channels)
                       :requires-grad t
                       :name "conv1d-bias")))
    
    ;; Register parameters
    (setf (slot-value layer 'neural-network:parameters)
          (if use-bias
              (list kernel bias)
              (list kernel)))))

(defmethod forward ((layer conv1d) input)
  "Forward pass for 1D convolution
   Input shape: (batch, in-channels, length)
   Output shape: (batch, out-channels, out-length)"
  (with-slots (kernel bias in-channels out-channels kernel-size stride padding 
               dilation groups use-bias) layer
    
    (let* ((input-shape (tensor-shape input))
           (batch-size (first input-shape))
           (in-length (third input-shape))
           
           ;; Calculate output length
           (out-length (calculate-output-size in-length kernel-size stride padding dilation))
           
           ;; Pad input if necessary
           (padded-input (if (> padding 0)
                            (pad-tensor input 
                                      (list '(0 0) '(0 0) (list padding padding)))
                            input))
           (padded-data (tensor-data padded-input))
           (padded-length (if (> padding 0)
                            (+ in-length (* 2 padding))
                            in-length))
           
           ;; Create output tensor
           (output-data (make-array (list batch-size out-channels out-length)
                                   :element-type 'double-float
                                   :initial-element 0.0d0))
           (kernel-data (tensor-data kernel)))
      
      ;; Perform convolution
      (loop for b from 0 below batch-size do
        (loop for oc from 0 below out-channels do
          (loop for ol from 0 below out-length do
            (let ((sum 0.0d0)
                  (channels-per-group (/ in-channels groups))
                  (group-id (floor oc (/ out-channels groups))))
              ;; Convolve kernel
              (loop for ic from 0 below channels-per-group do
                (let ((input-channel (+ (* group-id channels-per-group) ic)))
                  (loop for k from 0 below kernel-size do
                    (let ((input-pos (+ (* ol stride) (* k dilation))))
                      (when (< input-pos padded-length)
                        (incf sum 
                              (* (aref kernel-data oc ic k)
                                 (aref padded-data b input-channel input-pos))))))))
              
              ;; Add bias
              (when use-bias
                (incf sum (aref (tensor-data bias) oc)))
              
              (setf (aref output-data b oc ol) sum)))))
      
      (make-tensor output-data
                  :shape (list batch-size out-channels out-length)
                  :requires-grad (or (requires-grad input)
                                   (requires-grad kernel))))))

;;;; ============================================================================
;;;; 2D Convolution
;;;; ============================================================================

(defclass conv2d (conv-layer)
  ()
  (:documentation "2D Convolutional layer for image data"))

(defmethod initialize-instance :after ((layer conv2d) &key)
  (with-slots (kernel bias in-channels out-channels kernel-size stride padding 
               dilation groups use-bias) layer
    
    ;; Ensure kernel-size, stride, padding, dilation are lists of length 2
    (setf kernel-size (ensure-list kernel-size 2))
    (setf stride (ensure-list stride 2))
    (setf padding (ensure-list padding 2))
    (setf dilation (ensure-list dilation 2))
    
    ;; Initialize weights with He initialization
    (let* ((kh (first kernel-size))
           (kw (second kernel-size))
           (fan-in (* in-channels kh kw))
           (std (sqrt (/ 2.0d0 fan-in))))
      (setf kernel (randn (list out-channels (/ in-channels groups) kh kw)
                         :requires-grad t
                         :scale std
                         :name "conv2d-kernel")))
    
    ;; Initialize bias
    (when use-bias
      (setf bias (zeros (list out-channels)
                       :requires-grad t
                       :name "conv2d-bias")))
    
    ;; Register parameters
    (setf (slot-value layer 'neural-network:parameters)
          (if use-bias
              (list kernel bias)
              (list kernel)))))

(defmethod forward ((layer conv2d) input)
  "Forward pass for 2D convolution
   Input shape: (batch, in-channels, height, width)
   Output shape: (batch, out-channels, out-height, out-width)"
  (with-slots (kernel bias in-channels out-channels kernel-size stride padding 
               dilation groups use-bias) layer
    
    (let* ((input-shape (tensor-shape input))
           (batch-size (first input-shape))
           (in-h (third input-shape))
           (in-w (fourth input-shape))
           (kh (first kernel-size))
           (kw (second kernel-size))
           (sh (first stride))
           (sw (second stride))
           (ph (first padding))
           (pw (second padding))
           (dh (first dilation))
           (dw (second dilation))
           
           ;; Calculate output dimensions
           (out-h (calculate-output-size in-h kh sh ph dh))
           (out-w (calculate-output-size in-w kw sw pw dw))
           
           ;; Pad input if necessary
           (padded-input (if (or (> ph 0) (> pw 0))
                            (pad-tensor input 
                                      (list '(0 0) '(0 0) 
                                            (list ph ph)
                                            (list pw pw)))
                            input))
           (padded-data (tensor-data padded-input))
           (padded-h (if (> ph 0) (+ in-h (* 2 ph)) in-h))
           (padded-w (if (> pw 0) (+ in-w (* 2 pw)) in-w))
           
           ;; Create output tensor
           (output-data (make-array (list batch-size out-channels out-h out-w)
                                   :element-type 'double-float
                                   :initial-element 0.0d0))
           (kernel-data (tensor-data kernel)))
      
      ;; Perform convolution
      (loop for b from 0 below batch-size do
        (loop for oc from 0 below out-channels do
          (loop for oh from 0 below out-h do
            (loop for ow from 0 below out-w do
              (let ((sum 0.0d0)
                    (channels-per-group (/ in-channels groups))
                    (group-id (floor oc (/ out-channels groups))))
                ;; Convolve kernel
                (loop for ic from 0 below channels-per-group do
                  (let ((input-channel (+ (* group-id channels-per-group) ic)))
                    (loop for kh-idx from 0 below kh do
                      (loop for kw-idx from 0 below kw do
                        (let ((input-h (+ (* oh sh) (* kh-idx dh)))
                              (input-w (+ (* ow sw) (* kw-idx dw))))
                          (when (and (< input-h padded-h) (< input-w padded-w))
                            (incf sum 
                                  (* (aref kernel-data oc ic kh-idx kw-idx)
                                     (aref padded-data b input-channel input-h input-w)))))))))
                
                ;; Add bias
                (when use-bias
                  (incf sum (aref (tensor-data bias) oc)))
                
                (setf (aref output-data b oc oh ow) sum))))))
      
      (make-tensor output-data
                  :shape (list batch-size out-channels out-h out-w)
                  :requires-grad (or (requires-grad input)
                                   (requires-grad kernel))))))

;;;; ============================================================================
;;;; 3D Convolution
;;;; ============================================================================

(defclass conv3d (conv-layer)
  ()
  (:documentation "3D Convolutional layer for volumetric data"))

(defmethod initialize-instance :after ((layer conv3d) &key)
  (with-slots (kernel bias in-channels out-channels kernel-size stride padding 
               dilation groups use-bias) layer
    
    ;; Ensure kernel-size, stride, padding, dilation are lists of length 3
    (setf kernel-size (ensure-list kernel-size 3))
    (setf stride (ensure-list stride 3))
    (setf padding (ensure-list padding 3))
    (setf dilation (ensure-list dilation 3))
    
    ;; Initialize weights with He initialization
    (let* ((kd (first kernel-size))
           (kh (second kernel-size))
           (kw (third kernel-size))
           (fan-in (* in-channels kd kh kw))
           (std (sqrt (/ 2.0d0 fan-in))))
      (setf kernel (randn (list out-channels (/ in-channels groups) kd kh kw)
                         :requires-grad t
                         :scale std
                         :name "conv3d-kernel")))
    
    ;; Initialize bias
    (when use-bias
      (setf bias (zeros (list out-channels)
                       :requires-grad t
                       :name "conv3d-bias")))
    
    ;; Register parameters
    (setf (slot-value layer 'neural-network:parameters)
          (if use-bias
              (list kernel bias)
              (list kernel)))))

(defmethod forward ((layer conv3d) input)
  "Forward pass for 3D convolution
   Input shape: (batch, in-channels, depth, height, width)
   Output shape: (batch, out-channels, out-depth, out-height, out-width)"
  (with-slots (kernel bias in-channels out-channels kernel-size stride padding 
               dilation groups use-bias) layer
    
    (let* ((input-shape (tensor-shape input))
           (batch-size (first input-shape))
           (in-d (third input-shape))
           (in-h (fourth input-shape))
           (in-w (fifth input-shape))
           (kd (first kernel-size))
           (kh (second kernel-size))
           (kw (third kernel-size))
           (sd (first stride))
           (sh (second stride))
           (sw (third stride))
           (pd (first padding))
           (ph (second padding))
           (pw (third padding))
           (dd (first dilation))
           (dh (second dilation))
           (dw (third dilation))
           
           ;; Calculate output dimensions
           (out-d (calculate-output-size in-d kd sd pd dd))
           (out-h (calculate-output-size in-h kh sh ph dh))
           (out-w (calculate-output-size in-w kw sw pw dw))
           
           ;; Pad input if necessary
           (padded-input (if (or (> pd 0) (> ph 0) (> pw 0))
                            (pad-tensor input 
                                      (list '(0 0) '(0 0) 
                                            (list pd pd)
                                            (list ph ph)
                                            (list pw pw)))
                            input))
           (padded-data (tensor-data padded-input))
           (padded-d (if (> pd 0) (+ in-d (* 2 pd)) in-d))
           (padded-h (if (> ph 0) (+ in-h (* 2 ph)) in-h))
           (padded-w (if (> pw 0) (+ in-w (* 2 pw)) in-w))
           
           ;; Create output tensor
           (output-data (make-array (list batch-size out-channels out-d out-h out-w)
                                   :element-type 'double-float
                                   :initial-element 0.0d0))
           (kernel-data (tensor-data kernel)))
      
      ;; Perform convolution
      (loop for b from 0 below batch-size do
        (loop for oc from 0 below out-channels do
          (loop for od from 0 below out-d do
            (loop for oh from 0 below out-h do
              (loop for ow from 0 below out-w do
                (let ((sum 0.0d0)
                      (channels-per-group (/ in-channels groups))
                      (group-id (floor oc (/ out-channels groups))))
                  ;; Convolve kernel
                  (loop for ic from 0 below channels-per-group do
                    (let ((input-channel (+ (* group-id channels-per-group) ic)))
                      (loop for kd-idx from 0 below kd do
                        (loop for kh-idx from 0 below kh do
                          (loop for kw-idx from 0 below kw do
                            (let ((input-d (+ (* od sd) (* kd-idx dd)))
                                  (input-h (+ (* oh sh) (* kh-idx dh)))
                                  (input-w (+ (* ow sw) (* kw-idx dw))))
                              (when (and (< input-d padded-d) 
                                       (< input-h padded-h) 
                                       (< input-w padded-w))
                                (incf sum 
                                      (* (aref kernel-data oc ic kd-idx kh-idx kw-idx)
                                         (aref padded-data b input-channel input-d input-h input-w))))))))))
                  
                  ;; Add bias
                  (when use-bias
                    (incf sum (aref (tensor-data bias) oc)))
                  
                  (setf (aref output-data b oc od oh ow) sum)))))))
      
      (make-tensor output-data
                  :shape (list batch-size out-channels out-d out-h out-w)
                  :requires-grad (or (requires-grad input)
                                   (requires-grad kernel))))))

;;;; ============================================================================
;;;; N-Dimensional Convolution (Generalized)
;;;; ============================================================================

(defclass convnd (conv-layer)
  ((ndims :initarg :ndims
          :reader ndims
          :type integer
          :documentation "Number of spatial dimensions"))
  (:documentation "N-dimensional convolutional layer (generalized convolution)"))

(defmethod initialize-instance :after ((layer convnd) &key)
  (with-slots (kernel bias in-channels out-channels kernel-size stride padding 
               dilation groups use-bias ndims) layer
    
    ;; Ensure all parameters are lists of length ndims
    (setf kernel-size (ensure-list kernel-size ndims))
    (setf stride (ensure-list stride ndims))
    (setf padding (ensure-list padding ndims))
    (setf dilation (ensure-list dilation ndims))
    
    ;; Initialize weights with He initialization
    (let* ((fan-in (* in-channels (reduce #'* kernel-size)))
           (std (sqrt (/ 2.0d0 fan-in)))
           (kernel-shape (append (list out-channels (/ in-channels groups))
                                kernel-size)))
      (setf kernel (randn kernel-shape
                         :requires-grad t
                         :scale std
                         :name (format nil "conv~dd-kernel" ndims))))
    
    ;; Initialize bias
    (when use-bias
      (setf bias (zeros (list out-channels)
                       :requires-grad t
                       :name (format nil "conv~dd-bias" ndims))))
    
    ;; Register parameters
    (setf (slot-value layer 'neural-network:parameters)
          (if use-bias
              (list kernel bias)
              (list kernel)))))

(defmethod forward ((layer convnd) input)
  "Forward pass for N-D convolution - generalized for arbitrary dimensions"
  (with-slots (kernel bias in-channels out-channels kernel-size stride padding 
               dilation groups use-bias ndims) layer
    
    (let* ((input-shape (tensor-shape input))
           (batch-size (first input-shape))
           (spatial-dims (subseq input-shape 2))
           
           ;; Calculate output dimensions
           (output-spatial-dims
            (loop for in-size in spatial-dims
                  for k in kernel-size
                  for s in stride
                  for p in padding
                  for d in dilation
                  collect (calculate-output-size in-size k s p d)))
           
           ;; Pad input if necessary
           (padded-input (if (some #'plusp padding)
                            (pad-tensor input 
                                      (append '((0 0) (0 0))
                                            (mapcar (lambda (p) (list p p)) padding)))
                            input))
           (padded-data (tensor-data padded-input))
           (padded-dims (if (some #'plusp padding)
                           (loop for in-size in spatial-dims
                                 for p in padding
                                 collect (+ in-size (* 2 p)))
                           spatial-dims))
           
           ;; Create output tensor
           (output-shape (append (list batch-size out-channels) output-spatial-dims))
           (output-data (make-array output-shape
                                   :element-type 'double-float
                                   :initial-element 0.0d0))
           (kernel-data (tensor-data kernel)))
      
      ;; Recursive convolution for arbitrary dimensions
      (labels ((convolve-recursive (indices depth)
                 "Recursively iterate through all spatial positions"
                 (if (= depth (+ ndims 2))  ; Need to iterate: batch + channels + ndims spatial
                     ;; Base case: compute convolution at this position
                     (let ((b (first indices))
                           (oc (second indices))
                           (spatial-out-indices (subseq indices 2)))
                       (let ((sum 0.0d0)
                             (channels-per-group (/ in-channels groups))
                             (group-id (floor oc (/ out-channels groups))))
                         ;; Convolve kernel
                         (iterate-kernel 
                          (lambda (kernel-indices)
                            (let ((spatial-input-indices
                                   (loop for out-idx in spatial-out-indices
                                         for k-idx in kernel-indices
                                         for s in stride
                                         for d in dilation
                                         collect (+ (* out-idx s) (* k-idx d)))))
                              ;; Check bounds - ensure all spatial indices are within padded dimensions
                              (when (every (lambda (idx dim) (and (>= idx 0) (< idx dim)))
                                          spatial-input-indices 
                                          padded-dims)
                                (loop for ic from 0 below channels-per-group do
                                  (let* ((input-ch (+ (* group-id channels-per-group) ic))
                                         (kernel-idx (append (list oc ic) kernel-indices))
                                         (input-idx (append (list b input-ch) 
                                                          spatial-input-indices)))
                                    (incf sum 
                                          (* (apply #'aref kernel-data kernel-idx)
                                             (apply #'aref padded-data input-idx))))))))
                          kernel-size
                          nil
                          0)
                         
                         ;; Add bias
                         (when use-bias
                           (incf sum (aref (tensor-data bias) oc)))
                         
                         (apply #'(setf aref) sum output-data indices)))
                     
                     ;; Recursive case: iterate through dimension
                     (let ((dim-size (cond
                                       ((= depth 0) batch-size)
                                       ((= depth 1) out-channels)
                                       (t (nth (- depth 2) output-spatial-dims)))))
                       (dotimes (i dim-size)
                         (convolve-recursive (append indices (list i)) (1+ depth))))))
               
               (iterate-kernel (fn dims accumulated depth)
                 "Helper to iterate through kernel dimensions"
                 (if (= depth ndims)
                     (funcall fn (reverse accumulated))
                     (dotimes (i (nth depth dims))
                       (iterate-kernel fn dims (cons i accumulated) (1+ depth))))))
        
        ;; Start convolution
        (convolve-recursive nil 0))
      
      (make-tensor output-data
                  :shape output-shape
                  :requires-grad (or (requires-grad input)
                                   (requires-grad kernel))))))

;;;; ============================================================================
;;;; Pooling Layers
;;;; ============================================================================

(defclass pooling-layer (layer)
  ((kernel-size :initarg :kernel-size
                :reader kernel-size)
   (stride :initarg :stride
           :reader stride)
   (padding :initarg :padding
            :initform 0
            :reader padding)
   (pool-type :initarg :pool-type
              :reader pool-type
              :documentation "Type of pooling: :max or :avg"))
  (:documentation "Base class for pooling layers"))

(defclass max-pool2d (pooling-layer)
  ()
  (:default-initargs :pool-type :max)
  (:documentation "2D Max pooling layer"))

(defclass avg-pool2d (pooling-layer)
  ()
  (:default-initargs :pool-type :avg)
  (:documentation "2D Average pooling layer"))

(defmethod forward ((layer max-pool2d) input)
  "Forward pass for 2D max pooling"
  (pool-2d input 
           (kernel-size layer)
           (stride layer)
           (padding layer)
           :max))

(defmethod forward ((layer avg-pool2d) input)
  "Forward pass for 2D average pooling"
  (pool-2d input 
           (kernel-size layer)
           (stride layer)
           (padding layer)
           :avg))

(defun pool-2d (input kernel-size stride padding pool-type)
  "Generic 2D pooling operation"
  (let* ((kernel-size (ensure-list kernel-size 2))
         (stride (if stride (ensure-list stride 2) kernel-size))
         (padding (ensure-list padding 2))
         (input-shape (tensor-shape input))
         (batch-size (first input-shape))
         (channels (second input-shape))
         (in-h (third input-shape))
         (in-w (fourth input-shape))
         (kh (first kernel-size))
         (kw (second kernel-size))
         (sh (first stride))
         (sw (second stride))
         (ph (first padding))
         (pw (second padding))
         
         ;; Pad if necessary
         (padded-input (if (or (> ph 0) (> pw 0))
                          (pad-tensor input 
                                    (list '(0 0) '(0 0) 
                                          (list ph ph)
                                          (list pw pw))
                                    (if (eq pool-type :max)
                                        most-negative-double-float
                                        0.0d0))
                          input))
         (padded-data (tensor-data padded-input))
         (padded-h (if (> ph 0) (+ in-h (* 2 ph)) in-h))
         (padded-w (if (> pw 0) (+ in-w (* 2 pw)) in-w))
         
         ;; Calculate output dimensions
         (out-h (floor (/ (- padded-h kh) sh) 1))
         (out-w (floor (/ (- padded-w kw) sw) 1))
         (out-h (1+ out-h))
         (out-w (1+ out-w))
         
         ;; Create output
         (output-data (make-array (list batch-size channels out-h out-w)
                                 :element-type 'double-float
                                 :initial-element 0.0d0)))
    
    ;; Perform pooling
    (loop for b from 0 below batch-size do
      (loop for c from 0 below channels do
        (loop for oh from 0 below out-h do
          (loop for ow from 0 below out-w do
            (let ((pool-value (if (eq pool-type :max)
                                most-negative-double-float
                                0.0d0))
                  (count 0))
              (loop for kh-idx from 0 below kh do
                (loop for kw-idx from 0 below kw do
                  (let ((input-h (+ (* oh sh) kh-idx))
                        (input-w (+ (* ow sw) kw-idx)))
                    (when (and (< input-h padded-h) (< input-w padded-w))
                      (let ((val (aref padded-data b c input-h input-w)))
                        (ecase pool-type
                          (:max (setf pool-value (max pool-value val)))
                          (:avg (incf pool-value val)
                               (incf count))))))))
              
              (when (eq pool-type :avg)
                (setf pool-value (/ pool-value (max 1 count))))
              
              (setf (aref output-data b c oh ow) pool-value))))))
    
    (make-tensor output-data
                :shape (list batch-size channels out-h out-w)
                :requires-grad (requires-grad input))))

;;;; 1D and 3D pooling layers
(defclass max-pool1d (pooling-layer)
  ()
  (:default-initargs :pool-type :max))

(defclass avg-pool1d (pooling-layer)
  ()
  (:default-initargs :pool-type :avg))

(defclass max-pool3d (pooling-layer)
  ()
  (:default-initargs :pool-type :max))

(defclass avg-pool3d (pooling-layer)
  ()
  (:default-initargs :pool-type :avg))

;;;; ============================================================================
;;;; Global Average/Max Pooling Layers
;;;; ============================================================================

(defclass global-avg-pool1d (layer)
  ()
  (:documentation "Global average pooling for 1D data - averages over entire spatial dimension"))

(defclass global-avg-pool2d (layer)
  ()
  (:documentation "Global average pooling for 2D data - averages over entire spatial dimensions"))

(defclass global-avg-pool3d (layer)
  ()
  (:documentation "Global average pooling for 3D data - averages over entire spatial dimensions"))

(defclass global-max-pool1d (layer)
  ()
  (:documentation "Global max pooling for 1D data - takes max over entire spatial dimension"))

(defclass global-max-pool2d (layer)
  ()
  (:documentation "Global max pooling for 2D data - takes max over entire spatial dimensions"))

(defclass global-max-pool3d (layer)
  ()
  (:documentation "Global max pooling for 3D data - takes max over entire spatial dimensions"))

(defmethod forward ((layer global-avg-pool1d) input)
  "Global average pooling for 1D - output shape: (batch, channels, 1)"
  (let* ((input-shape (tensor-shape input))
         (batch-size (first input-shape))
         (channels (second input-shape))
         (length (third input-shape))
         (input-data (tensor-data input))
         (output-data (make-array (list batch-size channels 1)
                                 :element-type 'double-float)))
    
    (loop for b from 0 below batch-size do
      (loop for c from 0 below channels do
        (let ((sum 0.0d0))
          (loop for l from 0 below length do
            (incf sum (aref input-data b c l)))
          (setf (aref output-data b c 0) (/ sum length)))))
    
    (make-tensor output-data
                :shape (list batch-size channels 1)
                :requires-grad (requires-grad input))))

(defmethod forward ((layer global-avg-pool2d) input)
  "Global average pooling for 2D - output shape: (batch, channels, 1, 1)"
  (let* ((input-shape (tensor-shape input))
         (batch-size (first input-shape))
         (channels (second input-shape))
         (height (third input-shape))
         (width (fourth input-shape))
         (spatial-size (* height width))
         (input-data (tensor-data input))
         (output-data (make-array (list batch-size channels 1 1)
                                 :element-type 'double-float)))
    
    (loop for b from 0 below batch-size do
      (loop for c from 0 below channels do
        (let ((sum 0.0d0))
          (loop for h from 0 below height do
            (loop for w from 0 below width do
              (incf sum (aref input-data b c h w))))
          (setf (aref output-data b c 0 0) (/ sum spatial-size)))))
    
    (make-tensor output-data
                :shape (list batch-size channels 1 1)
                :requires-grad (requires-grad input))))

(defmethod forward ((layer global-avg-pool3d) input)
  "Global average pooling for 3D - output shape: (batch, channels, 1, 1, 1)"
  (let* ((input-shape (tensor-shape input))
         (batch-size (first input-shape))
         (channels (second input-shape))
         (depth (third input-shape))
         (height (fourth input-shape))
         (width (fifth input-shape))
         (spatial-size (* depth height width))
         (input-data (tensor-data input))
         (output-data (make-array (list batch-size channels 1 1 1)
                                 :element-type 'double-float)))
    
    (loop for b from 0 below batch-size do
      (loop for c from 0 below channels do
        (let ((sum 0.0d0))
          (loop for d from 0 below depth do
            (loop for h from 0 below height do
              (loop for w from 0 below width do
                (incf sum (aref input-data b c d h w)))))
          (setf (aref output-data b c 0 0 0) (/ sum spatial-size)))))
    
    (make-tensor output-data
                :shape (list batch-size channels 1 1 1)
                :requires-grad (requires-grad input))))

(defmethod forward ((layer global-max-pool1d) input)
  "Global max pooling for 1D - output shape: (batch, channels, 1)"
  (let* ((input-shape (tensor-shape input))
         (batch-size (first input-shape))
         (channels (second input-shape))
         (length (third input-shape))
         (input-data (tensor-data input))
         (output-data (make-array (list batch-size channels 1)
                                 :element-type 'double-float)))
    
    (loop for b from 0 below batch-size do
      (loop for c from 0 below channels do
        (let ((max-val most-negative-double-float))
          (loop for l from 0 below length do
            (setf max-val (max max-val (aref input-data b c l))))
          (setf (aref output-data b c 0) max-val))))
    
    (make-tensor output-data
                :shape (list batch-size channels 1)
                :requires-grad (requires-grad input))))

(defmethod forward ((layer global-max-pool2d) input)
  "Global max pooling for 2D - output shape: (batch, channels, 1, 1)"
  (let* ((input-shape (tensor-shape input))
         (batch-size (first input-shape))
         (channels (second input-shape))
         (height (third input-shape))
         (width (fourth input-shape))
         (input-data (tensor-data input))
         (output-data (make-array (list batch-size channels 1 1)
                                 :element-type 'double-float)))
    
    (loop for b from 0 below batch-size do
      (loop for c from 0 below channels do
        (let ((max-val most-negative-double-float))
          (loop for h from 0 below height do
            (loop for w from 0 below width do
              (setf max-val (max max-val (aref input-data b c h w)))))
          (setf (aref output-data b c 0 0) max-val))))
    
    (make-tensor output-data
                :shape (list batch-size channels 1 1)
                :requires-grad (requires-grad input))))

(defmethod forward ((layer global-max-pool3d) input)
  "Global max pooling for 3D - output shape: (batch, channels, 1, 1, 1)"
  (let* ((input-shape (tensor-shape input))
         (batch-size (first input-shape))
         (channels (second input-shape))
         (depth (third input-shape))
         (height (fourth input-shape))
         (width (fifth input-shape))
         (input-data (tensor-data input))
         (output-data (make-array (list batch-size channels 1 1 1)
                                 :element-type 'double-float)))
    
    (loop for b from 0 below batch-size do
      (loop for c from 0 below channels do
        (let ((max-val most-negative-double-float))
          (loop for d from 0 below depth do
            (loop for h from 0 below height do
              (loop for w from 0 below width do
                (setf max-val (max max-val (aref input-data b c d h w))))))
          (setf (aref output-data b c 0 0 0) max-val))))
    
    (make-tensor output-data
                :shape (list batch-size channels 1 1 1)
                :requires-grad (requires-grad input))))

;;;; ============================================================================
;;;; Spatial Pyramid Pooling (SPP)
;;;; ============================================================================

(defclass spatial-pyramid-pool2d (layer)
  ((pyramid-levels :initarg :pyramid-levels
                   :initform '(1 2 4)
                   :reader pyramid-levels
                   :documentation "List of pyramid levels (e.g., '(1 2 4) for 3-level pyramid)")
   (pool-type :initarg :pool-type
              :initform :max
              :reader pool-type
              :documentation "Type of pooling: :max or :avg"))
  (:documentation "Spatial Pyramid Pooling for 2D data
                   Pools input at multiple scales and concatenates the results.
                   Output is fixed-size regardless of input spatial dimensions.
                   Example: levels '(1 2 4) creates 1x1, 2x2, 4x4 grids = 21 bins per channel"))

(defmethod forward ((layer spatial-pyramid-pool2d) input)
  "Spatial pyramid pooling - creates fixed-size output from variable-size input"
  (let* ((input-shape (tensor-shape input))
         (batch-size (first input-shape))
         (channels (second input-shape))
         (in-h (third input-shape))
         (in-w (fourth input-shape))
         (input-data (tensor-data input))
         (levels (pyramid-levels layer))
         (pool-type (pool-type layer))
         
         ;; Calculate total output features: sum of (level^2) for all levels
         (total-bins (reduce #'+ (mapcar (lambda (l) (* l l)) levels)))
         
         ;; Output is (batch, channels * total-bins)
         (output-data (make-array (list batch-size (* channels total-bins))
                                 :element-type 'double-float
                                 :initial-element 0.0d0)))
    
    ;; Process each batch and channel
    (loop for b from 0 below batch-size do
      (loop for c from 0 below channels do
        (let ((bin-idx 0))
          ;; Process each pyramid level
          (dolist (level levels)
            (let* ((bin-h (/ in-h level))
                   (bin-w (/ in-w level)))
              ;; Pool each bin in this level
              (loop for row from 0 below level do
                (loop for col from 0 below level do
                  (let ((start-h (floor (* row bin-h)))
                        (end-h (floor (* (1+ row) bin-h)))
                        (start-w (floor (* col bin-w)))
                        (end-w (floor (* (1+ col) bin-w)))
                        (pool-val (if (eq pool-type :max)
                                    most-negative-double-float
                                    0.0d0))
                        (count 0))
                    
                    ;; Pool within this bin
                    (loop for h from start-h below end-h do
                      (loop for w from start-w below end-w do
                        (when (and (< h in-h) (< w in-w))
                          (let ((val (aref input-data b c h w)))
                            (ecase pool-type
                              (:max (setf pool-val (max pool-val val)))
                              (:avg (incf pool-val val)
                                   (incf count)))))))
                    
                    ;; Average if needed
                    (when (and (eq pool-type :avg) (> count 0))
                      (setf pool-val (/ pool-val count)))
                    
                    ;; Store result
                    (setf (aref output-data b (+ (* c total-bins) bin-idx)) 
                          pool-val)
                    (incf bin-idx)))))))))
    
    (make-tensor output-data
                :shape (list batch-size (* channels total-bins))
                :requires-grad (requires-grad input))))

;;;; ============================================================================
;;;; Transposed Convolutions (Deconvolutions)
;;;; ============================================================================

(defclass conv-transpose2d (conv-layer)
  ((output-padding :initarg :output-padding
                   :initform 0
                   :reader output-padding
                   :documentation "Additional padding for output"))
  (:documentation "2D Transposed convolution (deconvolution) layer"))

(defmethod initialize-instance :after ((layer conv-transpose2d) &key)
  (with-slots (kernel bias in-channels out-channels kernel-size stride padding 
               dilation output-padding groups use-bias) layer
    
    ;; Ensure parameters are lists
    (setf kernel-size (ensure-list kernel-size 2))
    (setf stride (ensure-list stride 2))
    (setf padding (ensure-list padding 2))
    (setf dilation (ensure-list dilation 2))
    (setf output-padding (ensure-list output-padding 2))
    
    ;; Initialize weights (note: in-channels and out-channels are swapped for transpose conv)
    (let* ((kh (first kernel-size))
           (kw (second kernel-size))
           (fan-in (* out-channels kh kw))
           (std (sqrt (/ 2.0d0 fan-in))))
      (setf kernel (randn (list in-channels (/ out-channels groups) kh kw)
                         :requires-grad t
                         :scale std
                         :name "conv-transpose2d-kernel")))
    
    ;; Initialize bias
    (when use-bias
      (setf bias (zeros (list out-channels)
                       :requires-grad t
                       :name "conv-transpose2d-bias")))
    
    ;; Register parameters
    (setf (slot-value layer 'neural-network:parameters)
          (if use-bias
              (list kernel bias)
              (list kernel)))))

(defmethod forward ((layer conv-transpose2d) input)
  "Forward pass for 2D transposed convolution"
  (with-slots (kernel bias in-channels out-channels kernel-size stride padding 
               dilation output-padding groups use-bias) layer
    
    (let* ((input-shape (tensor-shape input))
           (batch-size (first input-shape))
           (in-h (third input-shape))
           (in-w (fourth input-shape))
           (kh (first kernel-size))
           (kw (second kernel-size))
           (sh (first stride))
           (sw (second stride))
           (ph (first padding))
           (pw (second padding))
           (oph (first output-padding))
           (opw (second output-padding))
           
           ;; Calculate output dimensions for transposed convolution
           (out-h (+ (* (- in-h 1) sh) kh (- (* 2 ph)) oph))
           (out-w (+ (* (- in-w 1) sw) kw (- (* 2 pw)) opw))
           
           ;; Create output tensor
           (output-data (make-array (list batch-size out-channels out-h out-w)
                                   :element-type 'double-float
                                   :initial-element 0.0d0))
           (input-data (tensor-data input))
           (kernel-data (tensor-data kernel)))
      
      ;; Perform transposed convolution
      (loop for b from 0 below batch-size do
        (loop for ic from 0 below in-channels do
          (loop for ih from 0 below in-h do
            (loop for iw from 0 below in-w do
              (let ((in-val (aref input-data b ic ih iw)))
                ;; Distribute input value to output via kernel
                (loop for oc from 0 below (/ out-channels groups) do
                  (loop for kh-idx from 0 below kh do
                    (loop for kw-idx from 0 below kw do
                      (let ((out-h-idx (+ (* ih sh) kh-idx (- ph)))
                            (out-w-idx (+ (* iw sw) kw-idx (- pw))))
                        (when (and (>= out-h-idx 0) (< out-h-idx out-h)
                                  (>= out-w-idx 0) (< out-w-idx out-w))
                          (incf (aref output-data b oc out-h-idx out-w-idx)
                                (* in-val (aref kernel-data ic oc kh-idx kw-idx)))))))))))))
      
      ;; Add bias
      (when use-bias
        (let ((bias-data (tensor-data bias)))
          (loop for b from 0 below batch-size do
            (loop for oc from 0 below out-channels do
              (loop for oh from 0 below out-h do
                (loop for ow from 0 below out-w do
                  (incf (aref output-data b oc oh ow)
                        (aref bias-data oc))))))))
      
      (make-tensor output-data
                  :shape (list batch-size out-channels out-h out-w)
                  :requires-grad (or (requires-grad input)
                                   (requires-grad kernel))))))

(defclass conv-transpose1d (conv-layer)
  ((output-padding :initarg :output-padding
                   :initform 0
                   :reader output-padding))
  (:documentation "1D Transposed convolution layer"))

(defclass conv-transpose3d (conv-layer)
  ((output-padding :initarg :output-padding
                   :initform 0
                   :reader output-padding))
  (:documentation "3D Transposed convolution layer"))

;;;; ============================================================================
;;;; Separable Convolutions
;;;; ============================================================================

(defclass depthwise-conv2d (conv2d)
  ()
  (:documentation "Depthwise 2D convolution (each channel convolved separately)"))

(defmethod initialize-instance :after ((layer depthwise-conv2d) &key)
  (with-slots (kernel bias in-channels out-channels kernel-size stride padding 
               dilation groups use-bias) layer
    
    ;; For depthwise, groups = in-channels and out-channels = in-channels
    (assert (= groups in-channels))
    (assert (= out-channels in-channels))
    
    (setf kernel-size (ensure-list kernel-size 2))
    (setf stride (ensure-list stride 2))
    (setf padding (ensure-list padding 2))
    (setf dilation (ensure-list dilation 2))
    
    ;; Initialize weights - one kernel per input channel
    (let* ((kh (first kernel-size))
           (kw (second kernel-size))
           (fan-in (* kh kw))
           (std (sqrt (/ 2.0d0 fan-in))))
      (setf kernel (randn (list in-channels 1 kh kw)
                         :requires-grad t
                         :scale std
                         :name "depthwise-conv2d-kernel")))
    
    (when use-bias
      (setf bias (zeros (list in-channels)
                       :requires-grad t
                       :name "depthwise-conv2d-bias")))
    
    (setf (slot-value layer 'neural-network:parameters)
          (if use-bias
              (list kernel bias)
              (list kernel)))))

(defclass separable-conv2d (layer)
  ((depthwise :accessor depthwise
              :documentation "Depthwise convolution layer")
   (pointwise :accessor pointwise
              :documentation "Pointwise (1x1) convolution layer"))
  (:documentation "Separable 2D convolution (depthwise + pointwise)"))

(defmethod initialize-instance :after ((layer separable-conv2d) 
                                       &key in-channels out-channels 
                                       kernel-size stride padding dilation use-bias)
  (with-slots (depthwise pointwise) layer
    
    ;; Create depthwise layer
    (setf depthwise (make-instance 'depthwise-conv2d
                                  :in-channels in-channels
                                  :out-channels in-channels
                                  :kernel-size kernel-size
                                  :stride stride
                                  :padding padding
                                  :dilation dilation
                                  :groups in-channels
                                  :use-bias use-bias))
    
    ;; Create pointwise (1x1 conv) layer
    (setf pointwise (make-instance 'conv2d
                                  :in-channels in-channels
                                  :out-channels out-channels
                                  :kernel-size 1
                                  :stride 1
                                  :padding 0
                                  :use-bias use-bias))
    
    ;; Combine parameters
    (setf (slot-value layer 'neural-network:parameters)
          (append (layer-parameters depthwise)
                  (layer-parameters pointwise)))))

(defmethod forward ((layer separable-conv2d) input)
  "Forward pass: depthwise convolution followed by pointwise convolution"
  (with-slots (depthwise pointwise) layer
    (let ((depthwise-output (forward depthwise input)))
      (forward pointwise depthwise-output))))

;;;; ============================================================================
;;;; Dilated/Atrous Convolutions
;;;; ============================================================================

(defclass dilated-conv2d (conv2d)
  ()
  (:documentation "2D Dilated (Atrous) convolution with expanded receptive field"))

;; Dilated convolution is already handled by the dilation parameter in conv2d
;; This is just a convenience class

;;;; ============================================================================
;;;; Constructor Functions
;;;; ============================================================================

(defun conv1d (in-channels out-channels kernel-size 
               &key (stride 1) (padding 0) (dilation 1) (groups 1) (use-bias t))
  "Create a 1D convolutional layer"
  (make-instance 'conv1d
                 :in-channels in-channels
                 :out-channels out-channels
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding
                 :dilation dilation
                 :groups groups
                 :use-bias use-bias))

(defun conv2d (in-channels out-channels kernel-size 
               &key (stride 1) (padding 0) (dilation 1) (groups 1) (use-bias t))
  "Create a 2D convolutional layer"
  (make-instance 'conv2d
                 :in-channels in-channels
                 :out-channels out-channels
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding
                 :dilation dilation
                 :groups groups
                 :use-bias use-bias))

(defun conv3d (in-channels out-channels kernel-size 
               &key (stride 1) (padding 0) (dilation 1) (groups 1) (use-bias t))
  "Create a 3D convolutional layer"
  (make-instance 'conv3d
                 :in-channels in-channels
                 :out-channels out-channels
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding
                 :dilation dilation
                 :groups groups
                 :use-bias use-bias))

(defun convnd (ndims in-channels out-channels kernel-size 
               &key (stride 1) (padding 0) (dilation 1) (groups 1) (use-bias t))
  "Create an N-dimensional convolutional layer"
  (make-instance 'convnd
                 :ndims ndims
                 :in-channels in-channels
                 :out-channels out-channels
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding
                 :dilation dilation
                 :groups groups
                 :use-bias use-bias))

(defun max-pool1d (kernel-size &key stride (padding 0))
  "Create a 1D max pooling layer"
  (make-instance 'max-pool1d
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding))

(defun max-pool2d (kernel-size &key stride (padding 0))
  "Create a 2D max pooling layer"
  (make-instance 'max-pool2d
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding))

(defun max-pool3d (kernel-size &key stride (padding 0))
  "Create a 3D max pooling layer"
  (make-instance 'max-pool3d
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding))

(defun avg-pool1d (kernel-size &key stride (padding 0))
  "Create a 1D average pooling layer"
  (make-instance 'avg-pool1d
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding))

(defun avg-pool2d (kernel-size &key stride (padding 0))
  "Create a 2D average pooling layer"
  (make-instance 'avg-pool2d
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding))

(defun avg-pool3d (kernel-size &key stride (padding 0))
  "Create a 3D average pooling layer"
  (make-instance 'avg-pool3d
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding))

(defun conv-transpose1d (in-channels out-channels kernel-size 
                        &key (stride 1) (padding 0) (output-padding 0) 
                        (dilation 1) (groups 1) (use-bias t))
  "Create a 1D transposed convolutional layer"
  (make-instance 'conv-transpose1d
                 :in-channels in-channels
                 :out-channels out-channels
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding
                 :output-padding output-padding
                 :dilation dilation
                 :groups groups
                 :use-bias use-bias))

(defun conv-transpose2d (in-channels out-channels kernel-size 
                        &key (stride 1) (padding 0) (output-padding 0) 
                        (dilation 1) (groups 1) (use-bias t))
  "Create a 2D transposed convolutional layer"
  (make-instance 'conv-transpose2d
                 :in-channels in-channels
                 :out-channels out-channels
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding
                 :output-padding output-padding
                 :dilation dilation
                 :groups groups
                 :use-bias use-bias))

(defun conv-transpose3d (in-channels out-channels kernel-size 
                        &key (stride 1) (padding 0) (output-padding 0) 
                        (dilation 1) (groups 1) (use-bias t))
  "Create a 3D transposed convolutional layer"
  (make-instance 'conv-transpose3d
                 :in-channels in-channels
                 :out-channels out-channels
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding
                 :output-padding output-padding
                 :dilation dilation
                 :groups groups
                 :use-bias use-bias))

(defun separable-conv2d (in-channels out-channels kernel-size 
                        &key (stride 1) (padding 0) (dilation 1) (use-bias t))
  "Create a separable 2D convolutional layer"
  (make-instance 'separable-conv2d
                 :in-channels in-channels
                 :out-channels out-channels
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding
                 :dilation dilation
                 :use-bias use-bias))

(defun depthwise-conv2d (in-channels kernel-size 
                        &key (stride 1) (padding 0) (dilation 1) (use-bias t))
  "Create a depthwise 2D convolutional layer"
  (make-instance 'depthwise-conv2d
                 :in-channels in-channels
                 :out-channels in-channels
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding
                 :dilation dilation
                 :groups in-channels
                 :use-bias use-bias))

(defun dilated-conv2d (in-channels out-channels kernel-size dilation
                      &key (stride 1) (padding 0) (groups 1) (use-bias t))
  "Create a dilated 2D convolutional layer"
  (make-instance 'dilated-conv2d
                 :in-channels in-channels
                 :out-channels out-channels
                 :kernel-size kernel-size
                 :stride stride
                 :padding padding
                 :dilation dilation
                 :groups groups
                 :use-bias use-bias))

;;;; ============================================================================
;;;; Demo Function
;;;; ============================================================================

(defun demo-convolutions ()
  "Demonstrate various convolutional layers"
  (format t "~%╔══════════════════════════════════════════════════════════════╗~%")
  (format t "║           Convolutional Neural Networks - Demo               ║~%")
  (format t "╚══════════════════════════════════════════════════════════════╝~%")
  
  (format t "~%1. Conv1D - For sequential data (audio, text):~%")
  (format t "   Input: (batch=2, channels=3, length=10)~%")
  (format t "   Kernel: 3, Stride: 1, Padding: 1~%")
  
  (format t "~%2. Conv2D - For images:~%")
  (format t "   Input: (batch=2, channels=3, height=28, width=28)~%")
  (format t "   Kernel: (3,3), Stride: 1, Padding: 1~%")
  
  (format t "~%3. Conv3D - For videos/volumetric data:~%")
  (format t "   Input: (batch=1, channels=3, depth=16, height=28, width=28)~%")
  (format t "   Kernel: (3,3,3), Stride: 1~%")
  
  (format t "~%4. ConvND - Generalized N-dimensional convolution:~%")
  (format t "   Works for any number of spatial dimensions!~%")
  
  (format t "~%5. Max/Average Pooling (1D, 2D, 3D):~%")
  (format t "   Downsample feature maps while retaining important features~%")
  
  (format t "~%6. Transposed Convolutions (Deconvolutions):~%")
  (format t "   Upsample feature maps (used in GANs, autoencoders)~%")
  
  (format t "~%7. Separable Convolutions:~%")
  (format t "   Efficient: depthwise + pointwise = fewer parameters~%")
  
  (format t "~%8. Dilated/Atrous Convolutions:~%")
  (format t "   Expanded receptive field without increasing parameters~%")
  
  (format t "~%~%All convolution types support:~%")
  (format t "  - Custom stride and padding~%")
  (format t "  - Dilation for expanded receptive fields~%")
  (format t "  - Grouped convolutions~%")
  (format t "  - Automatic gradient computation~%")
  (format t "~%"))
