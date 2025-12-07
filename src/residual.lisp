;;;; Neural Tensor Library - Residual Blocks
;;;; High-level abstractions for ResNet, EfficientNet, and ConvNeXt architectures
;;;; Demonstrating Lisp's ability to build complex neural network components

(defpackage :neural-tensor-residual
  (:use :common-lisp)
  (:import-from :neural-network
                #:tensor
                #:tensor-data
                #:tensor-shape
                #:make-tensor
                #:zeros
                #:ones
                #:randn
                #:forward
                #:layer
                #:sequential
                #:linear
                #:t+
                #:t*)
  (:import-from :neural-tensor-activations
                #:relu
                #:sigmoid)
  (:import-from :neural-tensor-convolution
                #:conv2d
                #:conv-kernel
                #:global-avg-pool2d
                #:depthwise-conv2d)
  (:import-from :normalization
                #:batch-norm-layer
                #:layer-norm-layer)
  (:export ;; ResNet blocks
           #:resnet-basic-block
           #:resnet-bottleneck-block
           #:resnet-downsample
           ;; EfficientNet blocks
           #:mbconv-block
           #:squeeze-excitation
           #:fused-mbconv-block
           ;; ConvNeXt blocks
           #:convnext-block
           #:convnext-stage
           ;; Utility layers
           #:stochastic-depth
           #:layer-scale))

(in-package :neural-tensor-residual)

;;;; ============================================================================
;;;; ResNet Blocks (He et al., 2016)
;;;; ============================================================================

(defclass resnet-basic-block (layer)
  ((in-channels :initarg :in-channels
                :reader in-channels)
   (out-channels :initarg :out-channels
                 :reader out-channels)
   (stride :initarg :stride
           :initform 1
           :reader stride)
   (downsample :initarg :downsample
               :initform nil
               :accessor downsample)
   (conv1 :accessor conv1)
   (bn1 :accessor bn1)
   (conv2 :accessor conv2)
   (bn2 :accessor bn2))
  (:documentation "ResNet Basic Block with two 3x3 convolutions
                   Used in ResNet-18 and ResNet-34
                   Architecture: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+skip) -> ReLU"))

(defmethod initialize-instance :after ((block resnet-basic-block) &key)
  (with-slots (in-channels out-channels stride conv1 bn1 conv2 bn2 downsample) block
    ;; First convolution
    (setf conv1 (make-instance 'neural-tensor-convolution:conv2d
                              :in-channels in-channels
                              :out-channels out-channels
                              :kernel-size 3
                              :stride stride
                              :padding 1
                              :use-bias nil))
    (setf bn1 (make-instance 'normalization:batch-norm-layer
                            :num-features out-channels))
    
    ;; Second convolution
    (setf conv2 (make-instance 'neural-tensor-convolution:conv2d
                              :in-channels out-channels
                              :out-channels out-channels
                              :kernel-size 3
                              :stride 1
                              :padding 1
                              :use-bias nil))
    (setf bn2 (make-instance 'normalization:batch-norm-layer
                            :num-features out-channels))
    
    ;; Downsample if stride != 1 or channels change
    (when (or (/= stride 1) (/= in-channels out-channels))
      (setf downsample (make-instance 'resnet-downsample
                                     :in-channels in-channels
                                     :out-channels out-channels
                                     :stride stride)))
    
    ;; Register parameters
    (setf (slot-value block 'neural-network:parameters)
          (append (neural-network:layer-parameters conv1)
                  (neural-network:layer-parameters bn1)
                  (neural-network:layer-parameters conv2)
                  (neural-network:layer-parameters bn2)
                  (when downsample
                    (neural-network:layer-parameters downsample))))))

(defmethod forward ((block resnet-basic-block) input)
  "Forward pass: Conv -> BN -> ReLU -> Conv -> BN -> Add(skip) -> ReLU"
  (with-slots (conv1 bn1 conv2 bn2 downsample) block
    (let ((identity input))
      ;; Main path
      (let ((out (forward conv1 input)))
        (setf out (forward bn1 out))
        (setf out (relu out))
        (setf out (forward conv2 out))
        (setf out (forward bn2 out))
        
        ;; Skip connection
        (when downsample
          (setf identity (forward downsample identity)))
        
        ;; Add and activate
        (setf out (t+ out identity))
        (relu out)))))

;;;; ----------------------------------------------------------------------------

(defclass resnet-bottleneck-block (layer)
  ((in-channels :initarg :in-channels
                :reader in-channels)
   (out-channels :initarg :out-channels
                 :reader out-channels)
   (stride :initarg :stride
           :initform 1
           :reader stride)
   (expansion :initform 4
              :reader expansion
              :documentation "Channel expansion factor (bottleneck -> output)")
   (downsample :initarg :downsample
               :initform nil
               :accessor downsample)
   (conv1 :accessor conv1)
   (bn1 :accessor bn1)
   (conv2 :accessor conv2)
   (bn2 :accessor bn2)
   (conv3 :accessor conv3)
   (bn3 :accessor bn3))
  (:documentation "ResNet Bottleneck Block with 1x1 -> 3x3 -> 1x1 convolutions
                   Used in ResNet-50, ResNet-101, ResNet-152
                   Architecture: Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN -> (+skip) -> ReLU
                   Reduces then expands channels (bottleneck)"))

(defmethod initialize-instance :after ((block resnet-bottleneck-block) &key)
  (with-slots (in-channels out-channels stride expansion conv1 bn1 conv2 bn2 conv3 bn3 downsample) block
    (let ((bottleneck-channels out-channels))
      ;; 1x1 conv to reduce channels
      (setf conv1 (make-instance 'neural-tensor-convolution:conv2d
                                :in-channels in-channels
                                :out-channels bottleneck-channels
                                :kernel-size 1
                                :stride 1
                                :use-bias nil))
      (setf bn1 (make-instance 'normalization:batch-norm-layer
                              :num-features bottleneck-channels))
      
      ;; 3x3 conv
      (setf conv2 (make-instance 'neural-tensor-convolution:conv2d
                                :in-channels bottleneck-channels
                                :out-channels bottleneck-channels
                                :kernel-size 3
                                :stride stride
                                :padding 1
                                :use-bias nil))
      (setf bn2 (make-instance 'normalization:batch-norm-layer
                              :num-features bottleneck-channels))
      
      ;; 1x1 conv to expand channels
      (setf conv3 (make-instance 'neural-tensor-convolution:conv2d
                                :in-channels bottleneck-channels
                                :out-channels (* bottleneck-channels expansion)
                                :kernel-size 1
                                :stride 1
                                :use-bias nil))
      (setf bn3 (make-instance 'normalization:batch-norm-layer
                              :num-features (* bottleneck-channels expansion)))
      
      ;; Downsample if needed
      (when (or (/= stride 1) (/= in-channels (* out-channels expansion)))
        (setf downsample (make-instance 'resnet-downsample
                                       :in-channels in-channels
                                       :out-channels (* out-channels expansion)
                                       :stride stride)))
      
      ;; Register parameters
      (setf (slot-value block 'neural-network:parameters)
            (append (neural-network:layer-parameters conv1)
                    (neural-network:layer-parameters bn1)
                    (neural-network:layer-parameters conv2)
                    (neural-network:layer-parameters bn2)
                    (neural-network:layer-parameters conv3)
                    (neural-network:layer-parameters bn3)
                    (when downsample
                      (neural-network:layer-parameters downsample)))))))

(defmethod forward ((block resnet-bottleneck-block) input)
  "Forward pass through bottleneck block"
  (with-slots (conv1 bn1 conv2 bn2 conv3 bn3 downsample) block
    (let ((identity input))
      ;; Main path: 1x1 -> 3x3 -> 1x1
      (let ((out (forward conv1 input)))
        (setf out (forward bn1 out))
        (setf out (relu out))
        
        (setf out (forward conv2 out))
        (setf out (forward bn2 out))
        (setf out (relu out))
        
        (setf out (forward conv3 out))
        (setf out (forward bn3 out))
        
        ;; Skip connection
        (when downsample
          (setf identity (forward downsample identity)))
        
        ;; Add and activate
        (setf out (t+ out identity))
        (relu out)))))

;;;; ----------------------------------------------------------------------------

(defclass resnet-downsample (layer)
  ((in-channels :initarg :in-channels
                :reader in-channels)
   (out-channels :initarg :out-channels
                 :reader out-channels)
   (stride :initarg :stride
           :reader stride)
   (conv :accessor downsample-conv)
   (bn :accessor downsample-bn))
  (:documentation "Downsampling layer for ResNet skip connections"))

(defmethod initialize-instance :after ((layer resnet-downsample) &key)
  (with-slots (in-channels out-channels stride conv bn) layer
    (setf conv (make-instance 'neural-tensor-convolution:conv2d
                             :in-channels in-channels
                             :out-channels out-channels
                             :kernel-size 1
                             :stride stride
                             :use-bias nil))
    (setf bn (make-instance 'normalization:batch-norm-layer
                           :num-features out-channels))
    
    (setf (slot-value layer 'neural-network:parameters)
          (append (neural-network:layer-parameters conv)
                  (neural-network:layer-parameters bn)))))

(defmethod forward ((layer resnet-downsample) input)
  (with-slots (conv bn) layer
    (let ((out (forward conv input)))
      (forward bn out))))

;;;; ============================================================================
;;;; Squeeze-and-Excitation (Hu et al., 2018)
;;;; ============================================================================

(defclass squeeze-excitation (layer)
  ((channels :initarg :channels
             :reader channels)
   (reduction :initarg :reduction
              :initform 4
              :reader reduction
              :documentation "Channel reduction ratio")
   (pool :accessor se-pool)
   (fc1 :accessor se-fc1)
   (fc2 :accessor se-fc2))
  (:documentation "Squeeze-and-Excitation block for channel attention
                   Architecture: GlobalAvgPool -> FC -> ReLU -> FC -> Sigmoid -> Scale"))

(defmethod initialize-instance :after ((layer squeeze-excitation) &key)
  (with-slots (channels reduction pool fc1 fc2) layer
    (let ((reduced-channels (max 1 (floor channels reduction))))
      ;; Global average pooling
      (setf pool (make-instance 'neural-tensor-convolution:global-avg-pool2d))
      
      ;; Two FC layers for channel attention
      (setf fc1 (make-instance 'neural-network:linear-layer
                              :in-features channels
                              :out-features reduced-channels))
      (setf fc2 (make-instance 'neural-network:linear-layer
                              :in-features reduced-channels
                              :out-features channels))
      
      (setf (slot-value layer 'neural-network:parameters)
            (append (neural-network:layer-parameters fc1)
                    (neural-network:layer-parameters fc2))))))

(defmethod forward ((layer squeeze-excitation) input)
  "Apply channel-wise attention"
  (with-slots (pool fc1 fc2 channels) layer
    (let* ((input-shape (tensor-shape input))
           (batch-size (first input-shape))
           ;; Squeeze: Global average pooling
           (se (forward pool input))
           ;; Flatten for FC layers
           (se-data (tensor-data se))
           (se-flat (make-array (list batch-size channels)
                               :element-type 'double-float)))
      
      ;; Copy pooled values
      (loop for b from 0 below batch-size do
        (loop for c from 0 below channels do
          (setf (aref se-flat b c) (aref se-data b c 0 0))))
      
      (let ((se-tensor (make-tensor se-flat :shape (list batch-size channels))))
        ;; Excitation: FC -> ReLU -> FC -> Sigmoid
        (setf se-tensor (forward fc1 se-tensor))
        (setf se-tensor (relu se-tensor))
        (setf se-tensor (forward fc2 se-tensor))
        (setf se-tensor (sigmoid se-tensor))
        
        ;; Reshape and scale input
        (let* ((se-weights (tensor-data se-tensor))
               (output-data (make-array input-shape
                                       :element-type 'double-float))
               (input-data (tensor-data input))
               (height (third input-shape))
               (width (fourth input-shape)))
          
          (loop for b from 0 below batch-size do
            (loop for c from 0 below channels do
              (let ((weight (aref se-weights b c)))
                (loop for h from 0 below height do
                  (loop for w from 0 below width do
                    (setf (aref output-data b c h w)
                          (* (aref input-data b c h w) weight)))))))
          
          (make-tensor output-data :shape input-shape
                      :requires-grad (neural-network:requires-grad input)))))))

;;;; ============================================================================
;;;; EfficientNet MBConv Blocks (Tan & Le, 2019)
;;;; ============================================================================

(defclass mbconv-block (layer)
  ((in-channels :initarg :in-channels
                :reader in-channels)
   (out-channels :initarg :out-channels
                 :reader out-channels)
   (expand-ratio :initarg :expand-ratio
                 :initform 6
                 :reader expand-ratio
                 :documentation "Channel expansion ratio")
   (kernel-size :initarg :kernel-size
                :initform 3
                :reader kernel-size)
   (stride :initarg :stride
           :initform 1
           :reader stride)
   (use-se :initarg :use-se
           :initform t
           :reader use-se
           :documentation "Whether to use squeeze-excitation")
   (expand-conv :accessor expand-conv
                :initform nil)
   (expand-bn :accessor expand-bn
              :initform nil)
   (depthwise-conv :accessor mbconv-depthwise-conv)
   (depthwise-bn :accessor depthwise-bn)
   (se :accessor mbconv-se
       :initform nil)
   (project-conv :accessor project-conv)
   (project-bn :accessor project-bn)
   (use-skip :accessor use-skip))
  (:documentation "Mobile Inverted Bottleneck Convolution (MBConv) block
                   Used in EfficientNet and MobileNetV2
                   Architecture: Expand -> DWConv -> SE -> Project -> (+skip if applicable)"))

(defmethod initialize-instance :after ((block mbconv-block) &key)
  (with-slots (in-channels out-channels expand-ratio kernel-size stride use-se
               expand-conv expand-bn depthwise-conv depthwise-bn se project-conv project-bn use-skip) block
    (let ((expanded-channels (* in-channels expand-ratio)))
      
      ;; Expansion phase (skip if expand-ratio = 1)
      (when (> expand-ratio 1)
        (setf expand-conv (make-instance 'neural-tensor-convolution:conv2d
                                        :in-channels in-channels
                                        :out-channels expanded-channels
                                        :kernel-size 1
                                        :use-bias nil))
        (setf expand-bn (make-instance 'normalization:batch-norm-layer
                                      :num-features expanded-channels)))
      
      ;; Depthwise convolution
      (setf depthwise-conv (make-instance 'neural-tensor-convolution:depthwise-conv2d
                                         :in-channels expanded-channels
                                         :out-channels expanded-channels
                                         :kernel-size kernel-size
                                         :stride stride
                                         :padding (floor kernel-size 2)
                                         :groups expanded-channels))
      (setf depthwise-bn (make-instance 'normalization:batch-norm-layer
                                       :num-features expanded-channels))
      
      ;; Squeeze-and-Excitation
      (when use-se
        (setf se (make-instance 'squeeze-excitation
                               :channels expanded-channels
                               :reduction 4)))
      
      ;; Projection phase
      (setf project-conv (make-instance 'neural-tensor-convolution:conv2d
                                       :in-channels expanded-channels
                                       :out-channels out-channels
                                       :kernel-size 1
                                       :use-bias nil))
      (setf project-bn (make-instance 'normalization:batch-norm-layer
                                     :num-features out-channels))
      
      ;; Use skip connection if stride=1 and channels match
      (setf use-skip (and (= stride 1) (= in-channels out-channels)))
      
      ;; Register parameters
      (setf (slot-value block 'neural-network:parameters)
            (append (when expand-conv
                      (append (neural-network:layer-parameters expand-conv)
                              (neural-network:layer-parameters expand-bn)))
                    (neural-network:layer-parameters depthwise-conv)
                    (neural-network:layer-parameters depthwise-bn)
                    (when use-se
                      (neural-network:layer-parameters se))
                    (neural-network:layer-parameters project-conv)
                    (neural-network:layer-parameters project-bn))))))

(defmethod forward ((block mbconv-block) input)
  "Forward pass through MBConv block"
  (with-slots (expand-ratio expand-conv expand-bn depthwise-conv depthwise-bn 
               se project-conv project-bn use-skip) block
    (let ((out input))
      
      ;; Expansion
      (when (> expand-ratio 1)
        (setf out (forward expand-conv out))
        (setf out (forward expand-bn out))
        (setf out (relu out)))
      
      ;; Depthwise convolution
      (setf out (forward depthwise-conv out))
      (setf out (forward depthwise-bn out))
      (setf out (relu out))
      
      ;; Squeeze-and-Excitation
      (when se
        (setf out (forward se out)))
      
      ;; Projection
      (setf out (forward project-conv out))
      (setf out (forward project-bn out))
      
      ;; Skip connection
      (if use-skip
          (t+ out input)
          out))))

;;;; ============================================================================
;;;; ConvNeXt Blocks (Liu et al., 2022)
;;;; ============================================================================

(defclass convnext-block (layer)
  ((dim :initarg :dim
        :reader dim
        :documentation "Number of channels")
   (layer-scale-init :initarg :layer-scale-init
                     :initform 1.0d-6
                     :reader layer-scale-init)
   (dwconv :accessor convnext-dwconv)
   (norm :accessor convnext-norm)
   (pwconv1 :accessor convnext-pwconv1)
   (pwconv2 :accessor convnext-pwconv2)
   (gamma :accessor convnext-gamma))
  (:documentation "ConvNeXt Block - modernized ResNet architecture
                   Architecture: DWConv7x7 -> LayerNorm -> Linear(4x) -> GELU -> Linear -> LayerScale -> (+skip)
                   Key innovations: Large kernels, LayerNorm, inverted bottleneck"))

(defmethod initialize-instance :after ((block convnext-block) &key)
  (with-slots (dim layer-scale-init dwconv norm pwconv1 pwconv2 gamma) block
    ;; Depthwise conv 7x7
    (setf dwconv (make-instance 'neural-tensor-convolution:depthwise-conv2d
                               :in-channels dim
                               :out-channels dim
                               :kernel-size 7
                               :stride 1
                               :padding 3
                               :groups dim))
    
    ;; LayerNorm
    (setf norm (make-instance 'normalization:layer-norm-layer
                             :normalized-shape dim))
    
    ;; Pointwise/1x1 convolutions (inverted bottleneck: expand then compress)
    (setf pwconv1 (make-instance 'neural-tensor-convolution:conv2d
                                :in-channels dim
                                :out-channels (* 4 dim)
                                :kernel-size 1
                                :stride 1
                                :use-bias t))
    (setf pwconv2 (make-instance 'neural-tensor-convolution:conv2d
                                :in-channels (* 4 dim)
                                :out-channels dim
                                :kernel-size 1
                                :stride 1
                                :use-bias t))
    
    ;; Layer scale parameter
    (setf gamma (make-tensor (make-array (list dim)
                                        :element-type 'double-float
                                        :initial-element layer-scale-init)
                            :shape (list dim)
                            :requires-grad t
                            :name "convnext-layer-scale"))
    
    ;; Register parameters
    (setf (slot-value block 'neural-network:parameters)
          (append (neural-network:layer-parameters dwconv)
                  (neural-network:layer-parameters norm)
                  (neural-network:layer-parameters pwconv1)
                  (neural-network:layer-parameters pwconv2)
                  (list gamma)))))

(defmethod forward ((block convnext-block) input)
  "Forward pass through ConvNeXt block"
  (with-slots (dwconv norm pwconv1 pwconv2 gamma dim) block
    (let ((identity input))
      ;; Depthwise convolution (B, C, H, W) -> (B, C, H, W)
      (let* ((out (forward dwconv input))
             (shape (tensor-shape out))
             (batch-size (first shape))
             (channels (second shape))
             (height (third shape))
             (width (fourth shape)))
        
        ;; Permute for LayerNorm: (B, C, H, W) -> (B, H, W, C)
        (let* ((data (tensor-data out))
               (permuted (make-array (list batch-size height width channels)
                                    :element-type 'double-float)))
          
          (loop for b from 0 below batch-size do
            (loop for h from 0 below height do
              (loop for w from 0 below width do
                (loop for c from 0 below channels do
                  (setf (aref permuted b h w c) (aref data b c h w))))))
          
          (setf out (make-tensor permuted 
                                :shape (list batch-size height width channels)
                                :requires-grad (neural-network:requires-grad out)))
          
          ;; LayerNorm on last dimension
          (setf out (forward norm out))
          
          ;; Permute back to (B, C, H, W) for conv2d layers
          (let ((repermuted (make-array (list batch-size channels height width)
                                       :element-type 'double-float)))
            (loop for b from 0 below batch-size do
              (loop for h from 0 below height do
                (loop for w from 0 below width do
                  (loop for c from 0 below channels do
                    (setf (aref repermuted b c h w)
                          (aref (tensor-data out) b h w c))))))
            
            (setf out (make-tensor repermuted
                                  :shape (list batch-size channels height width)
                                  :requires-grad (neural-network:requires-grad out))))
          
          ;; Pointwise convolutions and GELU
          (setf out (forward pwconv1 out))
          (setf out (neural-tensor-activations:gelu out))
          (setf out (forward pwconv2 out))
          
          ;; Apply layer scale
          (let* ((gamma-data (tensor-data gamma))
                 (out-data (tensor-data out))
                 (scaled (make-array (list batch-size channels height width)
                                    :element-type 'double-float)))
            (loop for b from 0 below batch-size do
              (loop for c from 0 below channels do
                (loop for h from 0 below height do
                  (loop for w from 0 below width do
                    (setf (aref scaled b c h w)
                          (* (aref out-data b c h w) (aref gamma-data c)))))))
            
            (setf out (make-tensor scaled
                                  :shape (list batch-size channels height width)
                                  :requires-grad (neural-network:requires-grad out))))
          
          ;; Residual connection
          (t+ out identity))))))

;;;; ============================================================================
;;;; Utility Layers
;;;; ============================================================================

(defclass stochastic-depth (layer)
  ((drop-prob :initarg :drop-prob
              :initform 0.1
              :reader drop-prob
              :documentation "Probability of dropping the residual connection")
   (training :initform t
             :accessor layer-training
             :documentation "Whether layer is in training mode"))
  (:documentation "Stochastic Depth (Drop Path) for regularization
                   Randomly drops entire residual branches during training
                   
                   Implementation follows 'Deep Networks with Stochastic Depth' (Huang et al., 2016)
                   During training: randomly drops the path with probability drop-prob
                   During inference: uses deterministic scaling by (1 - drop-prob)"))

(defmethod forward ((layer stochastic-depth) input)
  "Apply stochastic depth with proper train/eval behavior
   
   Training mode:
     - With probability drop-prob: returns zeros (drops the path)
     - With probability (1 - drop-prob): returns input scaled by 1/(1 - drop-prob)
   
   Eval mode:
     - Always returns input (deterministic, no dropout)"
  (let ((drop-prob (drop-prob layer)))
    (cond
      ;; No dropout case
      ((<= drop-prob 0.0) input)
      
      ;; Eval mode: deterministic (no dropout)
      ((not (layer-training layer)) input)
      
      ;; Training mode: apply stochastic depth
      (t
       (let* ((shape (tensor-shape input))
              (data (tensor-data input))
              (result (make-array (array-dimensions data) :element-type 'double-float))
              (keep-prob (- 1.0d0 drop-prob))
              (scale (/ 1.0d0 keep-prob))
              ;; Sample once per batch element (drop entire samples)
              (batch-size (first shape))
              (elements-per-sample (/ (array-total-size data) batch-size))
              ;; Use variational module's random state if available
              (random-state (if (find-package :variational)
                                (symbol-value (find-symbol "*VARIATIONAL-RANDOM-STATE*" :variational))
                                *random-state*)))
         
         ;; For each batch element, decide whether to keep or drop
         (dotimes (b batch-size)
           (let ((keep-sample (> (random 1.0d0 random-state) drop-prob)))
             (if keep-sample
                 ;; Keep and scale
                 (let ((start (* b elements-per-sample))
                       (end (* (1+ b) elements-per-sample)))
                   (loop for i from start below end do
                     (setf (row-major-aref result i)
                           (* (row-major-aref data i) scale))))
                 ;; Drop (set to zero)
                 (let ((start (* b elements-per-sample))
                       (end (* (1+ b) elements-per-sample)))
                   (loop for i from start below end do
                     (setf (row-major-aref result i) 0.0d0))))))
         
         (make-tensor result :shape shape :requires-grad (neural-network::requires-grad input)))))))

;;;; ----------------------------------------------------------------------------

(defclass layer-scale (layer)
  ((dim :initarg :dim
        :reader dim)
   (init-value :initarg :init-value
               :initform 1.0d-6
               :reader init-value)
   (gamma :accessor layer-scale-gamma))
  (:documentation "Layer Scale parameter for stabilizing training"))

(defmethod initialize-instance :after ((layer layer-scale) &key)
  (with-slots (dim init-value gamma) layer
    (setf gamma (make-tensor (make-array (list dim)
                                        :element-type 'double-float
                                        :initial-element init-value)
                            :shape (list dim)
                            :requires-grad t
                            :name "layer-scale"))
    (setf (slot-value layer 'neural-network:parameters) (list gamma))))

(defmethod forward ((layer layer-scale) input)
  "Scale input by learnable gamma parameter"
  (with-slots (gamma dim) layer
    (let* ((shape (tensor-shape input))
           (gamma-data (tensor-data gamma))
           (input-data (tensor-data input))
           (output-data (make-array shape :element-type 'double-float)))
      
      ;; Assuming input is (B, C, ...) and gamma is (C,)
      (let ((batch-size (first shape))
            (channels (second shape)))
        (dotimes (b batch-size)
          (dotimes (c channels)
            ;; Scale channel c by gamma[c]
            (let ((scale (aref gamma-data c)))
              ;; Handle remaining dimensions
              (labels ((scale-recursive (indices depth)
                         (if (= depth (length shape))
                             (setf (apply #'aref output-data indices)
                                   (* (apply #'aref input-data indices) scale))
                             (dotimes (i (nth depth shape))
                               (scale-recursive (append indices (list i)) (1+ depth))))))
                (scale-recursive (list b c) 2))))))
      
      (make-tensor output-data
                  :shape shape
                  :requires-grad (neural-network:requires-grad input)))))

;;;; ============================================================================
;;;; Stage-Level Abstractions
;;;; ============================================================================

(defclass convnext-stage (layer)
  ((dim :initarg :dim
        :reader dim)
   (depth :initarg :depth
          :reader stage-depth
          :documentation "Number of blocks in this stage")
   (blocks :accessor stage-blocks))
  (:documentation "ConvNeXt stage consisting of multiple ConvNeXt blocks"))

(defmethod initialize-instance :after ((stage convnext-stage) &key)
  (with-slots (dim depth blocks) stage
    (setf blocks (loop repeat depth
                      collect (make-instance 'convnext-block :dim dim)))
    
    (setf (slot-value stage 'neural-network:parameters)
          (apply #'append (mapcar #'neural-network:layer-parameters blocks)))))

(defmethod forward ((stage convnext-stage) input)
  "Forward pass through all blocks in the stage"
  (with-slots (blocks) stage
    (reduce (lambda (x block) (forward block x))
            blocks
            :initial-value input)))
