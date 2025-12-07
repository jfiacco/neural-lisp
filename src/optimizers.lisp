;;;; Neural Tensor Library - Advanced Optimizers
;;;; Adam, RMSprop, AdaGrad, AdaDelta, NAdam, and more

(defpackage :neural-tensor-optimizers
  (:use :common-lisp)
  (:shadow #:step)
  (:import-from :neural-network
                #:tensor
                #:tensor-data
                #:tensor-grad
                #:layer-parameters)
  (:export #:make-optimizer
           ;; Optimizers
           #:sgd
           #:sgd-momentum
           #:adam
           #:adamw
           #:rmsprop
           #:adagrad
           #:adadelta
           #:nadam
           #:adamax
           ;; Learning rate scheduling
           #:lr-scheduler
           #:step-lr
           #:exponential-lr
           #:cosine-annealing-lr
           #:reduce-on-plateau-lr
           ;; Utilities
           #:get-lr
           #:set-lr
           #:step
           #:step-scheduler
           #:zero-grad
           #:clip-grad-norm
           #:clip-grad-value))

(in-package :neural-tensor-optimizers)

;;;; ============================================================================
;;;; Optimizer Base Class
;;;; ============================================================================

(defclass optimizer ()
  ((parameters :initarg :parameters
               :accessor optimizer-parameters
               :documentation "List of tensors to optimize")
   (lr :initarg :lr
       :accessor learning-rate
       :initform 0.001
       :documentation "Learning rate")
   (state :initform (make-hash-table :test 'eq)
          :accessor optimizer-state
          :documentation "Per-parameter optimizer state"))
  (:documentation "Base class for all optimizers"))

(defgeneric step (optimizer)
  (:documentation "Perform one optimization step"))

(defgeneric zero-grad (optimizer)
  (:documentation "Zero all parameter gradients"))

(defmethod zero-grad ((opt optimizer))
  "Zero all parameter gradients"
  (dolist (param (optimizer-parameters opt))
    (when (neural-network::tensor-grad param)
      (let ((grad (neural-network::tensor-grad param)))
        (dotimes (i (array-total-size grad))
          (setf (row-major-aref grad i)
                (coerce 0.0 (array-element-type grad))))))))

;;;; ============================================================================
;;;; SGD with Momentum
;;;; ============================================================================

(defclass sgd-optimizer (optimizer)
  ((momentum :initarg :momentum
             :accessor momentum
             :initform 0.0
             :documentation "Momentum factor")
   (dampening :initarg :dampening
              :accessor dampening
              :initform 0.0
              :documentation "Dampening for momentum")
   (weight-decay :initarg :weight-decay
                 :accessor weight-decay
                 :initform 0.0
                 :documentation "Weight decay (L2 penalty)"))
  (:documentation "SGD optimizer with momentum"))

(defun sgd (&key parameters (lr 0.01) (momentum 0.0) (dampening 0.0) (weight-decay 0.0))
  "Create SGD optimizer with optional momentum"
  (make-instance 'sgd-optimizer
                 :parameters parameters
                 :lr lr
                 :momentum momentum
                 :dampening dampening
                 :weight-decay weight-decay))

(defmethod step ((opt sgd-optimizer))
  "SGD with momentum step"
  (let ((lr (learning-rate opt))
        (momentum (momentum opt))
        (dampening (dampening opt))
        (weight-decay (weight-decay opt))
        (state (optimizer-state opt)))
    
    (dolist (param (optimizer-parameters opt))
      (let* ((data (tensor-data param))
             (grad (neural-network::tensor-grad param))
             (param-state (gethash param state)))
        
        (when grad
          ;; Add weight decay
          (when (> weight-decay 0.0)
            (dotimes (i (array-total-size grad))
              (incf (row-major-aref grad i)
                    (* weight-decay (row-major-aref data i)))))
          
          ;; Apply momentum
          (if (> momentum 0.0)
              (progn
                ;; Initialize momentum buffer
                (unless param-state
                  (setf param-state
                        (make-array (array-dimensions data)
                                   :initial-element 0.0
                                   :element-type 'single-float))
                  (setf (gethash param state) param-state))
                
                ;; Update momentum buffer: v = momentum * v + (1 - dampening) * g
                (dotimes (i (array-total-size data))
                  (setf (row-major-aref param-state i)
                        (coerce (+ (* momentum (row-major-aref param-state i))
                                  (* (- 1.0 dampening) (row-major-aref grad i)))
                               'single-float)))

                ;; Update parameters: theta = theta - lr * v
                (dotimes (i (array-total-size data))
                  (decf (row-major-aref data i)
                        (coerce (* lr (row-major-aref param-state i)) 'single-float))))
              
              ;; No momentum - vanilla SGD
              (dotimes (i (array-total-size data))
                (decf (row-major-aref data i)
                      (coerce (* lr (row-major-aref grad i)) 'single-float)))))))))

;;;; ============================================================================
;;;; Adam Optimizer
;;;; ============================================================================

(defclass adam-optimizer (optimizer)
  ((beta1 :initarg :beta1
          :accessor beta1
          :initform 0.9
          :documentation "Exponential decay rate for first moment")
   (beta2 :initarg :beta2
          :accessor beta2
          :initform 0.999
          :documentation "Exponential decay rate for second moment")
   (epsilon :initarg :epsilon
            :accessor epsilon
            :initform 1e-8
            :documentation "Term for numerical stability")
   (weight-decay :initarg :weight-decay
                 :accessor weight-decay
                 :initform 0.0
                 :documentation "Weight decay (L2 penalty)")
   (amsgrad :initarg :amsgrad
            :accessor amsgrad
            :initform nil
            :documentation "Whether to use AMSGrad variant")
   (step-count :initform 0
               :accessor step-count
               :documentation "Number of steps taken"))
  (:documentation "Adam optimizer"))

(defun adam (&key parameters (lr 0.001) (beta1 0.9) (beta2 0.999) 
                  (epsilon 1e-8) (weight-decay 0.0) (amsgrad nil))
  "Create Adam optimizer"
  (make-instance 'adam-optimizer
                 :parameters parameters
                 :lr lr
                 :beta1 beta1
                 :beta2 beta2
                 :epsilon epsilon
                 :weight-decay weight-decay
                 :amsgrad amsgrad))

(defmethod step ((opt adam-optimizer))
  "Adam optimization step"
  (incf (step-count opt))
  (let ((lr (learning-rate opt))
        (beta1 (beta1 opt))
        (beta2 (beta2 opt))
        (epsilon (epsilon opt))
        (weight-decay (weight-decay opt))
        (amsgrad (amsgrad opt))
        (t-step (step-count opt))
        (state (optimizer-state opt)))
    
    ;; Bias correction terms
    (let ((bias-correction1 (- 1.0 (expt beta1 t-step)))
          (bias-correction2 (- 1.0 (expt beta2 t-step))))
      
      (dolist (param (optimizer-parameters opt))
        (let* ((data (tensor-data param))
               (grad (neural-network::tensor-grad param))
               (param-state (gethash param state)))
          
          (when grad
            ;; Initialize state
            (unless param-state
              (setf param-state
                    (list :m (make-array (array-dimensions data)
                                        :initial-element 0.0
                                        :element-type 'single-float)
                          :v (make-array (array-dimensions data)
                                        :initial-element 0.0
                                        :element-type 'single-float)
                          :v-max (when amsgrad
                                  (make-array (array-dimensions data)
                                             :initial-element 0.0
                                             :element-type 'single-float))))
              (setf (gethash param state) param-state))
            
            (let ((m (getf param-state :m))
                  (v (getf param-state :v))
                  (v-max (getf param-state :v-max)))
              
              ;; Add weight decay to gradient
              (when (> weight-decay 0.0)
                (dotimes (i (array-total-size grad))
                  (incf (row-major-aref grad i)
                        (* weight-decay (row-major-aref data i)))))
              
              ;; Update biased first moment estimate
              ;; m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
              (dotimes (i (array-total-size data))
                (setf (row-major-aref m i)
                      (coerce (+ (* beta1 (row-major-aref m i))
                                (* (- 1.0 beta1) (row-major-aref grad i)))
                             'single-float)))

              ;; Update biased second raw moment estimate
              ;; v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
              (dotimes (i (array-total-size data))
                (setf (row-major-aref v i)
                      (coerce (+ (* beta2 (row-major-aref v i))
                                (* (- 1.0 beta2)
                                   (expt (row-major-aref grad i) 2)))
                             'single-float)))
              
              ;; AMSGrad: use max of past squared gradients
              (when amsgrad
                (dotimes (i (array-total-size data))
                  (setf (row-major-aref v-max i)
                        (max (row-major-aref v-max i)
                             (row-major-aref v i)))))
              
              ;; Compute bias-corrected estimates and update parameters
              ;; theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
              (let ((v-hat (if amsgrad v-max v)))
                (dotimes (i (array-total-size data))
                  (let* ((m-hat (/ (row-major-aref m i) bias-correction1))
                         (v-hat-val (/ (row-major-aref v-hat i) bias-correction2))
                         (step-size (/ (* lr m-hat)
                                      (+ (sqrt v-hat-val) epsilon))))
                    (decf (row-major-aref data i) (coerce step-size 'single-float)))))))))))

;;;; ============================================================================
;;;; AdamW (Adam with decoupled weight decay)
;;;; ============================================================================

(defclass adamw-optimizer (adam-optimizer) ()
  (:documentation "AdamW optimizer - Adam with decoupled weight decay"))

(defun adamw (&key parameters (lr 0.001) (beta1 0.9) (beta2 0.999) 
                   (epsilon 1e-8) (weight-decay 0.01))
  "Create AdamW optimizer"
  (make-instance 'adamw-optimizer
                 :parameters parameters
                 :lr lr
                 :beta1 beta1
                 :beta2 beta2
                 :epsilon epsilon
                 :weight-decay weight-decay))

(defmethod step ((opt adamw-optimizer))
  "AdamW optimization step with decoupled weight decay"
  (incf (step-count opt))
  (let ((lr (learning-rate opt))
        (beta1 (beta1 opt))
        (beta2 (beta2 opt))
        (epsilon (epsilon opt))
        (weight-decay (weight-decay opt))
        (t-step (step-count opt))
        (state (optimizer-state opt)))
    
    (let ((bias-correction1 (- 1.0 (expt beta1 t-step)))
          (bias-correction2 (- 1.0 (expt beta2 t-step))))
      
      (dolist (param (optimizer-parameters opt))
        (let* ((data (tensor-data param))
               (grad (neural-network::tensor-grad param))
               (param-state (gethash param state)))
          
          (when grad
            (unless param-state
              (setf param-state
                    (list :m (make-array (array-dimensions data)
                                        :initial-element 0.0d0
                                        :element-type 'double-float)
                          :v (make-array (array-dimensions data)
                                        :initial-element 0.0d0
                                        :element-type 'double-float)))
              (setf (gethash param state) param-state))
            
            (let ((m (getf param-state :m))
                  (v (getf param-state :v)))
              
              ;; Update moments (WITHOUT weight decay in gradient)
              (dotimes (i (array-total-size data))
                (setf (row-major-aref m i)
                      (+ (* beta1 (row-major-aref m i))
                         (* (- 1.0 beta1) (row-major-aref grad i))))
                (setf (row-major-aref v i)
                      (+ (* beta2 (row-major-aref v i))
                         (* (- 1.0 beta2)
                            (expt (row-major-aref grad i) 2)))))
              
              ;; Update parameters with DECOUPLED weight decay
              (dotimes (i (array-total-size data))
                (let* ((m-hat (/ (row-major-aref m i) bias-correction1))
                       (v-hat (/ (row-major-aref v i) bias-correction2))
                       (step-size (/ (* lr m-hat)
                                    (+ (sqrt v-hat) epsilon))))
                  ;; Adam update
                  (decf (row-major-aref data i) (coerce step-size 'single-float))
                  ;; Decoupled weight decay
                  (when (> weight-decay 0.0)
                    (setf (row-major-aref data i)
                          (* (row-major-aref data i)
                             (coerce (- 1.0 (* lr weight-decay)) 'single-float)))))))))))))

;;;; ============================================================================
;;;; RMSprop Optimizer
;;;; ============================================================================

(defclass rmsprop-optimizer (optimizer)
  ((alpha :initarg :alpha
          :accessor alpha
          :initform 0.99
          :documentation "Smoothing constant")
   (epsilon :initarg :epsilon
            :accessor epsilon
            :initform 1e-8
            :documentation "Term for numerical stability")
   (momentum :initarg :momentum
             :accessor momentum
             :initform 0.0
             :documentation "Momentum factor")
   (weight-decay :initarg :weight-decay
                 :accessor weight-decay
                 :initform 0.0
                 :documentation "Weight decay"))
  (:documentation "RMSprop optimizer"))

(defun rmsprop (&key parameters (lr 0.01) (alpha 0.99) (epsilon 1e-8) 
                     (momentum 0.0) (weight-decay 0.0))
  "Create RMSprop optimizer"
  (make-instance 'rmsprop-optimizer
                 :parameters parameters
                 :lr lr
                 :alpha alpha
                 :epsilon epsilon
                 :momentum momentum
                 :weight-decay weight-decay))

(defmethod step ((opt rmsprop-optimizer))
  "RMSprop optimization step"
  (let ((lr (learning-rate opt))
        (alpha (alpha opt))
        (epsilon (epsilon opt))
        (momentum (momentum opt))
        (weight-decay (weight-decay opt))
        (state (optimizer-state opt)))
    
    (dolist (param (optimizer-parameters opt))
      (let* ((data (tensor-data param))
             (grad (neural-network::tensor-grad param))
             (param-state (gethash param state)))
        
        (when grad
          (unless param-state
            (setf param-state
                  (list :v (make-array (array-dimensions data)
                                      :initial-element 0.0
                                      :element-type 'single-float)
                        :buf (when (> momentum 0.0)
                              (make-array (array-dimensions data)
                                         :initial-element 0.0
                                         :element-type 'single-float))))
            (setf (gethash param state) param-state))
          
          (let ((v (getf param-state :v))
                (buf (getf param-state :buf)))
            
            ;; Add weight decay
            (when (> weight-decay 0.0)
              (dotimes (i (array-total-size grad))
                (incf (row-major-aref grad i)
                      (* weight-decay (row-major-aref data i)))))
            
            ;; Update running average of squared gradients
            ;; v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
            (dotimes (i (array-total-size data))
              (setf (row-major-aref v i)
                    (coerce (+ (* alpha (row-major-aref v i))
                               (* (- 1.0 alpha)
                                  (expt (row-major-aref grad i) 2)))
                            'single-float)))
            
            ;; Apply momentum if specified
            (if (> momentum 0.0)
                (progn
                  ;; buf_t = momentum * buf_{t-1} + g_t / sqrt(v_t + epsilon)
                  (dotimes (i (array-total-size data))
                    (setf (row-major-aref buf i)
                          (coerce (+ (* momentum (row-major-aref buf i))
                                    (/ (row-major-aref grad i)
                                       (+ (sqrt (row-major-aref v i)) epsilon)))
                                 'single-float)))
                  ;; theta_t = theta_{t-1} - lr * buf_t
                  (dotimes (i (array-total-size data))
                    (decf (row-major-aref data i)
                          (coerce (* lr (row-major-aref buf i)) 'single-float))))
                
                ;; Without momentum
                ;; theta_t = theta_{t-1} - lr * g_t / sqrt(v_t + epsilon)
                (dotimes (i (array-total-size data))
                  (decf (row-major-aref data i)
                        (coerce (* lr (/ (row-major-aref grad i)
                                        (+ (sqrt (row-major-aref v i)) epsilon)))
                               'single-float)))))))))))

;;;; ============================================================================
;;;; AdaGrad Optimizer
;;;; ============================================================================

(defclass adagrad-optimizer (optimizer)
  ((epsilon :initarg :epsilon
            :accessor epsilon
            :initform 1e-10
            :documentation "Term for numerical stability")
   (weight-decay :initarg :weight-decay
                 :accessor weight-decay
                 :initform 0.0
                 :documentation "Weight decay"))
  (:documentation "AdaGrad optimizer"))

(defun adagrad (&key parameters (lr 0.01) (epsilon 1e-10) (weight-decay 0.0))
  "Create AdaGrad optimizer"
  (make-instance 'adagrad-optimizer
                 :parameters parameters
                 :lr lr
                 :epsilon epsilon
                 :weight-decay weight-decay))

(defmethod step ((opt adagrad-optimizer))
  "AdaGrad optimization step"
  (let ((lr (learning-rate opt))
        (epsilon (epsilon opt))
        (weight-decay (weight-decay opt))
        (state (optimizer-state opt)))
    
    (dolist (param (optimizer-parameters opt))
      (let* ((data (tensor-data param))
             (grad (neural-network::tensor-grad param))
             (param-state (gethash param state)))
        
        (when grad
          (unless param-state
            (setf param-state
                  (make-array (array-dimensions data)
                             :initial-element 0.0
                             :element-type 'single-float))
            (setf (gethash param state) param-state))
          
          ;; Add weight decay
          (when (> weight-decay 0.0)
            (dotimes (i (array-total-size grad))
              (incf (row-major-aref grad i)
                    (* weight-decay (row-major-aref data i)))))
          
          ;; Accumulate squared gradients
          ;; sum_t = sum_{t-1} + g_t^2
          (dotimes (i (array-total-size data))
            (incf (row-major-aref param-state i)
                  (expt (row-major-aref grad i) 2)))
          
          ;; Update parameters
          ;; theta_t = theta_{t-1} - lr * g_t / sqrt(sum_t + epsilon)
          (dotimes (i (array-total-size data))
            (decf (row-major-aref data i)
                  (* lr (/ (row-major-aref grad i)
                          (+ (sqrt (row-major-aref param-state i)) epsilon))))))))))

;;;; ============================================================================
;;;; AdaDelta Optimizer
;;;; ============================================================================

(defclass adadelta-optimizer (optimizer)
  ((rho :initarg :rho
        :accessor rho
        :initform 0.9
        :documentation "Decay rate")
   (epsilon :initarg :epsilon
            :accessor epsilon
            :initform 1e-6
            :documentation "Term for numerical stability")
   (weight-decay :initarg :weight-decay
                 :accessor weight-decay
                 :initform 0.0
                 :documentation "Weight decay"))
  (:documentation "AdaDelta optimizer"))

(defun adadelta (&key parameters (lr 1.0) (rho 0.9) (epsilon 1e-6) (weight-decay 0.0))
  "Create AdaDelta optimizer"
  (make-instance 'adadelta-optimizer
                 :parameters parameters
                 :lr lr
                 :rho rho
                 :epsilon epsilon
                 :weight-decay weight-decay))

(defmethod step ((opt adadelta-optimizer))
  "AdaDelta optimization step"
  (let ((lr (learning-rate opt))
        (rho (rho opt))
        (epsilon (epsilon opt))
        (weight-decay (weight-decay opt))
        (state (optimizer-state opt)))
    
    (dolist (param (optimizer-parameters opt))
      (let* ((data (tensor-data param))
             (grad (neural-network::tensor-grad param))
             (param-state (gethash param state)))
        
        (when grad
          (unless param-state
            (setf param-state
                  (list :acc-grad (make-array (array-dimensions data)
                                             :initial-element 0.0
                                             :element-type 'single-float)
                        :acc-delta (make-array (array-dimensions data)
                                              :initial-element 0.0
                                              :element-type 'single-float)))
            (setf (gethash param state) param-state))
          
          (let ((acc-grad (getf param-state :acc-grad))
                (acc-delta (getf param-state :acc-delta)))
            
            ;; Add weight decay
            (when (> weight-decay 0.0)
              (dotimes (i (array-total-size grad))
                (incf (row-major-aref grad i)
                      (* weight-decay (row-major-aref data i)))))
            
            ;; Accumulate gradient
            ;; E[g^2]_t = rho * E[g^2]_{t-1} + (1-rho) * g_t^2
            (dotimes (i (array-total-size data))
              (setf (row-major-aref acc-grad i)
                    (+ (* rho (row-major-aref acc-grad i))
                       (* (- 1.0 rho) (expt (row-major-aref grad i) 2)))))
            
            ;; Compute update
            ;; delta_t = -sqrt(E[delta^2]_{t-1} + eps) / sqrt(E[g^2]_t + eps) * g_t
            (let ((deltas (make-array (array-dimensions data) :element-type 'single-float)))
              (dotimes (i (array-total-size data))
                (let ((delta (* (- lr)
                               (/ (sqrt (+ (row-major-aref acc-delta i) epsilon))
                                  (sqrt (+ (row-major-aref acc-grad i) epsilon)))
                               (row-major-aref grad i))))
                  (setf (row-major-aref deltas i) delta)
                  ;; Apply update
                  (incf (row-major-aref data i) delta)))
              
              ;; Accumulate updates
              ;; E[delta^2]_t = rho * E[delta^2]_{t-1} + (1-rho) * delta_t^2
              (dotimes (i (array-total-size data))
                (setf (row-major-aref acc-delta i)
                      (+ (* rho (row-major-aref acc-delta i))
                         (* (- 1.0 rho) (expt (row-major-aref deltas i) 2))))))))))))

;;;; ============================================================================
;;;; Utility Functions
;;;; ============================================================================

(defun get-lr (optimizer)
  "Get current learning rate"
  (learning-rate optimizer))

(defun set-lr (optimizer new-lr)
  "Set learning rate"
  (setf (learning-rate optimizer) new-lr))

(defun clip-grad-norm (parameters max-norm &optional (norm-type 2))
  "Clip gradient norm of parameters"
  (let ((total-norm 0.0))
    ;; Compute total norm
    (dolist (param parameters)
      (when (neural-network::tensor-grad param)
        (let ((grad (neural-network::tensor-grad param)))
          (dotimes (i (array-total-size grad))
            (incf total-norm 
                  (expt (abs (row-major-aref grad i)) norm-type))))))
    
    (setf total-norm (expt total-norm (/ 1.0 norm-type)))
    
    ;; Clip if necessary
    (when (> total-norm max-norm)
      (let ((clip-coef (/ max-norm (+ total-norm 1e-6))))
        (dolist (param parameters)
          (when (neural-network::tensor-grad param)
            (let ((grad (neural-network::tensor-grad param)))
              (dotimes (i (array-total-size grad))
                (setf (row-major-aref grad i)
                      (* (row-major-aref grad i) clip-coef))))))))
    
    total-norm))

(defun clip-grad-value (parameters clip-value)
  "Clip gradients to a maximum absolute value"
  (dolist (param parameters)
    (when (neural-network::tensor-grad param)
      (let ((grad (neural-network::tensor-grad param)))
        (dotimes (i (array-total-size grad))
          (setf (row-major-aref grad i)
                (max (min (row-major-aref grad i) clip-value)
                     (- clip-value))))))))
