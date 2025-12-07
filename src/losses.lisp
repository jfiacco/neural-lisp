;;;; Neural Tensor Library - Loss Functions and LR Schedulers

(defpackage :neural-tensor-losses
  (:use :common-lisp)
  (:import-from :neural-network
                #:tensor
                #:tensor-data
                #:tensor-shape
                #:make-tensor
                #:t+
                #:t-
                #:t*
                #:tsum
                #:tmean)
  (:export ;; Loss functions
           #:mse-loss
           #:mae-loss
           #:cross-entropy-loss
           #:binary-cross-entropy
           #:nll-loss
           #:kl-divergence
           #:smooth-l1-loss
           #:huber-loss
           ;; LR Schedulers
           #:lr-scheduler
           #:step-lr-scheduler
           #:exponential-lr-scheduler
           #:cosine-annealing-scheduler
           #:reduce-on-plateau-scheduler
           #:cyclic-lr-scheduler
           #:one-cycle-scheduler
           #:step-scheduler
           #:get-last-lr))

(in-package :neural-tensor-losses)

;;;; ============================================================================
;;;; Loss Functions
;;;; ============================================================================

(defun mse-loss (pred target &key (reduction :mean))
  "Mean Squared Error loss"
  (let* ((diff (t- pred target))
         (squared (t* diff diff)))
    (ecase reduction
      (:mean (tmean squared))
      (:sum (tsum squared))
      (:none squared))))

(defun mae-loss (pred target &key (reduction :mean))
  "Mean Absolute Error loss"
  (let* ((diff (neural-network::t- pred target))
         (abs-diff (neural-network::make-tensor
                    (let ((arr (make-array (neural-network::tensor-shape diff) :element-type 'double-float)))
                      (dotimes (i (array-total-size arr))
                        (setf (row-major-aref arr i)
                              (abs (row-major-aref (neural-network::tensor-data diff) i))))
                      arr)
                    :shape (neural-network::tensor-shape diff)
                    :requires-grad (neural-network::requires-grad diff))))
    (ecase reduction
      (:mean (neural-network::tmean abs-diff))
      (:sum (neural-network::tsum abs-diff))
      (:none abs-diff))))

(defun cross-entropy-loss (logits target &key (reduction :mean))
  "Cross entropy loss for classification
   logits: (batch_size, num_classes)
   target: (batch_size,) with class indices"
  (let* ((logits-data (neural-network::tensor-data logits))
         (target-data (neural-network::tensor-data target))
         (batch-size (first (neural-network::tensor-shape logits)))
         (num-classes (second (neural-network::tensor-shape logits))))
    
    (let ((loss-tensor 
           (neural-network::make-tensor
            (let ((arr (make-array (list batch-size) :element-type 'double-float :initial-element 0.0d0)))
              ;; Compute log-softmax and loss for each sample
              (dotimes (i batch-size)
                (let ((max-logit most-negative-single-float)
                      (target-class (floor (aref target-data i))))
                  
                  ;; Find max for numerical stability
                  (dotimes (j num-classes)
                    (setf max-logit (max max-logit (aref logits-data i j))))
                  
                  ;; Compute log-sum-exp
                  (let ((sum-exp 0.0))
                    (dotimes (j num-classes)
                      (incf sum-exp (exp (- (aref logits-data i j) max-logit))))
                    
                    ;; Loss = -log(p_target) = -(logit_target - max - log(sum_exp))
                    ;;      = -logit_target + max + log(sum_exp)
                    (setf (aref arr i)
                          (- (+ max-logit (log sum-exp))
                             (aref logits-data i target-class))))))
              arr)
            :shape (list batch-size)
            :requires-grad (neural-network::requires-grad logits))))
      (ecase reduction
        (:mean (tmean loss-tensor))
        (:sum (tsum loss-tensor))
        (:none loss-tensor)))))

(defun binary-cross-entropy (pred target &key (reduction :mean) (epsilon 1e-7))
  "Binary cross entropy loss
   BCE = -[y*log(p) + (1-y)*log(1-p)]"
  (let* ((pred-data (neural-network::tensor-data pred))
         (target-data (neural-network::tensor-data target))
         (loss-data (make-array (array-dimensions pred-data) :element-type 'double-float)))
    
    (dotimes (i (array-total-size pred-data))
      (let ((p (max epsilon (min (- 1.0d0 epsilon) (row-major-aref pred-data i))))
            (y (row-major-aref target-data i)))
        (setf (row-major-aref loss-data i)
              (- (+ (* y (log p))
                   (* (- 1.0d0 y) (log (- 1.0d0 p))))))))
    
    (let ((loss-tensor (neural-network::make-tensor loss-data
                                    :shape (neural-network::tensor-shape pred)
                                    :requires-grad (neural-network::requires-grad pred))))
      (ecase reduction
        (:mean (neural-network::tmean loss-tensor))
        (:sum (neural-network::tsum loss-tensor))
        (:none loss-tensor)))))

(defun nll-loss (log-probs target &key (reduction :mean))
  "Negative log likelihood loss"
  (let* ((log-probs-data (neural-network::tensor-data log-probs))
         (target-data (neural-network::tensor-data target))
         (batch-size (first (neural-network::tensor-shape log-probs)))
         (result-losses (make-array (list batch-size) :element-type 'double-float :initial-element 0.0d0)))
    
    (dotimes (i batch-size)
      (let ((target-class (floor (aref target-data i))))
        (setf (aref result-losses i)
              (- (aref log-probs-data i target-class)))))
    
    (let ((loss-tensor (neural-network::make-tensor result-losses
                                    :shape (list batch-size)
                                    :requires-grad (neural-network::requires-grad log-probs))))
      (ecase reduction
        (:mean (neural-network::tmean loss-tensor))
        (:sum (neural-network::tsum loss-tensor))
        (:none loss-tensor)))))

(defun kl-divergence (p q &key (reduction :mean) (epsilon 1e-7))
  "KL divergence: KL(P||Q) = sum(P * log(P/Q))"
  (let* ((p-data (neural-network::tensor-data p))
         (q-data (neural-network::tensor-data q))
         (kl-data (make-array (array-dimensions p-data) :element-type 'double-float)))
    
    (dotimes (i (array-total-size p-data))
      (let ((p-val (max epsilon (row-major-aref p-data i)))
            (q-val (max epsilon (row-major-aref q-data i))))
        (setf (row-major-aref kl-data i)
              (* p-val (log (/ p-val q-val))))))
    
    (let ((kl-tensor (neural-network::make-tensor kl-data
                                  :shape (neural-network::tensor-shape p)
                                  :requires-grad (or (neural-network::requires-grad p)
                                                    (neural-network::requires-grad q)))))
      (ecase reduction
        (:mean (neural-network::tmean kl-tensor))
        (:sum (neural-network::tsum kl-tensor))
        (:none kl-tensor)))))

(defun smooth-l1-loss (pred target &key (reduction :mean) (beta 1.0))
  "Smooth L1 loss (Huber loss variant)"
  (let* ((diff-data (neural-network::tensor-data (neural-network::t- pred target)))
         (loss-data (make-array (array-dimensions diff-data) :element-type 'double-float)))
    
    (dotimes (i (array-total-size diff-data))
      (let ((abs-diff (abs (row-major-aref diff-data i))))
        (setf (row-major-aref loss-data i)
              (if (< abs-diff beta)
                  (* 0.5d0 (/ (expt (row-major-aref diff-data i) 2) beta))
                  (- abs-diff (* 0.5d0 beta))))))
    
    (let ((loss-tensor (neural-network::make-tensor loss-data
                                    :shape (neural-network::tensor-shape pred)
                                    :requires-grad (neural-network::requires-grad pred))))
      (ecase reduction
        (:mean (neural-network::tmean loss-tensor))
        (:sum (neural-network::tsum loss-tensor))
        (:none loss-tensor)))))

(defun huber-loss (pred target &key (reduction :mean) (delta 1.0))
  "Huber loss - less sensitive to outliers than MSE"
  (smooth-l1-loss pred target :reduction reduction :beta delta))

;;;; ============================================================================
;;;; Learning Rate Schedulers
;;;; ============================================================================

(defclass lr-scheduler ()
  ((optimizer :initarg :optimizer
              :accessor scheduler-optimizer
              :documentation "Optimizer to adjust")
   (last-epoch :initform -1
               :accessor last-epoch
               :documentation "Last epoch number")
   (base-lrs :initform nil
             :accessor base-lrs
             :documentation "Base learning rates"))
  (:documentation "Base class for learning rate schedulers"))

(defgeneric step-scheduler (scheduler &optional epoch)
  (:documentation "Update learning rate"))

(defgeneric get-last-lr (scheduler)
  (:documentation "Get last computed learning rate"))

(defmethod initialize-instance :after ((sched lr-scheduler) &key)
  "Initialize base learning rates"
  (setf (base-lrs sched)
        (list (neural-tensor-optimizers::learning-rate 
               (scheduler-optimizer sched)))))

;;;; Step LR Scheduler
(defclass step-lr-scheduler (lr-scheduler)
  ((step-size :initarg :step-size
              :accessor step-size
              :documentation "Period of learning rate decay")
   (gamma :initarg :gamma
          :accessor gamma
          :initform 0.1
          :documentation "Multiplicative factor of learning rate decay"))
  (:documentation "Decays learning rate by gamma every step-size epochs"))

(defun step-lr-scheduler (optimizer step-size &key (gamma 0.1))
  "Create step learning rate scheduler"
  (make-instance 'step-lr-scheduler
                 :optimizer optimizer
                 :step-size step-size
                 :gamma gamma))

(defmethod step-scheduler ((sched step-lr-scheduler) &optional epoch)
  "Step the scheduler"
  (incf (last-epoch sched))
  (let* ((current-epoch (or epoch (last-epoch sched)))
         (step-size (step-size sched))
         (gamma (gamma sched))
         (base-lr (first (base-lrs sched)))
         ;; Use (current-epoch + 1) to count steps starting from 1
         (new-lr (* base-lr (expt gamma (floor (+ current-epoch 1) step-size)))))
    (setf (neural-tensor-optimizers::learning-rate (scheduler-optimizer sched))
          new-lr)
    new-lr))

;;;; Exponential LR Scheduler
(defclass exponential-lr-scheduler (lr-scheduler)
  ((gamma :initarg :gamma
          :accessor gamma
          :initform 0.95
          :documentation "Multiplicative factor of learning rate decay"))
  (:documentation "Decays learning rate by gamma every epoch"))

(defun exponential-lr-scheduler (optimizer &key (gamma 0.95))
  "Create exponential learning rate scheduler"
  (make-instance 'exponential-lr-scheduler
                 :optimizer optimizer
                 :gamma gamma))

(defmethod step-scheduler ((sched exponential-lr-scheduler) &optional epoch)
  "Step the scheduler"
  (declare (ignore epoch))
  (incf (last-epoch sched))
  (let* ((gamma (gamma sched))
         (current-lr (neural-tensor-optimizers::learning-rate (scheduler-optimizer sched)))
         (new-lr (* current-lr gamma)))
    (setf (neural-tensor-optimizers::learning-rate (scheduler-optimizer sched))
          new-lr)
    new-lr))

;;;; Cosine Annealing LR Scheduler
(defclass cosine-annealing-scheduler (lr-scheduler)
  ((t-max :initarg :t-max
          :accessor t-max
          :documentation "Maximum number of iterations")
   (eta-min :initarg :eta-min
            :accessor eta-min
            :initform 0.0
            :documentation "Minimum learning rate"))
  (:documentation "Cosine annealing learning rate schedule"))

(defun cosine-annealing-scheduler (optimizer t-max &key (eta-min 0.0))
  "Create cosine annealing learning rate scheduler"
  (make-instance 'cosine-annealing-scheduler
                 :optimizer optimizer
                 :t-max t-max
                 :eta-min eta-min))

(defmethod step-scheduler ((sched cosine-annealing-scheduler) &optional epoch)
  "Step the scheduler"
  (incf (last-epoch sched))
  (let* ((current-epoch (or epoch (last-epoch sched)))
         (t-max (t-max sched))
         (eta-min (eta-min sched))
         (base-lr (first (base-lrs sched)))
         (new-lr (+ eta-min
                   (* (- base-lr eta-min)
                      (+ 1.0 (cos (/ (* pi current-epoch) t-max)))
                      0.5))))
    (setf (neural-tensor-optimizers::learning-rate (scheduler-optimizer sched))
          new-lr)
    new-lr))

;;;; Reduce On Plateau Scheduler
(defclass reduce-on-plateau-scheduler (lr-scheduler)
  ((mode :initarg :mode
         :accessor mode
         :initform :min
         :documentation ":min or :max")
   (factor :initarg :factor
           :accessor factor
           :initform 0.1
           :documentation "Factor to reduce LR")
   (patience :initarg :patience
             :accessor patience
             :initform 10
             :documentation "Number of epochs to wait")
   (threshold :initarg :threshold
              :accessor threshold
              :initform 1e-4
              :documentation "Threshold for measuring improvement")
   (cooldown :initarg :cooldown
             :accessor cooldown
             :initform 0
             :documentation "Cooldown period after LR reduction")
   (min-lr :initarg :min-lr
           :accessor min-lr
           :initform 0.0
           :documentation "Minimum learning rate")
   (num-bad-epochs :initform 0
                   :accessor num-bad-epochs)
   (cooldown-counter :initform 0
                     :accessor cooldown-counter)
   (best-metric :initform nil
                :accessor best-metric))
  (:documentation "Reduce learning rate when metric plateaus"))

(defun reduce-on-plateau-scheduler (optimizer &key (mode :min) (factor 0.1)
                                                   (patience 10) (threshold 1e-4)
                                                   (cooldown 0) (min-lr 0.0))
  "Create reduce on plateau learning rate scheduler"
  (make-instance 'reduce-on-plateau-scheduler
                 :optimizer optimizer
                 :mode mode
                 :factor factor
                 :patience patience
                 :threshold threshold
                 :cooldown cooldown
                 :min-lr min-lr))

(defmethod step-scheduler ((sched reduce-on-plateau-scheduler) &optional metric)
  "Step the scheduler with metric"
  (when (null metric)
    (error "Metric required for reduce-on-plateau scheduler"))
  
  ;; Cooldown
  (when (> (cooldown-counter sched) 0)
    (decf (cooldown-counter sched))
    (return-from step-scheduler 
      (neural-tensor-optimizers::learning-rate (scheduler-optimizer sched))))
  
  ;; Check if metric improved
  (let ((improved nil))
    (if (null (best-metric sched))
        (progn
          (setf (best-metric sched) metric)
          (setf improved t))
        (let ((threshold (threshold sched)))
          (setf improved
                (ecase (mode sched)
                  (:min (< metric (- (best-metric sched) threshold)))
                  (:max (> metric (+ (best-metric sched) threshold)))))))
    
    (if improved
        (progn
          (setf (best-metric sched) metric)
          (setf (num-bad-epochs sched) 0))
        (progn
          (incf (num-bad-epochs sched))
          (when (>= (num-bad-epochs sched) (patience sched))
            ;; Reduce learning rate
            (let* ((current-lr (neural-tensor-optimizers::learning-rate 
                               (scheduler-optimizer sched)))
                   (new-lr (max (min-lr sched) (* current-lr (factor sched)))))
              (setf (neural-tensor-optimizers::learning-rate 
                     (scheduler-optimizer sched))
                    new-lr)
              (setf (num-bad-epochs sched) 0)
              (setf (cooldown-counter sched) (cooldown sched))
              (format t "Reducing learning rate to ~,6f~%" new-lr))))))
  
  (neural-tensor-optimizers::learning-rate (scheduler-optimizer sched)))

;;;; Cyclic LR Scheduler
(defclass cyclic-lr-scheduler (lr-scheduler)
  ((max-lr :initarg :max-lr
           :accessor max-lr
           :documentation "Maximum learning rate")
   (step-size-up :initarg :step-size-up
                 :accessor step-size-up
                 :initform 2000
                 :documentation "Steps in increasing phase")
   (step-size-down :initarg :step-size-down
                   :accessor step-size-down
                   :initform nil
                   :documentation "Steps in decreasing phase")
   (mode :initarg :mode
         :accessor mode
         :initform :triangular
         :documentation "Mode: :triangular, :triangular2, :exp-range")
   (gamma :initarg :gamma
          :accessor gamma
          :initform 1.0
          :documentation "Gamma for exp-range mode")
   (cycle :initform 0
          :accessor cycle)
   (step-count :initform 0
               :accessor step-count))
  (:documentation "Cyclic learning rate scheduler"))

(defun cyclic-lr-scheduler (optimizer max-lr &key (step-size-up 2000)
                                                  (step-size-down nil)
                                                  (mode :triangular)
                                                  (gamma 1.0))
  "Create cyclic learning rate scheduler"
  (make-instance 'cyclic-lr-scheduler
                 :optimizer optimizer
                 :max-lr max-lr
                 :step-size-up step-size-up
                 :step-size-down (or step-size-down step-size-up)
                 :mode mode
                 :gamma gamma))

(defmethod step-scheduler ((sched cyclic-lr-scheduler) &optional epoch)
  "Step the scheduler"
  (declare (ignore epoch))
  (incf (step-count sched))
  
  (let* ((step-up (step-size-up sched))
         (step-down (step-size-down sched))
         (total-size (+ step-up step-down))
         (step-in-cycle (mod (step-count sched) total-size))
         (base-lr (first (base-lrs sched)))
         (max-lr (max-lr sched)))
    
    ;; Compute scale factor based on mode
    (let ((scale-factor
            (ecase (mode sched)
              (:triangular 1.0)
              (:triangular2 (/ 1.0 (expt 2.0 (cycle sched))))
              (:exp-range (expt (gamma sched) (step-count sched))))))
      
      ;; Compute learning rate
      (let ((new-lr
              (if (< step-in-cycle step-up)
                  ;; Increasing phase
                  (+ base-lr
                     (* (- max-lr base-lr)
                        (/ step-in-cycle step-up)
                        scale-factor))
                  ;; Decreasing phase
                  (+ base-lr
                     (* (- max-lr base-lr)
                        (- 1.0 (/ (- step-in-cycle step-up) step-down))
                        scale-factor)))))
        
        ;; Update cycle counter
        (when (and (= step-in-cycle 0) (> (step-count sched) 0))
          (incf (cycle sched)))
        
        (setf (neural-tensor-optimizers::learning-rate (scheduler-optimizer sched))
              new-lr)
        new-lr))))

(defmethod get-last-lr ((sched lr-scheduler))
  "Get last learning rate"
  (neural-tensor-optimizers::learning-rate (scheduler-optimizer sched)))
