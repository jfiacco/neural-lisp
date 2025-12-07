;;;; tests/test-optimizers.lisp - Optimizer Tests

(in-package #:neural-lisp-tests)

(def-suite optimizer-tests
  :description "Tests for optimization algorithms"
  :in neural-lisp-tests)

(in-suite optimizer-tests)

(test sgd-creation
  "Test SGD optimizer creation"
  (let* ((param (make-tensor #(1.0 2.0) :shape '(2) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 0.01)))
    (is (not (null opt)))
    (is (= 0.01 (get-lr opt)))))

(test sgd-step
  "Test SGD parameter update"
  (let* ((param (make-tensor #(1.0 2.0) :shape '(2) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 0.1)))
    ;; Set gradients manually
    (setf (tensor-grad param) (make-array '(2) :initial-contents '(1.0d0 1.0d0) :element-type 'double-float))
    ;; Take optimization step
    (neural-tensor-optimizers:step opt)
    ;; param = param - lr * grad = [1.0, 2.0] - 0.1 * [1.0, 1.0] = [0.9, 1.9]
    (is (< 0.89 (aref (tensor-data param) 0) 0.91))
    (is (< 1.89 (aref (tensor-data param) 1) 1.91))))

(test sgd-with-momentum
  "Test SGD with momentum"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 0.1 :momentum 0.9)))
    ;; First step
    (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (neural-tensor-optimizers:step opt)
    (let ((val1 (aref (tensor-data param) 0)))
      ;; Second step with same gradient
      (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
      (neural-tensor-optimizers:step opt)
      (let ((val2 (aref (tensor-data param) 0)))
        ;; With momentum, second step should be larger
        (is (< (- 1.0 val1) (- val1 val2)))))))

(test adam-creation
  "Test Adam optimizer creation"
  (let* ((param (make-tensor #(1.0 2.0) :shape '(2) :requires-grad t))
         (opt (adam :parameters (list param) :lr 0.001)))
    (is (not (null opt)))
    (is (= 0.001 (get-lr opt)))))

(test adam-step
  "Test Adam parameter update"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (adam :parameters (list param) :lr 0.1)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (let ((original-val (aref (tensor-data param) 0)))
      (neural-tensor-optimizers:step opt)
      (is (not (= original-val (aref (tensor-data param) 0)))))))

(test adamw-creation
  "Test AdamW optimizer creation"
  (let* ((param (make-tensor #(1.0 2.0) :shape '(2) :requires-grad t))
         (opt (adamw :parameters (list param) :lr 0.001 :weight-decay 0.01)))
    (is (not (null opt)))
    (is (= 0.001 (get-lr opt)))))

(test adamw-weight-decay
  "Test AdamW applies weight decay"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (adamw :parameters (list param) :lr 0.1 :weight-decay 0.1)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 0.0d0 :element-type 'double-float))
    (let ((original-val (aref (tensor-data param) 0)))
      (neural-tensor-optimizers:step opt)
      ;; Even with zero gradient, weight decay should reduce parameter
      (is (< (aref (tensor-data param) 0) original-val)))))

(test rmsprop-creation
  "Test RMSprop optimizer creation"
  (let* ((param (make-tensor #(1.0 2.0) :shape '(2) :requires-grad t))
         (opt (rmsprop :parameters (list param) :lr 0.01)))
    (is (not (null opt)))
    (is (= 0.01 (get-lr opt)))))

(test rmsprop-step
  "Test RMSprop parameter update"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (rmsprop :parameters (list param) :lr 0.1)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (let ((original-val (aref (tensor-data param) 0)))
      (neural-tensor-optimizers:step opt)
      (is (not (= original-val (aref (tensor-data param) 0)))))))

(test zero-grad-optimizer
  "Test optimizer zero-grad method"
  (let* ((param (make-tensor #(1.0 2.0) :shape '(2) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 0.01)))
    ;; Set gradients
    (setf (tensor-grad param) (make-array '(2) :initial-contents '(1.0d0 2.0d0) :element-type 'double-float))
    ;; Zero them
    (zero-grad opt)
    (is (= 0.0 (aref (tensor-grad param) 0)))
    (is (= 0.0 (aref (tensor-grad param) 1)))))

(test multiple-parameters
  "Test optimizer with multiple parameters"
  (let* ((param1 (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (param2 (make-tensor #(2.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param1 param2) :lr 0.1)))
    (setf (tensor-grad param1) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (setf (tensor-grad param2) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (neural-tensor-optimizers:step opt)
    (is (< 0.89 (aref (tensor-data param1) 0) 0.91))
    (is (< 1.89 (aref (tensor-data param2) 0) 1.91))))

(test lr-step-scheduler
  "Test learning rate scheduler"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 1.0))
         (scheduler (step-lr-scheduler opt 2 :gamma 0.5)))
    (is (= 1.0 (get-lr opt)))
    (step-scheduler scheduler)
    (is (= 1.0 (get-lr opt)))  ; Not changed yet
    (step-scheduler scheduler)
    (is (= 0.5 (get-lr opt)))  ; Changed after 2 steps
    (step-scheduler scheduler)
    (is (= 0.5 (get-lr opt)))
    (step-scheduler scheduler)
    (is (= 0.25 (get-lr opt)))))

(test cosine-annealing-scheduler
  "Test cosine annealing scheduler"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 1.0))
         (scheduler (cosine-annealing-scheduler opt 10)))
    (is (= 1.0 (get-lr opt)))
    ;; Step through half cycle
    (dotimes (i 5)
      (step-scheduler scheduler))
    ;; LR should be lower than initial
    (is (< (get-lr opt) 1.0))))

;;; Edge Cases and Robustness Tests

(test sgd-zero-learning-rate
  "Test SGD with zero learning rate"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 0.0)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (let ((original-val (aref (tensor-data param) 0)))
      (neural-tensor-optimizers:step opt)
      ;; Parameter should not change with zero LR
      (is (= original-val (aref (tensor-data param) 0))))))

(test sgd-very-small-learning-rate
  "Test SGD with very small learning rate"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 1.0e-10)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (let ((original-val (aref (tensor-data param) 0)))
      (neural-tensor-optimizers:step opt)
      ;; Change should be very small
      (is (< (abs (- original-val (aref (tensor-data param) 0))) 1.0e-9)))))

(test sgd-large-learning-rate
  "Test SGD with large learning rate"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 100.0)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (let ((original-val (aref (tensor-data param) 0)))
      (neural-tensor-optimizers:step opt)
      ;; Change should be large
      (is (> (abs (- original-val (aref (tensor-data param) 0))) 50.0)))))

(test sgd-negative-gradient
  "Test SGD with negative gradients"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 0.1)))
    (setf (tensor-grad param) (make-array '(1) :initial-element -1.0d0 :element-type 'double-float))
    (neural-tensor-optimizers:step opt)
    ;; Parameter should increase (gradient descent on negative gradient)
    (is (> (aref (tensor-data param) 0) 1.0))))

(test sgd-zero-gradient
  "Test SGD with zero gradient"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 0.1)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 0.0d0 :element-type 'double-float))
    (let ((original-val (aref (tensor-data param) 0)))
      (neural-tensor-optimizers:step opt)
      ;; Parameter should not change
      (is (= original-val (aref (tensor-data param) 0))))))

(test sgd-momentum-zero
  "Test SGD with zero momentum"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 0.1 :momentum 0.0)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (neural-tensor-optimizers:step opt)
    ;; Should behave like regular SGD
    (is (< 0.89 (aref (tensor-data param) 0) 0.91))))

(test sgd-momentum-one
  "Test SGD with maximum momentum"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 0.1 :momentum 1.0)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (neural-tensor-optimizers:step opt)
    ;; With momentum=1, velocity accumulates fully
    (is (< (aref (tensor-data param) 0) 1.0))))

(test sgd-multiple-steps-consistency
  "Test SGD produces consistent updates across multiple steps"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 0.1)))
    (dotimes (i 10)
      (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
      (neural-tensor-optimizers:step opt))
    ;; After 10 steps with gradient 1.0, param should be 1.0 - 10*0.1 = 0.0
    (is (< (abs (aref (tensor-data param) 0)) 0.01))))

(test adam-default-hyperparameters
  "Test Adam with default hyperparameters"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (adam :parameters (list param) :lr 0.001)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (neural-tensor-optimizers:step opt)
    ;; Should update parameter
    (is (not (= 1.0 (aref (tensor-data param) 0))))))

(test adam-multiple-steps
  "Test Adam across multiple optimization steps"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (adam :parameters (list param) :lr 0.01)))
    (dotimes (i 5)
      (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
      (neural-tensor-optimizers:step opt))
    ;; Parameter should decrease
    (is (< (aref (tensor-data param) 0) 1.0))))

(test adam-zero-gradient
  "Test Adam with zero gradient"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (adam :parameters (list param) :lr 0.1)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 0.0d0 :element-type 'double-float))
    (let ((original-val (aref (tensor-data param) 0)))
      (neural-tensor-optimizers:step opt)
      ;; Parameter should not change significantly
      (is (< (abs (- original-val (aref (tensor-data param) 0))) 0.01)))))

(test adamw-zero-weight-decay
  "Test AdamW with zero weight decay (should behave like Adam)"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (adamw :parameters (list param) :lr 0.1 :weight-decay 0.0)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 0.0d0 :element-type 'double-float))
    (let ((original-val (aref (tensor-data param) 0)))
      (neural-tensor-optimizers:step opt)
      ;; With zero gradient and zero weight decay, param should stay same
      (is (< (abs (- original-val (aref (tensor-data param) 0))) 0.01)))))

(test adamw-large-weight-decay
  "Test AdamW with large weight decay"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (adamw :parameters (list param) :lr 0.1 :weight-decay 0.5)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 0.0d0 :element-type 'double-float))
    (let ((original-val (aref (tensor-data param) 0)))
      (neural-tensor-optimizers:step opt)
      ;; Weight decay should significantly reduce parameter
      (is (< (aref (tensor-data param) 0) original-val)))))

(test rmsprop-zero-gradient
  "Test RMSprop with zero gradient"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (rmsprop :parameters (list param) :lr 0.1)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 0.0d0 :element-type 'double-float))
    (let ((original-val (aref (tensor-data param) 0)))
      (neural-tensor-optimizers:step opt)
      ;; Parameter should not change significantly
      (is (< (abs (- original-val (aref (tensor-data param) 0))) 0.01)))))

(test rmsprop-consistent-gradients
  "Test RMSprop with consistent gradients over multiple steps"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (rmsprop :parameters (list param) :lr 0.1)))
    (dotimes (i 5)
      (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
      (neural-tensor-optimizers:step opt))
    ;; Parameter should have decreased
    (is (< (aref (tensor-data param) 0) 1.0))))

(test optimizer-single-parameter
  "Test optimizer with single parameter"
  (let* ((param (make-tensor #(5.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 0.1)))
    (setf (tensor-grad param) (make-array '(1) :initial-element 2.0d0 :element-type 'double-float))
    (neural-tensor-optimizers:step opt)
    ;; 5.0 - 0.1*2.0 = 4.8
    (is (< 4.79 (aref (tensor-data param) 0) 4.81))))

(test optimizer-many-parameters
  "Test optimizer with many parameters"
  (let* ((params (loop for i from 1 to 10
                       collect (make-tensor #(1.0) :shape '(1) :requires-grad t)))
         (opt (sgd :parameters params :lr 0.1)))
    (dolist (p params)
      (setf (tensor-grad p) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float)))
    (neural-tensor-optimizers:step opt)
    ;; All parameters should be updated
    (dolist (p params)
      (is (< 0.89 (aref (tensor-data p) 0) 0.91)))))

(test zero-grad-all-parameters
  "Test zero-grad zeros all parameter gradients"
  (let* ((p1 (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (p2 (make-tensor #(2.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list p1 p2) :lr 0.1)))
    (setf (tensor-grad p1) (make-array '(1) :initial-element 5.0d0 :element-type 'double-float))
    (setf (tensor-grad p2) (make-array '(1) :initial-element 3.0d0 :element-type 'double-float))
    (zero-grad opt)
    (is (= 0.0 (aref (tensor-grad p1) 0)))
    (is (= 0.0 (aref (tensor-grad p2) 0)))))

(test lr-scheduler-step-consistency
  "Test learning rate scheduler step consistency"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 1.0))
         (scheduler (step-lr-scheduler opt 1 :gamma 0.5)))
    (is (= 1.0 (get-lr opt)))
    (step-scheduler scheduler)
    (is (= 0.5 (get-lr opt)))
    (step-scheduler scheduler)
    (is (= 0.25 (get-lr opt)))))

(test lr-scheduler-multiple-steps-before-decay
  "Test LR scheduler with multiple steps before decay"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 1.0))
         (scheduler (step-lr-scheduler opt 5 :gamma 0.1)))
    (dotimes (i 4)
      (step-scheduler scheduler)
      (is (= 1.0 (get-lr opt))))
    (step-scheduler scheduler)
    (is (= 0.1 (get-lr opt)))))

(test cosine-annealing-reaches-minimum
  "Test cosine annealing reaches minimum at half cycle"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 1.0))
         (scheduler (cosine-annealing-scheduler opt 100)))
    ;; Step to middle of cycle
    (dotimes (i 50)
      (step-scheduler scheduler))
    ;; LR should be around 0.5 at midpoint (between 1.0 and 0.0)
    (is (< (get-lr opt) 0.6))
    (is (> (get-lr opt) 0.4))))

(test optimizer-parameter-independence
  "Test that optimizer updates parameters independently"
  (let* ((p1 (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (p2 (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list p1 p2) :lr 0.1)))
    (setf (tensor-grad p1) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (setf (tensor-grad p2) (make-array '(1) :initial-element 2.0d0 :element-type 'double-float))
    (neural-tensor-optimizers:step opt)
    ;; p1: 1.0 - 0.1*1.0 = 0.9
    ;; p2: 1.0 - 0.1*2.0 = 0.8
    (is (< 0.89 (aref (tensor-data p1) 0) 0.91))
    (is (< 0.79 (aref (tensor-data p2) 0) 0.81))))

(test optimizer-state-persistence
  "Test that optimizer maintains state across steps"
  (let* ((param (make-tensor #(1.0) :shape '(1) :requires-grad t))
         (opt (sgd :parameters (list param) :lr 0.1 :momentum 0.9)))
    ;; First step
    (setf (tensor-grad param) (make-array '(1) :initial-element 1.0d0 :element-type 'double-float))
    (neural-tensor-optimizers:step opt)
    (let ((val1 (aref (tensor-data param) 0)))
      ;; Second step with different gradient
      (setf (tensor-grad param) (make-array '(1) :initial-element 0.5d0 :element-type 'double-float))
      (neural-tensor-optimizers:step opt)
      (let ((val2 (aref (tensor-data param) 0)))
        ;; Values should be different due to momentum state
        (is (not (= val1 val2)))))))
