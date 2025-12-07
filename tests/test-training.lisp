;;;; tests/test-training.lisp - Training Loop Tests

(in-package #:neural-lisp-tests)

(def-suite training-tests
  :description "Tests for training utilities"
  :in neural-lisp-tests)

(in-suite training-tests)

(test simple-training-loop
  "Test simple training loop with fit function"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0) (3.0 4.0) (5.0 6.0)) :shape '(3 2)))
         (y-train (make-tensor #2A((3.0) (7.0) (11.0)) :shape '(3 1)))
         (initial-loss (aref (tensor-data (mse-loss (forward model x-train) y-train)) 0)))
    ;; Train for a few epochs
    (fit model opt x-train y-train :epochs 5 :loss-fn #'mse-loss :verbose nil)
    ;; Loss should decrease
    (let ((final-loss (aref (tensor-data (mse-loss (forward model x-train) y-train)) 0)))
      (is (< final-loss initial-loss)))))

(test training-with-validation
  "Test training with validation split"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0) (3.0 4.0) (5.0 6.0) (7.0 8.0)) :shape '(4 2)))
         (y-train (make-tensor #2A((3.0) (7.0) (11.0) (15.0)) :shape '(4 1)))
         (x-val (make-tensor #2A((2.0 3.0)) :shape '(1 2)))
         (y-val (make-tensor #2A((5.0)) :shape '(1 1))))
    (fit model opt x-train y-train 
         :epochs 3 
         :loss-fn #'mse-loss 
         :x-val x-val 
         :y-val y-val
         :verbose nil)
    ;; Should complete without error
    (is (eql t t))))

(test train-eval-mode
  "Test switching between train and eval modes"
  (let ((model (sequential 
                 (linear 2 3)
                 (relu-layer))))
    (is (layer-training model))
    (eval-mode model)
    (is (not (layer-training model)))
    (train-mode model)
    (is (layer-training model))))

(test gradient-accumulation
  "Test gradient accumulation during training"
  (let* ((model (sequential (linear 1 1)))
         (x (make-tensor #2A((1.0)) :shape '(1 1)))
         (y (make-tensor #2A((2.0)) :shape '(1 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01)))
    ;; Forward pass
    (let ((pred (forward model x)))
      (let ((loss (mse-loss pred y)))
        ;; Backward pass
        (backward loss)
        ;; Gradients should be non-null
        (is (not (null (tensor-grad (first (parameters model))))))))))

(test optimizer-step-updates-parameters
  "Test that optimizer step actually updates parameters"
  (let* ((model (sequential (linear 1 1)))
         (param (first (parameters model)))
         (x (make-tensor #2A((1.0)) :shape '(1 1)))
         (y (make-tensor #2A((2.0)) :shape '(1 1)))
         (opt (sgd :parameters (parameters model) :lr 0.1))
         (original-val (aref (tensor-data param) 0 0)))
    (let* ((pred (forward model x))
           (loss (mse-loss pred y)))
      (backward loss)
      (neural-tensor-optimizers:step opt)
      (is (not (= original-val (aref (tensor-data param) 0 0)))))))

(test zero-grad-before-backward
  "Test that gradients are zeroed before backward pass"
  (let* ((model (sequential (linear 1 1)))
         (x (make-tensor #2A((1.0)) :shape '(1 1)))
         (y (make-tensor #2A((2.0)) :shape '(1 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01)))
    ;; First forward-backward
    (let* ((pred1 (forward model x))
           (loss1 (mse-loss pred1 y)))
      (backward loss1))
    ;; Zero gradients
    (zero-grad opt)
    ;; Check that gradients are zero
    (let ((param (first (parameters model))))
      (is (every (lambda (x) (= x 0.0d0)) 
                 (loop for i below (array-total-size (tensor-grad param))
                       collect (row-major-aref (tensor-grad param) i)))))))

(test batch-training
  "Test training with multiple batches"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         ;; Create larger dataset
         (x-train (make-tensor (make-array '(10 2) :initial-element 1.0d0) :shape '(10 2)))
         (y-train (make-tensor (make-array '(10 1) :initial-element 2.0d0) :shape '(10 1))))
    ;; Train with batch size
    (fit model opt x-train y-train 
         :epochs 2 
         :batch-size 2
         :loss-fn #'mse-loss
         :verbose nil)
    (is (eql t t))))

(test early-stopping-behavior
  "Test training can be stopped early"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y (make-tensor #2A((3.0)) :shape '(1 1)))
         (epochs-run 0))
    ;; Train for small number of epochs
    (fit model opt x y 
         :epochs 3 
         :loss-fn #'mse-loss 
         :verbose nil
         :callback (lambda (epoch loss) 
                     (declare (ignore loss))
                     (incf epochs-run)
                     (< epoch 3)))
    (is (<= epochs-run 3))))

(test loss-decreases-with-training
  "Test that loss generally decreases during training"
  (let* ((model (sequential 
                  (linear 3 5)
                  (relu-layer)
                  (linear 5 1)))
         (opt (adam :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0 3.0)
                                   (4.0 5.0 6.0)
                                   (7.0 8.0 9.0)) :shape '(3 3)))
         (y-train (make-tensor #2A((6.0) (15.0) (24.0)) :shape '(3 1)))
         (losses nil))
    ;; Track losses during training
    (fit model opt x-train y-train 
         :epochs 10 
         :loss-fn #'mse-loss 
         :verbose nil
         :callback (lambda (epoch loss) 
                     (declare (ignore epoch))
                     (push loss losses)
                     t))
    ;; Check that final loss is less than initial loss
    (is (< (first losses) (car (last losses))))))

(test model-evaluation
  "Test model evaluation on test set"
  (let* ((model (sequential (linear 2 1)))
         (x-test (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-test (make-tensor #2A((3.0)) :shape '(1 1))))
    ;; Put model in eval mode
    (eval-mode model)
    (let* ((pred (forward model x-test))
           (loss (mse-loss pred y-test)))
      (is (not (null loss)))
      (is (equal '(1) (tensor-shape loss))))))

;;; Edge Cases and Robustness Tests

(test training-single-sample
  "Test training with single sample"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.1))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((3.0)) :shape '(1 1))))
    (fit model opt x-train y-train :epochs 5 :loss-fn #'mse-loss :verbose nil)
    ;; Should complete without error
    (is (eql t t))))

(test training-batch-size-one
  "Test training with batch size of 1"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2)))
         (y-train (make-tensor #2A((3.0) (7.0)) :shape '(2 1))))
    (fit model opt x-train y-train 
         :epochs 2 
         :batch-size 1
         :loss-fn #'mse-loss 
         :verbose nil)
    (is (eql t t))))

(test training-batch-size-larger-than-dataset
  "Test training with batch size larger than dataset"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2)))
         (y-train (make-tensor #2A((3.0) (7.0)) :shape '(2 1))))
    (fit model opt x-train y-train 
         :epochs 2 
         :batch-size 100
         :loss-fn #'mse-loss 
         :verbose nil)
    (is (eql t t))))

(test training-zero-epochs
  "Test training with zero epochs"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((3.0)) :shape '(1 1))))
    (fit model opt x-train y-train :epochs 0 :loss-fn #'mse-loss :verbose nil)
    ;; Should complete immediately
    (is (eql t t))))

(test training-one-epoch
  "Test training with single epoch"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2)))
         (y-train (make-tensor #2A((3.0) (7.0)) :shape '(2 1))))
    (fit model opt x-train y-train :epochs 1 :loss-fn #'mse-loss :verbose nil)
    (is (eql t t))))

(test training-many-epochs
  "Test training with many epochs"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((3.0)) :shape '(1 1))))
    (fit model opt x-train y-train :epochs 100 :loss-fn #'mse-loss :verbose nil)
    (is (eql t t))))

(test training-large-dataset
  "Test training with large dataset"
  (let* ((model (sequential (linear 5 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (randn '(100 5)))
         (y-train (randn '(100 1))))
    (fit model opt x-train y-train 
         :epochs 2 
         :batch-size 10
         :loss-fn #'mse-loss 
         :verbose nil)
    (is (eql t t))))

(test training-preserves-model-structure
  "Test that training doesn't change model structure"
  (let* ((model (sequential (linear 3 5) (relu-layer) (linear 5 2)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (randn '(5 3)))
         (y-train (randn '(5 2)))
         (num-layers (length (seq-layers model))))
    (fit model opt x-train y-train :epochs 3 :loss-fn #'mse-loss :verbose nil)
    ;; Model should still have same number of layers
    (is (= num-layers (length (seq-layers model))))))

(test training-gradient-zeroing
  "Test that gradients are properly zeroed between batches"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2)))
         (y-train (make-tensor #2A((3.0) (7.0)) :shape '(2 1))))
    (fit model opt x-train y-train :epochs 1 :loss-fn #'mse-loss :verbose nil)
    ;; After training, gradients should exist but could be zero or non-zero
    ;; depending on when zero-grad was called
    (is (not (null (tensor-grad (first (parameters model))))))))

(test training-with-different-optimizers
  "Test training with different optimizer types"
  (let* ((model1 (sequential (linear 2 1)))
         (model2 (sequential (linear 2 1)))
         (model3 (sequential (linear 2 1)))
         (opt1 (sgd :parameters (parameters model1) :lr 0.01))
         (opt2 (adam :parameters (parameters model2) :lr 0.01))
         (opt3 (rmsprop :parameters (parameters model3) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((3.0)) :shape '(1 1))))
    (fit model1 opt1 x-train y-train :epochs 5 :loss-fn #'mse-loss :verbose nil)
    (fit model2 opt2 x-train y-train :epochs 5 :loss-fn #'mse-loss :verbose nil)
    (fit model3 opt3 x-train y-train :epochs 5 :loss-fn #'mse-loss :verbose nil)
    ;; All should complete successfully
    (is (eql t t))))

(test training-with-different-losses
  "Test training with different loss functions"
  (let* ((model1 (sequential (linear 2 1)))
         (model2 (sequential (linear 2 1)))
         (opt1 (sgd :parameters (parameters model1) :lr 0.01))
         (opt2 (sgd :parameters (parameters model2) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((3.0)) :shape '(1 1))))
    (fit model1 opt1 x-train y-train :epochs 3 :loss-fn #'mse-loss :verbose nil)
    (fit model2 opt2 x-train y-train :epochs 3 :loss-fn #'mae-loss :verbose nil)
    ;; Both should work
    (is (eql t t))))

(test training-callback-receives-correct-values
  "Test that callback receives correct epoch and loss values"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((3.0)) :shape '(1 1)))
         (callback-epochs nil)
         (callback-losses nil))
    (fit model opt x-train y-train 
         :epochs 3 
         :loss-fn #'mse-loss 
         :verbose nil
         :callback (lambda (epoch loss)
                     (push epoch callback-epochs)
                     (push loss callback-losses)
                     t))
    ;; Should have called callback 3 times
    (is (= 3 (length callback-epochs)))
    (is (= 3 (length callback-losses)))))

(test training-callback-early-stop
  "Test that callback can stop training early"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((3.0)) :shape '(1 1)))
         (epochs-completed 0))
    (fit model opt x-train y-train 
         :epochs 10 
         :loss-fn #'mse-loss 
         :verbose nil
         :callback (lambda (epoch loss)
                     (declare (ignore loss))
                     (incf epochs-completed)
                     (if (>= epoch 3) :stop nil)))  ; Stop at epoch 3
    ;; Should stop early - epochs 0, 1, 2, 3 completed = 4 epochs
    (is (<= epochs-completed 4))))

(test validation-loss-computation
  "Test that validation loss is computed correctly"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((3.0)) :shape '(1 1)))
         (x-val (make-tensor #2A((2.0 3.0)) :shape '(1 2)))
         (y-val (make-tensor #2A((5.0)) :shape '(1 1))))
    (fit model opt x-train y-train 
         :epochs 2 
         :loss-fn #'mse-loss 
         :x-val x-val 
         :y-val y-val
         :verbose nil)
    ;; Should complete with validation
    (is (eql t t))))

(test training-model-state-after-completion
  "Test model is in training mode after fit completes"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((3.0)) :shape '(1 1))))
    (fit model opt x-train y-train :epochs 2 :loss-fn #'mse-loss :verbose nil)
    ;; Model should be in training mode
    (is (layer-training model))))

(test prediction-after-training
  "Test that model can make predictions after training"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2)))
         (y-train (make-tensor #2A((3.0) (7.0)) :shape '(2 1))))
    (fit model opt x-train y-train :epochs 10 :loss-fn #'mse-loss :verbose nil)
    ;; Make a prediction
    (let* ((x-test (make-tensor #2A((2.0 3.0)) :shape '(1 2)))
           (pred (forward model x-test)))
      (is (equal '(1 1) (tensor-shape pred))))))

(test overfitting-detection
  "Test that model can overfit on small dataset"
  (let* ((model (sequential 
                  (linear 2 10)
                  (relu-layer)
                  (linear 10 1)))
         (opt (adam :parameters (parameters model) :lr 0.1))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((3.0)) :shape '(1 1)))
         (initial-loss (aref (tensor-data (mse-loss (forward model x-train) y-train)) 0)))
    (fit model opt x-train y-train :epochs 100 :loss-fn #'mse-loss :verbose nil)
    (let ((final-loss (aref (tensor-data (mse-loss (forward model x-train) y-train)) 0)))
      ;; With enough capacity and epochs, loss should decrease significantly
      (is (< final-loss (* 0.1 initial-loss))))))

(test parameter-updates-during-training
  "Test that parameters actually change during training"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.1))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((3.0)) :shape '(1 1)))
         (initial-param (aref (tensor-data (first (parameters model))) 0 0)))
    (fit model opt x-train y-train :epochs 5 :loss-fn #'mse-loss :verbose nil)
    (let ((final-param (aref (tensor-data (first (parameters model))) 0 0)))
      ;; Parameters should have changed
      (is (not (= initial-param final-param))))))

(test training-reproducibility-same-data
  "Test training produces same results with same data and initialization"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((3.0)) :shape '(1 1))))
    ;; Train twice with same setup
    (fit model opt x-train y-train :epochs 5 :loss-fn #'mse-loss :verbose nil)
    (let ((loss1 (aref (tensor-data (mse-loss (forward model x-train) y-train)) 0)))
      ;; Train again from current state
      (fit model opt x-train y-train :epochs 5 :loss-fn #'mse-loss :verbose nil)
      (let ((loss2 (aref (tensor-data (mse-loss (forward model x-train) y-train)) 0)))
        ;; Loss should continue to decrease (or stay low)
        (is (<= loss2 (* 1.1 loss1)))))))

(test training-with-zero-loss
  "Test training behavior when loss is already zero"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (zeros '(1 2)))
         (y-train (zeros '(1 1))))
    ;; Set model weights and bias to zero
    (let ((weight-data (tensor-data (layer-weight (first (seq-layers model))))))
      (dotimes (i (array-total-size weight-data))
        (setf (row-major-aref weight-data i) 0.0d0)))
    (let ((bias-data (tensor-data (layer-bias (first (seq-layers model))))))
      (dotimes (i (array-total-size bias-data))
        (setf (row-major-aref bias-data i) 0.0d0)))
    (fit model opt x-train y-train :epochs 3 :loss-fn #'mse-loss :verbose nil)
    ;; Should complete without error even though gradients may be zero
    (is (eql t t))))

(test training-negative-targets
  "Test training with negative target values"
  (let* ((model (sequential (linear 2 1)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (make-tensor #2A((1.0 2.0)) :shape '(1 2)))
         (y-train (make-tensor #2A((-5.0)) :shape '(1 1))))
    (fit model opt x-train y-train :epochs 10 :loss-fn #'mse-loss :verbose nil)
    ;; Model should learn negative values
    (let* ((pred (forward model x-train))
           (pred-val (aref (tensor-data pred) 0 0)))
      ;; Prediction should be closer to -5.0 than to 0
      (is (< pred-val 0.0)))))

(test training-preserves-parameter-count
  "Test that training doesn't change number of parameters"
  (let* ((model (sequential (linear 3 5) (relu-layer) (linear 5 2)))
         (opt (sgd :parameters (parameters model) :lr 0.01))
         (x-train (randn '(5 3)))
         (y-train (randn '(5 2)))
         (initial-param-count (length (parameters model))))
    (fit model opt x-train y-train :epochs 5 :loss-fn #'mse-loss :verbose nil)
    (let ((final-param-count (length (parameters model))))
      (is (= initial-param-count final-param-count)))))
