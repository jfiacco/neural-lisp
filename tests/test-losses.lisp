;;;; tests/test-losses.lisp - Loss Function Tests

(in-package #:neural-lisp-tests)

(def-suite loss-tests
  :description "Tests for loss functions"
  :in neural-lisp-tests)

(in-suite loss-tests)

(test mse-loss-zero
  "Test MSE loss with identical predictions and targets"
  (let* ((pred (make-tensor #(1.0 2.0 3.0) :shape '(3)))
         (target (make-tensor #(1.0 2.0 3.0) :shape '(3)))
         (loss (mse-loss pred target)))
    (is (< (abs (aref (tensor-data loss) 0)) 0.001))))

(test mse-loss-nonzero
  "Test MSE loss with different predictions and targets"
  (let* ((pred (make-tensor #(1.0 2.0 3.0) :shape '(3)))
         (target (make-tensor #(2.0 3.0 4.0) :shape '(3)))
         (loss (mse-loss pred target)))
    ;; MSE = mean((1-2)^2 + (2-3)^2 + (3-4)^2) = mean(1+1+1) = 1
    (is (< 0.99 (aref (tensor-data loss) 0) 1.01))))

(test mse-loss-gradient
  "Test MSE loss gradient"
  (let* ((pred (make-tensor #(2.0 3.0) :shape '(2) :requires-grad t))
         (target (make-tensor #(1.0 2.0) :shape '(2)))
         (loss (mse-loss pred target)))
    (backward loss)
    (is (not (null (tensor-grad pred))))))

(test mae-loss-zero
  "Test MAE loss with identical predictions and targets"
  (let* ((pred (make-tensor #(1.0 2.0 3.0) :shape '(3)))
         (target (make-tensor #(1.0 2.0 3.0) :shape '(3)))
         (loss (mae-loss pred target)))
    (is (< (abs (aref (tensor-data loss) 0)) 0.001))))

(test mae-loss-nonzero
  "Test MAE loss with different predictions and targets"
  (let* ((pred (make-tensor #(1.0 2.0 3.0) :shape '(3)))
         (target (make-tensor #(2.0 4.0 5.0) :shape '(3)))
         (loss (mae-loss pred target)))
    ;; MAE = mean(|1-2| + |2-4| + |3-5|) = mean(1+2+2) = 5/3 ≈ 1.667
    (is (< 1.6 (aref (tensor-data loss) 0) 1.7))))

(test cross-entropy-loss
  "Test cross-entropy loss"
  (let* ((logits (make-tensor #2A((2.0 1.0 0.1)) :shape '(1 3) :requires-grad t))
         (target (make-tensor #(0) :shape '(1)))  ; First class
         (loss (cross-entropy-loss logits target)))
    (is (> (aref (tensor-data loss) 0) 0.0))
    (backward loss)
    (is (not (null (tensor-grad logits))))))

(test binary-cross-entropy-loss
  "Test binary cross-entropy loss"
  (let* ((pred (make-tensor #(0.8 0.3 0.9) :shape '(3) :requires-grad t))
         (target (make-tensor #(1.0 0.0 1.0) :shape '(3)))
         (loss (binary-cross-entropy pred target)))
    (is (> (aref (tensor-data loss) 0) 0.0))
    (backward loss)
    (is (not (null (tensor-grad pred))))))

(test huber-loss-small-error
  "Test Huber loss with small error (quadratic region)"
  (let* ((pred (make-tensor #(1.0 2.0) :shape '(2)))
         (target (make-tensor #(1.1 2.1) :shape '(2)))
         (loss (huber-loss pred target :delta 1.0)))
    ;; Error is 0.1, less than delta, should be quadratic
    (is (< (aref (tensor-data loss) 0) 0.1))))

(test huber-loss-large-error
  "Test Huber loss with large error (linear region)"
  (let* ((pred (make-tensor #(1.0) :shape '(1)))
         (target (make-tensor #(5.0) :shape '(1)))
         (loss (huber-loss pred target :delta 1.0)))
    ;; Error is 4.0, greater than delta, should be linear
    (is (> (aref (tensor-data loss) 0) 1.0))))

(test smooth-l1-loss
  "Test Smooth L1 loss"
  (let* ((pred (make-tensor #(1.0 2.0 3.0) :shape '(3)))
         (target (make-tensor #(1.5 2.5 3.5) :shape '(3)))
         (loss (smooth-l1-loss pred target)))
    (is (> (aref (tensor-data loss) 0) 0.0))))

(test kl-divergence-loss
  "Test KL divergence loss"
  (let* ((p (make-tensor #(0.5 0.3 0.2) :shape '(3)))
         (q (make-tensor #(0.4 0.4 0.2) :shape '(3)))
         (loss (kl-divergence p q)))
    (is (>= (aref (tensor-data loss) 0) 0.0))))

(test nll-loss
  "Test negative log-likelihood loss"
  (let* ((log-probs (make-tensor #2A((-0.5 -1.0 -2.0)) :shape '(1 3)))
         (target (make-tensor #(0) :shape '(1)))
         (loss (nll-loss log-probs target)))
    (is (< 0.49 (aref (tensor-data loss) 0) 0.51))))

(test loss-reduction-mean
  "Test that losses use mean reduction by default"
  (let* ((pred (make-tensor #(1.0 2.0) :shape '(2)))
         (target (make-tensor #(2.0 3.0) :shape '(2)))
         (loss (mse-loss pred target)))
    ;; Should be scalar output
    (is (equal '(1) (tensor-shape loss)))))

(test loss-batch-dimension
  "Test loss with batch dimension"
  (let* ((pred (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2)))
         (target (make-tensor #2A((2.0 3.0) (4.0 5.0)) :shape '(2 2)))
         (loss (mse-loss pred target)))
    ;; Should average over all elements
    (is (equal '(1) (tensor-shape loss)))))

;;; Edge Cases and Robustness Tests

(test mse-loss-single-element
  "Test MSE loss with single element"
  (let* ((pred (make-tensor #(5.0) :shape '(1)))
         (target (make-tensor #(3.0) :shape '(1)))
         (loss (mse-loss pred target)))
    ;; (5-3)^2 = 4
    (is (< 3.99 (aref (tensor-data loss) 0) 4.01))))

(test mse-loss-all-zeros
  "Test MSE loss with all zero predictions and targets"
  (let* ((pred (zeros '(5)))
         (target (zeros '(5)))
         (loss (mse-loss pred target)))
    (is (< (abs (aref (tensor-data loss) 0)) 0.001))))

(test mse-loss-large-errors
  "Test MSE loss with large errors"
  (let* ((pred (make-tensor #(100.0) :shape '(1)))
         (target (make-tensor #(0.0) :shape '(1)))
         (loss (mse-loss pred target)))
    ;; 100^2 = 10000
    (is (> (aref (tensor-data loss) 0) 9999.0))))

(test mse-loss-negative-values
  "Test MSE loss with negative predictions and targets"
  (let* ((pred (make-tensor #(-2.0 -3.0) :shape '(2)))
         (target (make-tensor #(-1.0 -4.0) :shape '(2)))
         (loss (mse-loss pred target)))
    ;; mean((-2-(-1))^2 + (-3-(-4))^2) = mean(1 + 1) = 1
    (is (< 0.99 (aref (tensor-data loss) 0) 1.01))))

(test mse-loss-symmetry
  "Test MSE loss symmetry: loss(a,b) = loss(b,a)"
  (let* ((pred (make-tensor #(1.0 2.0) :shape '(2)))
         (target (make-tensor #(3.0 4.0) :shape '(2)))
         (loss1 (mse-loss pred target))
         (loss2 (mse-loss target pred)))
    (is (< (abs (- (aref (tensor-data loss1) 0) 
                   (aref (tensor-data loss2) 0))) 1.0e-10))))

(test mae-loss-single-element
  "Test MAE loss with single element"
  (let* ((pred (make-tensor #(7.0) :shape '(1)))
         (target (make-tensor #(3.0) :shape '(1)))
         (loss (mae-loss pred target)))
    ;; |7-3| = 4
    (is (< 3.99 (aref (tensor-data loss) 0) 4.01))))

(test mae-loss-with-negative
  "Test MAE loss with negative values"
  (let* ((pred (make-tensor #(-5.0 -3.0) :shape '(2)))
         (target (make-tensor #(-2.0 -7.0) :shape '(2)))
         (loss (mae-loss pred target)))
    ;; mean(|-5-(-2)| + |-3-(-7)|) = mean(3 + 4) = 3.5
    (is (< 3.49 (aref (tensor-data loss) 0) 3.51))))

(test mae-loss-all-zeros
  "Test MAE loss with all zeros"
  (let* ((pred (zeros '(10)))
         (target (zeros '(10)))
         (loss (mae-loss pred target)))
    (is (< (abs (aref (tensor-data loss) 0)) 0.001))))

(test cross-entropy-single-class
  "Test cross-entropy with single class prediction"
  (let* ((logits (make-tensor #2A((5.0)) :shape '(1 1)))
         (target (make-tensor #(0) :shape '(1)))
         (loss (cross-entropy-loss logits target)))
    ;; Should be close to 0 for high confidence correct prediction
    (is (< (aref (tensor-data loss) 0) 1.0))))

(test cross-entropy-multiple-samples
  "Test cross-entropy with multiple samples"
  (let* ((logits (make-tensor #2A((2.0 1.0 0.1)
                                  (0.1 2.0 1.0)) :shape '(2 3)))
         (target (make-tensor #(0 1) :shape '(2)))
         (loss (cross-entropy-loss logits target)))
    (is (> (aref (tensor-data loss) 0) 0.0))))

(test binary-cross-entropy-perfect-prediction
  "Test binary cross-entropy with perfect predictions"
  (let* ((pred (make-tensor #(0.9999 0.0001) :shape '(2)))
         (target (make-tensor #(1.0 0.0) :shape '(2)))
         (loss (binary-cross-entropy pred target)))
    ;; Should be very small for near-perfect predictions
    (is (< (aref (tensor-data loss) 0) 0.01))))

(test binary-cross-entropy-worst-prediction
  "Test binary cross-entropy with worst predictions"
  (let* ((pred (make-tensor #(0.0001 0.9999) :shape '(2)))
         (target (make-tensor #(1.0 0.0) :shape '(2)))
         (loss (binary-cross-entropy pred target)))
    ;; Should be large for wrong predictions
    (is (> (aref (tensor-data loss) 0) 5.0))))

(test binary-cross-entropy-boundary
  "Test binary cross-entropy at prediction boundaries"
  (let* ((pred (make-tensor #(0.5 0.5) :shape '(2)))
         (target (make-tensor #(1.0 0.0) :shape '(2)))
         (loss (binary-cross-entropy pred target)))
    ;; Maximum uncertainty (0.5) should give same loss for both classes
    (is (> (aref (tensor-data loss) 0) 0.6))
    (is (< (aref (tensor-data loss) 0) 0.8))))

(test huber-loss-at-delta
  "Test Huber loss exactly at delta boundary"
  (let* ((pred (make-tensor #(0.0) :shape '(1)))
         (target (make-tensor #(1.0) :shape '(1)))
         (loss (huber-loss pred target :delta 1.0)))
    ;; Error is exactly delta, should be at transition point
    (is (> (aref (tensor-data loss) 0) 0.0))))

(test huber-loss-zero-delta
  "Test Huber loss with very small delta"
  (let* ((pred (make-tensor #(1.0) :shape '(1)))
         (target (make-tensor #(1.5) :shape '(1)))
         (loss (huber-loss pred target :delta 0.1)))
    (is (> (aref (tensor-data loss) 0) 0.0))))

(test smooth-l1-transition
  "Test Smooth L1 loss transition from quadratic to linear"
  (let* ((pred1 (make-tensor #(0.0) :shape '(1)))
         (target1 (make-tensor #(0.5) :shape '(1)))
         (loss1 (smooth-l1-loss pred1 target1))
         (pred2 (make-tensor #(0.0) :shape '(1)))
         (target2 (make-tensor #(2.0) :shape '(1)))
         (loss2 (smooth-l1-loss pred2 target2)))
    ;; Small error should be less than large error
    (is (< (aref (tensor-data loss1) 0) (aref (tensor-data loss2) 0)))))

(test kl-divergence-identical-distributions
  "Test KL divergence with identical distributions"
  (let* ((p (make-tensor #(0.3 0.3 0.4) :shape '(3)))
         (q (make-tensor #(0.3 0.3 0.4) :shape '(3)))
         (loss (kl-divergence p q)))
    ;; KL(p||p) = 0
    (is (< (abs (aref (tensor-data loss) 0)) 0.01))))

(test kl-divergence-different-distributions
  "Test KL divergence with different distributions"
  (let* ((p (make-tensor #(0.9 0.05 0.05) :shape '(3)))
         (q (make-tensor #(0.33 0.33 0.34) :shape '(3)))
         (loss (kl-divergence p q)))
    ;; Should be positive and non-zero
    (is (> (aref (tensor-data loss) 0) 0.0))))

(test kl-divergence-asymmetry
  "Test KL divergence is asymmetric"
  (let* ((p (make-tensor #(0.9 0.1) :shape '(2)))
         (q (make-tensor #(0.2 0.8) :shape '(2)))
         (kl-pq (kl-divergence p q))
         (kl-qp (kl-divergence q p)))
    ;; KL(p||q) != KL(q||p) - should differ significantly with asymmetric distributions
    (is (not (< (abs (- (aref (tensor-data kl-pq) 0) 
                        (aref (tensor-data kl-qp) 0))) 0.01)))))

(test nll-loss-correct-class
  "Test NLL loss for correct class with high confidence"
  (let* ((log-probs (make-tensor #2A((-0.01 -10.0 -10.0)) :shape '(1 3)))
         (target (make-tensor #(0) :shape '(1)))
         (loss (nll-loss log-probs target)))
    ;; Should be small for high confidence
    (is (< (aref (tensor-data loss) 0) 0.1))))

(test nll-loss-wrong-class
  "Test NLL loss for wrong class with high confidence"
  (let* ((log-probs (make-tensor #2A((-10.0 -0.01 -10.0)) :shape '(1 3)))
         (target (make-tensor #(0) :shape '(1)))
         (loss (nll-loss log-probs target)))
    ;; Should be large for low confidence in correct class
    (is (> (aref (tensor-data loss) 0) 5.0))))

(test nll-loss-batch
  "Test NLL loss with batch of samples"
  (let* ((log-probs (make-tensor #2A((-0.5 -1.0) 
                                     (-1.5 -0.3)) :shape '(2 2)))
         (target (make-tensor #(0 1) :shape '(2)))
         (loss (nll-loss log-probs target)))
    ;; Should average over batch
    (is (equal '(1) (tensor-shape loss)))))

(test loss-positive-values
  "Test that all loss functions return positive values"
  (let* ((pred (make-tensor #(0.5 1.5 2.5) :shape '(3)))
         (target (make-tensor #(1.0 2.0 3.0) :shape '(3)))
         (mse (mse-loss pred target))
         (mae (mae-loss pred target)))
    (is (>= (aref (tensor-data mse) 0) 0.0))
    (is (>= (aref (tensor-data mae) 0) 0.0))))

(test loss-gradient-existence
  "Test that loss gradients can be computed"
  (let* ((pred (make-tensor #(1.0 2.0) :shape '(2) :requires-grad t))
         (target (make-tensor #(1.5 2.5) :shape '(2)))
         (loss (mse-loss pred target)))
    (backward loss)
    (is (not (null (tensor-grad pred))))
    (is (not (= 0.0 (aref (tensor-grad pred) 0))))))

(test loss-scalar-output
  "Test that all losses return scalar output"
  (let* ((pred (make-tensor #2A((1.0 2.0) (3.0 4.0)) :shape '(2 2)))
         (target (make-tensor #2A((1.5 2.5) (3.5 4.5)) :shape '(2 2)))
         (mse (mse-loss pred target))
         (mae (mae-loss pred target)))
    (is (equal '(1) (tensor-shape mse)))
    (is (equal '(1) (tensor-shape mae)))))

(test mse-vs-mae-comparison
  "Test MSE penalizes large errors more than MAE"
  (let* ((pred (make-tensor #(0.0 0.0) :shape '(2)))
         (target (make-tensor #(1.0 10.0) :shape '(2)))
         (mse (mse-loss pred target))
         (mae (mae-loss pred target)))
    ;; MSE = mean(1 + 100) = 50.5
    ;; MAE = mean(1 + 10) = 5.5
    ;; MSE should be much larger
    (is (> (aref (tensor-data mse) 0) (aref (tensor-data mae) 0)))))

(test loss-numerical-stability-small-values
  "Test loss functions with very small values"
  (let* ((pred (make-tensor #(1.0e-10 1.0e-10) :shape '(2)))
         (target (make-tensor #(2.0e-10 3.0e-10) :shape '(2)))
         (loss (mse-loss pred target)))
    ;; Should not overflow or become NaN
    (is (numberp (aref (tensor-data loss) 0)))))

(test loss-numerical-stability-large-values
  "Test loss functions with very large values"
  (let* ((pred (make-tensor #(1.0e10) :shape '(1)))
         (target (make-tensor #(2.0e10) :shape '(1)))
         (loss (mae-loss pred target)))
    ;; Should handle large values
    (is (numberp (aref (tensor-data loss) 0)))))

(test huber-loss-parameter-sensitivity
  "Test Huber loss with different delta values"
  (let* ((pred (make-tensor #(0.0) :shape '(1)))
         (target (make-tensor #(2.0) :shape '(1)))
         (loss1 (huber-loss pred target :delta 0.5))
         (loss2 (huber-loss pred target :delta 5.0)))
    ;; Different deltas should give different losses
    (is (not (= (aref (tensor-data loss1) 0) (aref (tensor-data loss2) 0))))))
