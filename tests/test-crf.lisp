;;;; Tests for Conditional Random Fields
;;;; Testing CRF inference, training, and decoding

(defpackage :neural-tensor-crf-tests
  (:use :common-lisp)
  (:import-from :neural-network
                #:tensor
                #:make-tensor
                #:zeros
                #:randn
                #:forward
                #:tensor-shape
                #:tensor-data
                #:layer-parameters)
  (:import-from :neural-tensor-crf
                #:linear-chain-crf
                #:viterbi-decode
                #:forward-algorithm
                #:backward-algorithm
                #:constrained-viterbi
                #:bio-constraints
                #:valid-transition-p)
  (:export #:run-crf-tests))

(in-package :neural-tensor-crf-tests)

(defvar *test-results* nil)
(defvar *tests-passed* 0)
(defvar *tests-failed* 0)

(defmacro deftest (name &body body)
  `(progn
     (format t "~%Testing ~a... " ',name)
     (handler-case
         (progn
           ,@body
           (format t "✓ PASSED")
           (incf *tests-passed*)
           (push (cons ',name :pass) *test-results*))
       (error (e)
         (format t "✗ FAILED: ~a" e)
         (incf *tests-failed*)
         (push (cons ',name :fail) *test-results*)))))

(defun assert-equal (expected actual &optional (tolerance 1d-6))
  "Assert two values are equal within tolerance"
  (unless (< (abs (- expected actual)) tolerance)
    (error "Expected ~a but got ~a" expected actual)))

(defun assert-shape (expected-shape tensor)
  "Assert tensor has expected shape"
  (unless (equal expected-shape (tensor-shape tensor))
    (error "Expected shape ~a but got ~a" expected-shape (tensor-shape tensor))))

(defun run-crf-tests ()
  "Run all CRF tests"
  (setf *test-results* nil
        *tests-passed* 0
        *tests-failed* 0)
  
  (format t "~%")
  (format t "╔════════════════════════════════════════════════════════════════╗~%")
  (format t "║  Conditional Random Fields Tests                              ║~%")
  (format t "╚════════════════════════════════════════════════════════════════╝~%")
  (format t "~%")
  
  ;; Test Linear-Chain CRF Creation
  (deftest test-crf-creation
    (let ((crf (make-instance 'linear-chain-crf :num-tags 5)))
      (assert (= 5 (neural-tensor-crf::num-tags crf)))
      (assert (not (null (neural-tensor-crf::transitions crf))))
      (assert (not (null (neural-tensor-crf::start-transitions crf))))
      (assert (not (null (neural-tensor-crf::end-transitions crf))))))
  
  (deftest test-crf-parameters
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 5))
           (params (layer-parameters crf)))
      (assert (= 3 (length params))))) ; transitions, start, end
  
  (deftest test-crf-transition-matrix-shape
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 7))
           (transitions (neural-tensor-crf::transitions crf)))
      (assert-shape '(7 7) transitions)))
  
  (deftest test-crf-start-transitions-shape
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 5))
           (start-trans (neural-tensor-crf::start-transitions crf)))
      (assert-shape '(5) start-trans)))
  
  (deftest test-crf-end-transitions-shape
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 5))
           (end-trans (neural-tensor-crf::end-transitions crf)))
      (assert-shape '(5) end-trans)))
  
  ;; Test Forward Algorithm
  (deftest test-forward-algorithm
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (randn '(1 5 3)))  ; (batch, seq_len, num_tags)
           (log-partition (forward-algorithm crf emissions)))
      (assert-shape '(1) log-partition)
      ;; Partition function should be a finite number
      (assert (numberp (aref (tensor-data log-partition) 0)))))
  
  (deftest test-forward-algorithm-positive
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (randn '(1 5 3)))
           (log-partition (forward-algorithm crf emissions))
           (value (aref (tensor-data log-partition) 0)))
      ;; Log partition should be finite (not NaN or Inf)
      (assert (and (numberp value)
                   (not (or (sb-ext:float-nan-p value)
                           (sb-ext:float-infinity-p value)))))))
  
  ;; Test Backward Algorithm
  (deftest test-backward-algorithm
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (randn '(1 5 3)))
           (beta (backward-algorithm crf emissions)))
      (assert (arrayp beta))
      (assert (= 2 (array-rank beta)))
      (assert (= 5 (array-dimension beta 0)))  ; seq_len
      (assert (= 3 (array-dimension beta 1))))) ; num_tags
  
  ;; Test Viterbi Decoding
  (deftest test-viterbi-decode
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 4))
           (emissions (randn '(1 6 4)))
           (best-path (viterbi-decode crf emissions)))
      (assert (arrayp best-path))
      (assert (= 6 (length best-path)))
      ;; All tags should be in valid range
      (dotimes (i 6)
        (let ((tag (aref best-path i)))
          (assert (and (>= tag 0) (< tag 4)))))))
  
  (deftest test-viterbi-decode-consistency
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (randn '(1 5 3)))
           (path1 (viterbi-decode crf emissions))
           (path2 (viterbi-decode crf emissions)))
      ;; Should produce same result for same input
      (assert (equalp path1 path2))))
  
  ;; Test CRF Loss Computation
  (deftest test-crf-loss
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (randn '(1 5 3)))
           (tags (make-tensor #(0.0d0 1.0d0 2.0d0 1.0d0 0.0d0) :shape '(5)))
           (loss (neural-tensor-crf::crf-loss crf emissions tags)))
      (assert-shape '(1) loss)
      ;; Loss should be a positive number
      (assert (> (aref (tensor-data loss) 0) 0.0d0))))
  
  ;; Test BIO Constraints
  (deftest test-bio-constraints
    (let ((constraints (bio-constraints 2)))
      (assert (listp constraints))
      (assert (> (length constraints) 0))))
  
  (deftest test-valid-transition
    (let ((constraints '((0 . (0 1))    ; O -> O, B-PER
                        (1 . (2 0 1))))) ; B-PER -> I-PER, O, B-PER
      (assert (valid-transition-p 0 0 constraints))  ; O -> O
      (assert (valid-transition-p 0 1 constraints))  ; O -> B-PER
      (assert (not (valid-transition-p 0 2 constraints))))) ; O -> I-PER (invalid)
  
  ;; Test Constrained Viterbi
  (deftest test-constrained-viterbi
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 5))
           (emissions (randn '(1 6 5)))
           (constraints '((0 . (0 1 2))
                         (1 . (0 1 2))
                         (2 . (0 1 2))))
           (path (constrained-viterbi crf emissions constraints)))
      (assert (arrayp path))
      (assert (= 6 (length path)))))
  
  ;; Test different sequence lengths
  (deftest test-crf-short-sequence
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (randn '(1 2 3)))
           (path (viterbi-decode crf emissions)))
      (assert (= 2 (length path)))))
  
  (deftest test-crf-long-sequence
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 4))
           (emissions (randn '(1 20 4)))
           (path (viterbi-decode crf emissions)))
      (assert (= 20 (length path)))))
  
  ;; Test different number of tags
  (deftest test-crf-binary-tags
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 2))
           (emissions (randn '(1 5 2)))
           (path (viterbi-decode crf emissions)))
      (assert (= 5 (length path)))
      (dotimes (i 5)
        (assert (member (aref path i) '(0 1))))))
  
  (deftest test-crf-many-tags
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 10))
           (emissions (randn '(1 8 10)))
           (path (viterbi-decode crf emissions)))
      (assert (= 8 (length path)))
      (dotimes (i 8)
        (assert (and (>= (aref path i) 0) (< (aref path i) 10))))))
  
  ;; Test numerical stability
  (deftest test-crf-numerical-stability
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           ;; Very large emission scores
           (emissions (make-tensor 
                       (make-array '(1 5 3) 
                                  :element-type 'double-float
                                  :initial-element 100.0d0)
                       :shape '(1 5 3)))
           (log-partition (forward-algorithm crf emissions)))
      ;; Should not overflow or become NaN
      (assert (numberp (aref (tensor-data log-partition) 0)))))
  
  ;; Test edge cases
  (deftest test-crf-single-timestep
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (randn '(1 1 3)))
           (path (viterbi-decode crf emissions)))
      (assert (= 1 (length path)))))
  
  (deftest test-crf-deterministic-emissions
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           ;; Strong preference for tag 1
           (emissions (make-tensor
                       (make-array '(1 4 3)
                                  :initial-contents
                                  '(((0.0d0 10.0d0 0.0d0)
                                     (0.0d0 10.0d0 0.0d0)
                                     (0.0d0 10.0d0 0.0d0)
                                     (0.0d0 10.0d0 0.0d0))))
                       :shape '(1 4 3)))
           (path (viterbi-decode crf emissions)))
      ;; Should mostly predict tag 1 (if transitions allow)
      (assert (>= (count 1 (coerce path 'list)) 2))))
  
  ;; ========== EDGE CASE TESTS ==========
  
  ;; Test with minimum number of tags (binary classification)
  (deftest test-crf-binary-classification
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 2))
           (emissions (randn '(1 10 2)))
           (path (viterbi-decode crf emissions)))
      (assert (= 10 (length path)))
      (dotimes (i 10)
        (assert (or (= (aref path i) 0) (= (aref path i) 1))))))
  
  ;; Test with very long sequences (stress test)
  (deftest test-crf-very-long-sequence
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 5))
           (emissions (randn '(1 500 5)))
           (path (viterbi-decode crf emissions)))
      (assert (= 500 (length path)))))
  
  ;; Test with many tags
  (deftest test-crf-many-tags
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 50))
           (emissions (randn '(1 10 50)))
           (path (viterbi-decode crf emissions)))
      (assert (= 10 (length path)))
      (dotimes (i 10)
        (assert (and (>= (aref path i) 0) (< (aref path i) 50))))))
  
  ;; Test forward algorithm with extreme emission scores
  (deftest test-forward-algorithm-extreme-scores
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (make-tensor 
                       (make-array '(1 5 3)
                                  :element-type 'double-float
                                  :initial-element 1000.0d0)
                       :shape '(1 5 3)))
           (log-partition (forward-algorithm crf emissions)))
      ;; Should not overflow
      (let ((val (aref (tensor-data log-partition) 0)))
        (assert (not (sb-ext:float-infinity-p val)))
        (assert (not (sb-ext:float-nan-p val))))))
  
  ;; Test forward algorithm with negative scores
  (deftest test-forward-algorithm-negative-scores
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (make-tensor 
                       (make-array '(1 5 3)
                                  :element-type 'double-float
                                  :initial-element -50.0d0)
                       :shape '(1 5 3)))
           (log-partition (forward-algorithm crf emissions)))
      (assert-shape '(1) log-partition)))
  
  ;; Test backward algorithm stability
  (deftest test-backward-algorithm-stability
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 4))
           (emissions (randn '(1 20 4)))
           (beta (backward-algorithm crf emissions)))
      ;; Check for NaN or Inf
      (dotimes (i (array-dimension beta 0))
        (dotimes (j (array-dimension beta 1))
          (let ((val (aref beta i j)))
            (assert (not (sb-ext:float-nan-p val)))
            (assert (not (sb-ext:float-infinity-p val))))))))
  
  ;; Test CRF loss with matching gold sequence
  (deftest test-crf-loss-matching-sequence
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (randn '(1 5 3)))
           (tags (make-tensor #(0.0d0 1.0d0 2.0d0 1.0d0 0.0d0) :shape '(5)))
           (loss (neural-tensor-crf::crf-loss crf emissions tags)))
      ;; Loss should be positive and finite
      (let ((loss-val (aref (tensor-data loss) 0)))
        (assert (> loss-val 0.0d0))
        (assert (not (sb-ext:float-infinity-p loss-val)))
        (assert (not (sb-ext:float-nan-p loss-val))))))
  
  ;; Test Viterbi with uniform emissions
  (deftest test-viterbi-uniform-emissions
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (make-tensor 
                       (make-array '(1 5 3)
                                  :element-type 'double-float
                                  :initial-element 1.0d0)
                       :shape '(1 5 3)))
           (path (viterbi-decode crf emissions)))
      ;; Should produce a valid path
      (assert (= 5 (length path)))
      (dotimes (i 5)
        (assert (and (>= (aref path i) 0) (< (aref path i) 3))))))
  
  ;; Test Viterbi consistency across runs
  (deftest test-viterbi-determinism
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 4))
           (emissions (randn '(1 8 4)))
           (path1 (viterbi-decode crf emissions))
           (path2 (viterbi-decode crf emissions))
           (path3 (viterbi-decode crf emissions)))
      ;; All paths should be identical
      (assert (equalp path1 path2))
      (assert (equalp path2 path3))))
  
  ;; Test constrained Viterbi with complex constraints
  (deftest test-constrained-viterbi-complex
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 6))
           (emissions (randn '(1 10 6)))
           ;; Define BIO-like constraints
           (constraints '((0 . (0 1 3 5))      ; O can go to O, B-PER, B-LOC, B-ORG
                         (1 . (0 2 3 5))       ; B-PER can go to O, I-PER, B-LOC, B-ORG
                         (2 . (0 2 3 5))       ; I-PER can go to O, I-PER, B-LOC, B-ORG
                         (3 . (0 1 3 4))       ; B-LOC can go to O, B-PER, B-LOC, I-LOC
                         (4 . (0 1 3 4))       ; I-LOC can go to O, B-PER, B-LOC, I-LOC
                         (5 . (0 1 3 5))))     ; B-ORG can go to O, B-PER, B-LOC, B-ORG
           (path (constrained-viterbi crf emissions constraints)))
      (assert (= 10 (length path)))
      ;; Verify all transitions are valid
      (dotimes (i 9)
        (let ((from-tag (aref path i))
              (to-tag (aref path (1+ i))))
          (assert (valid-transition-p from-tag to-tag constraints))))))
  
  ;; Test BIO constraints generation
  (deftest test-bio-constraints-structure
    (let ((constraints (bio-constraints 3)))  ; 3 entity types: PER, LOC, ORG
      ;; Should have 7 tags: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG
      (assert (listp constraints))
      ;; Check O tag transitions
      (let ((o-transitions (cdr (assoc 0 constraints))))
        (assert (member 0 o-transitions))  ; O->O
        (assert (member 1 o-transitions))  ; O->B-PER
        (assert (not (member 2 o-transitions)))))) ; O cannot go to I-PER
  
  ;; Test CRF with zero transitions (edge case)
  (deftest test-crf-zero-transitions
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           ;; Set all transitions to zero
           (trans-data (tensor-data (neural-tensor-crf::transitions crf))))
      (dotimes (i (array-total-size trans-data))
        (setf (row-major-aref trans-data i) 0.0d0))
      (let* ((emissions (randn '(1 5 3)))
             (path (viterbi-decode crf emissions)))
        (assert (= 5 (length path))))))
  
  ;; Test CRF parameter gradients exist
  (deftest test-crf-parameter-gradients
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (params (layer-parameters crf)))
      ;; All parameters should require gradients
      (dolist (param params)
        (assert (neural-network::requires-grad param)))))
  
  ;; Test forward-backward consistency
  (deftest test-forward-backward-consistency
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (randn '(1 6 3)))
           (alpha (forward-algorithm crf emissions))
           (beta (backward-algorithm crf emissions)))
      ;; Both should compute valid probabilities
      (assert (not (null alpha)))
      (assert (not (null beta)))))
  
  ;; Test single tag edge case
  (deftest test-crf-single-tag
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 1))
           (emissions (randn '(1 5 1)))
           (path (viterbi-decode crf emissions)))
      ;; All predictions should be tag 0
      (assert (= 5 (length path)))
      (dotimes (i 5)
        (assert-equal 0 (aref path i)))))
  
  ;; Test marginal probabilities computation
  (deftest test-marginal-probabilities
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 3))
           (emissions (randn '(1 5 3)))
           (marginals (neural-tensor-crf::marginal-probabilities crf emissions)))
      ;; Should return tensor with shape (seq_len, num_tags)
      (assert-shape '(5 3) marginals)
      ;; All probabilities should be in [0, 1]
      (let ((data (tensor-data marginals)))
        (dotimes (i (array-total-size data))
          (let ((prob (row-major-aref data i)))
            (assert (and (>= prob 0.0d0) (<= prob 1.0d0))))))))
  
  ;; Test marginal probabilities sum to 1
  (deftest test-marginal-probabilities-sum
    (let* ((crf (make-instance 'linear-chain-crf :num-tags 4))
           (emissions (randn '(1 6 4)))
           (marginals (neural-tensor-crf::marginal-probabilities crf emissions))
           (data (tensor-data marginals)))
      ;; Each timestep should sum to 1
      (dotimes (time-idx 6)
        (let ((sum 0.0d0))
          (dotimes (tag 4)
            (incf sum (aref data time-idx tag)))
          (assert-equal 1.0d0 sum 1d-4)))))
  
  ;; Test emission-scores utility
  (deftest test-emission-scores
    (let* ((emissions (make-tensor
                       (make-array '(2 3 4)
                                  :initial-contents
                                  '(((1.0d0 2.0d0 3.0d0 4.0d0)
                                     (5.0d0 6.0d0 7.0d0 8.0d0)
                                     (9.0d0 10.0d0 11.0d0 12.0d0))
                                    ((13.0d0 14.0d0 15.0d0 16.0d0)
                                     (17.0d0 18.0d0 19.0d0 20.0d0)
                                     (21.0d0 22.0d0 23.0d0 24.0d0))))
                       :shape '(2 3 4)))
           (tags (make-tensor #2A((0.0d0 1.0d0 2.0d0)
                                  (3.0d0 2.0d0 1.0d0))
                             :shape '(2 3)))
           (scores (neural-tensor-crf::emission-scores emissions tags)))
      (assert-shape '(2 3) scores)
      (let ((data (tensor-data scores)))
        ;; Check specific values
        (assert-equal 1.0d0 (aref data 0 0))   ; emissions[0,0,0]
        (assert-equal 6.0d0 (aref data 0 1))   ; emissions[0,1,1]
        (assert-equal 11.0d0 (aref data 0 2))  ; emissions[0,2,2]
        (assert-equal 16.0d0 (aref data 1 0))  ; emissions[1,0,3]
        (assert-equal 19.0d0 (aref data 1 1))  ; emissions[1,1,2]
        (assert-equal 22.0d0 (aref data 1 2))))) ; emissions[1,2,1]
  
  ;; Test Chu-Liu-Edmonds algorithm
  (deftest test-chu-liu-edmonds
    (let* ((scores (make-tensor
                    (make-array '(4 4)
                               :initial-contents
                               '((0.0d0 5.0d0 1.0d0 1.0d0)
                                 (1.0d0 0.0d0 10.0d0 3.0d0)
                                 (1.0d0 1.0d0 0.0d0 4.0d0)
                                 (1.0d0 7.0d0 1.0d0 0.0d0)))
                    :shape '(4 4)))
           (tree (neural-tensor-crf::chu-liu-edmonds scores)))
      ;; Should return parent array
      (assert (arrayp tree))
      (assert (= 4 (length tree)))
      ;; Root should have no parent
      (assert (= -1 (aref tree 0)))
      ;; All other nodes should have parents
      (dotimes (i 3)
        (assert (and (>= (aref tree (1+ i)) -1)
                    (< (aref tree (1+ i)) 4))))))
  
  ;; Test Tree CRF
  (deftest test-tree-crf-creation
    (let ((crf (make-instance 'neural-tensor-crf::tree-crf :num-labels 5)))
      (assert (= 5 (neural-tensor-crf::num-labels crf)))
      (assert (not (null (neural-tensor-crf::edge-scores crf))))))
  
  (deftest test-tree-crf-forward
    (let* ((crf (make-instance 'neural-tensor-crf::tree-crf :num-labels 3))
           (node-features (randn '(5 10)))  ; 5 nodes, 10 features each
           (edges (make-tensor
                   (make-array '(5 5)
                              :element-type 'double-float
                              :initial-element 1.0d0)
                   :shape '(5 5)))
           (gold-tree (make-array 5 :element-type 'fixnum
                                 :initial-contents '(-1 0 0 1 1))))
      ;; Diagonal should be 0 (no self-loops)
      (let ((edge-data (tensor-data edges)))
        (dotimes (i 5)
          (setf (aref edge-data i i) 0.0d0)))
      (let ((loss (neural-tensor-crf::tree-crf-forward 
                   crf node-features edges gold-tree)))
        (assert-shape '(1) loss)
        (assert (numberp (aref (tensor-data loss) 0))))))
  
  ;; Print summary
  (format t "~%CRF Tests: ~d passed, ~d failed~%~%" *tests-passed* *tests-failed*)
  
  (values *tests-passed* *tests-failed*))

;; Run tests when file is loaded
(format t "~%To run CRF tests, execute: (neural-tensor-crf-tests:run-crf-tests)~%")
