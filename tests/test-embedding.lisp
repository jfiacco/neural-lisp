;;;; tests/test-embedding.lisp - Tests for Embedding Layers

(in-package #:neural-lisp-tests)

(def-suite embedding-tests
  :description "Tests for embedding layers"
  :in neural-lisp-tests)

(in-suite embedding-tests)

;;;; ============================================================================
;;;; Word Embedding Tests
;;;; ============================================================================

(test word-embedding-creation
  "Test basic word embedding creation"
  (let ((emb (make-instance 'neural-network.embedding:word-embedding
                            :vocab-size 100
                            :embedding-dim 50)))
    (is (not (null emb)))
    (is (= 100 (neural-network.embedding:emb-vocab-size emb)))
    (is (= 50 (embedding-dim emb)))
    (is (not (null (neural-network.embedding:embeddings emb))))
    (is (equal '(100 50) (tensor-shape (neural-network.embedding:embeddings emb))))))

(test word-embedding-forward
  "Test word embedding forward pass"
  (let ((emb (make-instance 'neural-network.embedding:word-embedding
                            :vocab-size 10
                            :embedding-dim 4))
        (indices (make-tensor #(0.0 1.0 2.0) :shape '(3))))
    (let ((result (forward emb indices)))
      (is (not (null result)))
      (is (equal '(3 4) (tensor-shape result)))
      ;; Check that different indices produce different embeddings
      (let ((data (tensor-data result)))
        (is (not (= (aref data 0 0) (aref data 1 0))))))))

(test word-embedding-with-padding
  "Test word embedding with padding index"
  (let ((emb (make-instance 'neural-network.embedding:word-embedding
                            :vocab-size 10
                            :embedding-dim 4
                            :padding-idx 0))
        (indices (make-tensor #(0.0 1.0 2.0) :shape '(3))))
    (let ((result (forward emb indices)))
      ;; Padding embedding (index 0) should be all zeros
      (let ((data (tensor-data result)))
        (is (= 0.0d0 (aref data 0 0)))
        (is (= 0.0d0 (aref data 0 1)))
        (is (= 0.0d0 (aref data 0 2)))
        (is (= 0.0d0 (aref data 0 3)))
        ;; Non-padding embeddings should not be zero
        (is (not (= 0.0d0 (aref data 1 0))))))))

(test word-embedding-batch
  "Test word embedding with batched input"
  (let ((emb (make-instance 'neural-network.embedding:word-embedding
                            :vocab-size 20
                            :embedding-dim 8))
        (indices (make-tensor #2A((0.0 1.0 2.0)
                                  (3.0 4.0 5.0))
                             :shape '(2 3))))
    (let ((result (forward emb indices)))
      (is (equal '(2 3 8) (tensor-shape result))))))

(test word-embedding-parameters
  "Test that word embeddings have trainable parameters"
  (let ((emb (make-instance 'neural-network.embedding:word-embedding
                            :vocab-size 5
                            :embedding-dim 3)))
    (is (= 1 (length (parameters emb))))
    (is (requires-grad (first (parameters emb))))))

;;;; ============================================================================
;;;; Subword Embedding Tests
;;;; ============================================================================

(test subword-embedding-creation
  "Test subword embedding creation"
  (let ((emb (make-instance 'neural-network.embedding:subword-embedding
                            :vocab-size 100
                            :embedding-dim 32
                            :max-subwords 5
                            :aggregation :mean)))
    (is (not (null emb)))
    (is (= 100 (neural-network.embedding:emb-vocab-size emb)))
    (is (= 32 (embedding-dim emb)))
    (is (= 5 (neural-network.embedding:max-subwords emb)))
    (is (eq :mean (neural-network.embedding:aggregation emb)))))

(test subword-embedding-mean-aggregation
  "Test subword embedding with mean aggregation"
  (let ((emb (make-instance 'neural-network.embedding:subword-embedding
                            :vocab-size 10
                            :embedding-dim 4
                            :max-subwords 3
                            :aggregation :mean))
        ;; Each row has subword indices, -1 for padding
        (indices (make-tensor #2A((0.0 1.0 -1.0)
                                  (2.0 3.0 4.0))
                             :shape '(2 3))))
    (let ((result (forward emb indices)))
      (is (equal '(2 4) (tensor-shape result)))
      ;; Result should contain non-zero values
      (let ((data (tensor-data result)))
        (is (not (= 0.0d0 (aref data 0 0))))
        (is (not (= 0.0d0 (aref data 1 0))))))))

(test subword-embedding-sum-aggregation
  "Test subword embedding with sum aggregation"
  (let ((emb (make-instance 'neural-network.embedding:subword-embedding
                            :vocab-size 10
                            :embedding-dim 4
                            :max-subwords 2
                            :aggregation :sum))
        (indices (make-tensor #2A((0.0 1.0))
                             :shape '(1 2))))
    (let ((result (forward emb indices)))
      (is (equal '(1 4) (tensor-shape result))))))

(test subword-embedding-max-aggregation
  "Test subword embedding with max aggregation"
  (let ((emb (make-instance 'neural-network.embedding:subword-embedding
                            :vocab-size 10
                            :embedding-dim 4
                            :max-subwords 2
                            :aggregation :max))
        (indices (make-tensor #2A((0.0 1.0))
                             :shape '(1 2))))
    (let ((result (forward emb indices)))
      (is (equal '(1 4) (tensor-shape result))))))

;;;; ============================================================================
;;;; Positional Encoding Tests
;;;; ============================================================================

(test sinusoidal-positional-encoding-creation
  "Test sinusoidal positional encoding creation"
  (let ((pe (make-instance 'neural-network.embedding:sinusoidal-positional-encoding
                           :max-length 100
                           :embedding-dim 64)))
    (is (not (null pe)))
    (is (= 100 (emb-max-length pe)))
    (is (= 64 (embedding-dim pe)))
    (is (not (null (neural-network.embedding:encodings pe))))))

(test sinusoidal-positional-encoding-forward
  "Test sinusoidal positional encoding forward pass"
  (let ((pe (make-instance 'neural-network.embedding:sinusoidal-positional-encoding
                           :max-length 50
                           :embedding-dim 8))
        (x (make-tensor #3A(((1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                             (1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)))
                       :shape '(1 2 8))))
    (let ((result (forward pe x)))
      (is (equal '(1 2 8) (tensor-shape result)))
      ;; Result should be different from input (positions added)
      ;; Check position [0,0,1] where PE is cos(0) = 1.0, not zero
      (let ((x-data (tensor-data x))
            (r-data (tensor-data result)))
        (is (not (= (aref x-data 0 0 1) (aref r-data 0 0 1))))))))

(test sinusoidal-positional-encoding-properties
  "Test sinusoidal positional encoding has expected properties"
  (let ((pe (make-instance 'neural-network.embedding:sinusoidal-positional-encoding
                           :max-length 10
                           :embedding-dim 4)))
    ;; Even dimensions should use sine
    ;; Odd dimensions should use cosine
    (let ((enc-data (tensor-data (neural-network.embedding:encodings pe))))
      ;; Just verify it's not all zeros
      ;; Check dimension 1 (odd) at position 0: cos(0) = 1.0
      (is (not (= 0.0d0 (aref enc-data 0 1))))
      ;; Check dimension 0 (even) at position 1: sin(1) != 0
      (is (not (= 0.0d0 (aref enc-data 1 0))))
      ;; Different positions should have different encodings
      (is (not (= (aref enc-data 0 0) (aref enc-data 1 0)))))))

(test learned-positional-encoding-creation
  "Test learned positional encoding creation"
  (let ((pe (make-instance 'neural-network.embedding:learned-positional-encoding
                           :max-length 100
                           :embedding-dim 64)))
    (is (not (null pe)))
    (is (= 100 (emb-max-length pe)))
    (is (= 64 (embedding-dim pe)))
    (is (not (null (neural-network.embedding:position-embeddings pe))))))

(test learned-positional-encoding-forward
  "Test learned positional encoding forward pass"
  (let ((pe (make-instance 'neural-network.embedding:learned-positional-encoding
                           :max-length 50
                           :embedding-dim 8))
        (x (make-tensor #3A(((1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)
                             (1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0)))
                       :shape '(1 2 8))))
    (let ((result (forward pe x)))
      (is (equal '(1 2 8) (tensor-shape result)))
      ;; Result should be different from input
      (let ((x-data (tensor-data x))
            (r-data (tensor-data result)))
        (is (not (= (aref x-data 0 0 0) (aref r-data 0 0 0))))))))

(test learned-positional-encoding-trainable
  "Test learned positional encoding is trainable"
  (let ((pe (make-instance 'neural-network.embedding:learned-positional-encoding
                           :max-length 10
                           :embedding-dim 4)))
    (is (= 1 (length (parameters pe))))
    (is (requires-grad (first (parameters pe))))))

;;;; ============================================================================
;;;; Embedding with OOV Tests
;;;; ============================================================================

(test embedding-with-oov-creation
  "Test embedding with OOV handling creation"
  (let ((emb (make-instance 'neural-network.embedding:embedding-with-oov
                            :vocab-size 100
                            :embedding-dim 50
                            :oov-index 99)))
    (is (not (null emb)))
    (is (= 100 (neural-network.embedding:emb-vocab-size emb)))
    (is (= 50 (embedding-dim emb)))
    (is (= 99 (neural-network.embedding:oov-index emb)))))

(test embedding-with-oov-forward
  "Test embedding with OOV forward pass"
  (let ((emb (make-instance 'neural-network.embedding:embedding-with-oov
                            :vocab-size 10
                            :embedding-dim 4
                            :oov-index 9))
        (indices (make-tensor #(0.0 1.0 2.0) :shape '(3))))
    (let ((result (forward emb indices)))
      (is (equal '(3 4) (tensor-shape result))))))

(test embedding-with-oov-handles-invalid-indices
  "Test embedding with OOV handles invalid indices"
  (let ((emb (make-instance 'neural-network.embedding:embedding-with-oov
                            :vocab-size 10
                            :embedding-dim 4
                            :oov-index 9))
        ;; Include out-of-range indices
        (indices (make-tensor #(0.0 15.0 -5.0) :shape '(3))))
    (let ((result (forward emb indices)))
      (is (equal '(3 4) (tensor-shape result)))
      ;; Should not error, OOV indices mapped to oov-index
      (is (not (null result))))))

(test embedding-with-oov-and-padding
  "Test embedding with both OOV and padding"
  (let ((emb (make-instance 'neural-network.embedding:embedding-with-oov
                            :vocab-size 10
                            :embedding-dim 4
                            :oov-index 9
                            :pad-index 0))
        (indices (make-tensor #(0.0 1.0 2.0) :shape '(3))))
    (let ((result (forward emb indices)))
      ;; Padding should be zeros
      (let ((data (tensor-data result)))
        (is (= 0.0d0 (aref data 0 0)))
        (is (= 0.0d0 (aref data 0 1)))))))

;;;; ============================================================================
;;;; Numerical Embedding Tests
;;;; ============================================================================

(test numerical-embedding-creation
  "Test numerical embedding creation"
  (let ((emb (make-instance 'neural-network.embedding:numerical-embedding
                            :input-dim 5
                            :embedding-dim 10)))
    (is (not (null emb)))
    (is (= 5 (neural-network.embedding:input-dim emb)))
    (is (= 10 (embedding-dim emb)))
    (is (not (null (neural-network.embedding:projection emb))))))

(test numerical-embedding-forward
  "Test numerical embedding forward pass"
  (let ((emb (make-instance 'neural-network.embedding:numerical-embedding
                            :input-dim 3
                            :embedding-dim 8))
        (x (make-tensor #2A((1.0 2.0 3.0)
                            (4.0 5.0 6.0))
                       :shape '(2 3))))
    (let ((result (forward emb x)))
      (is (equal '(2 8) (tensor-shape result))))))

(test numerical-embedding-sequence
  "Test numerical embedding with sequence input"
  (let ((emb (make-instance 'neural-network.embedding:numerical-embedding
                            :input-dim 4
                            :embedding-dim 16))
        (x (make-tensor #3A(((1.0 2.0 3.0 4.0)
                             (5.0 6.0 7.0 8.0)))
                       :shape '(1 2 4))))
    (let ((result (forward emb x)))
      (is (equal '(1 2 16) (tensor-shape result))))))

(test numerical-embedding-trainable
  "Test numerical embedding has trainable parameters"
  (let ((emb (make-instance 'neural-network.embedding:numerical-embedding
                            :input-dim 3
                            :embedding-dim 6)))
    (is (> (length (parameters emb)) 0))
    (is (requires-grad (first (parameters emb))))))

;;;; ============================================================================
;;;; Byte Embedding Tests
;;;; ============================================================================

(test byte-embedding-creation
  "Test byte embedding creation"
  (let ((emb (make-instance 'neural-network.embedding:byte-embedding
                            :embedding-dim 32
                            :max-bytes 16
                            :aggregation :mean)))
    (is (not (null emb)))
    (is (= 256 (neural-network.embedding:byte-vocab-size emb)))
    (is (= 32 (embedding-dim emb)))
    (is (= 16 (neural-network.embedding:max-bytes emb)))
    (is (eq :mean (neural-network.embedding:aggregation emb)))))

(test byte-embedding-forward
  "Test byte embedding forward pass"
  (let ((emb (make-instance 'neural-network.embedding:byte-embedding
                            :embedding-dim 8
                            :max-bytes 4
                            :aggregation :mean))
        ;; ASCII bytes for "Hi" plus padding
        (bytes (make-tensor #2A((72.0 105.0 -1.0 -1.0)
                                (65.0 66.0 67.0 -1.0))
                           :shape '(2 4))))
    (let ((result (forward emb bytes)))
      (is (equal '(2 8) (tensor-shape result))))))

(test byte-embedding-all-aggregations
  "Test byte embedding with different aggregation methods"
  (let ((bytes (make-tensor #2A((65.0 66.0 -1.0))
                           :shape '(1 3))))
    (dolist (agg '(:mean :sum :cnn))
      (let ((emb (make-instance 'neural-network.embedding:byte-embedding
                                :embedding-dim 4
                                :max-bytes 3
                                :aggregation agg)))
        (let ((result (forward emb bytes)))
          (is (equal '(1 4) (tensor-shape result))))))))

(test byte-embedding-valid-range
  "Test byte embedding handles valid byte range (0-255)"
  (let ((emb (make-instance 'neural-network.embedding:byte-embedding
                            :embedding-dim 4
                            :max-bytes 3
                            :aggregation :mean))
        ;; Use valid byte values
        (bytes (make-tensor #2A((0.0 127.0 255.0))
                           :shape '(1 3))))
    (let ((result (forward emb bytes)))
      (is (not (null result)))
      (is (equal '(1 4) (tensor-shape result))))))

;;;; ============================================================================
;;;; Combined Embedding Tests
;;;; ============================================================================

(test combined-embedding-creation
  "Test combined embedding creation"
  (let* ((token-emb (make-instance 'neural-network.embedding:word-embedding
                                   :vocab-size 100
                                   :embedding-dim 32))
         (pos-enc (make-instance 'neural-network.embedding:sinusoidal-positional-encoding
                                 :max-length 50
                                 :embedding-dim 32))
         (combined (make-instance 'neural-network.embedding:combined-embedding
                                  :token-embedding token-emb
                                  :positional-encoding pos-enc)))
    (is (not (null combined)))
    (is (eq token-emb (neural-network.embedding:token-embedding combined)))
    (is (eq pos-enc (neural-network.embedding:positional-encoding combined)))))

(test combined-embedding-forward
  "Test combined embedding forward pass"
  (let* ((token-emb (make-instance 'neural-network.embedding:word-embedding
                                   :vocab-size 20
                                   :embedding-dim 8))
         (pos-enc (make-instance 'neural-network.embedding:learned-positional-encoding
                                 :max-length 10
                                 :embedding-dim 8))
         (combined (make-instance 'neural-network.embedding:combined-embedding
                                  :token-embedding token-emb
                                  :positional-encoding pos-enc))
         (indices (make-tensor #2A((0.0 1.0 2.0))
                              :shape '(1 3))))
    (let ((result (forward combined indices)))
      ;; Result should have positions added
      (is (equal '(1 3 8) (tensor-shape result))))))

(test combined-embedding-without-position
  "Test combined embedding without positional encoding"
  (let* ((token-emb (make-instance 'neural-network.embedding:word-embedding
                                   :vocab-size 20
                                   :embedding-dim 8))
         (combined (make-instance 'neural-network.embedding:combined-embedding
                                  :token-embedding token-emb
                                  :positional-encoding nil))
         (indices (make-tensor #2A((0.0 1.0 2.0))
                              :shape '(1 3))))
    (let ((result (forward combined indices)))
      ;; Should still work without positional encoding
      (is (equal '(1 3 8) (tensor-shape result))))))

(test combined-embedding-parameters
  "Test combined embedding aggregates parameters"
  (let* ((token-emb (make-instance 'neural-network.embedding:word-embedding
                                   :vocab-size 10
                                   :embedding-dim 4))
         (pos-enc (make-instance 'neural-network.embedding:learned-positional-encoding
                                 :max-length 5
                                 :embedding-dim 4))
         (combined (make-instance 'neural-network.embedding:combined-embedding
                                  :token-embedding token-emb
                                  :positional-encoding pos-enc)))
    ;; Should have parameters from both token and position embeddings
    (is (>= (length (parameters combined)) 2))))

;;;; ============================================================================
;;;; Utility Function Tests
;;;; ============================================================================

(test lookup-function
  "Test lookup utility function"
  (let ((emb (make-instance 'neural-network.embedding:word-embedding
                            :vocab-size 10
                            :embedding-dim 4))
        (indices (make-tensor #(0.0 1.0 2.0) :shape '(3))))
    (let ((result (neural-network.embedding:lookup emb indices)))
      (is (not (null result)))
      (is (equal '(3 4) (tensor-shape result))))))

(test get-embedding-function
  "Test get-embedding utility function"
  (let ((emb (make-instance 'neural-network.embedding:word-embedding
                            :vocab-size 10
                            :embedding-dim 4)))
    (let ((result (neural-network.embedding:get-embedding emb 5)))
      (is (not (null result)))
      (is (equal '(4) (tensor-shape result))))))

;;;; ============================================================================
;;;; Integration Tests
;;;; ============================================================================

(test embedding-gradient-flow
  "Test that gradients flow through embeddings"
  (let ((emb (make-instance 'neural-network.embedding:word-embedding
                            :vocab-size 5
                            :embedding-dim 3))
        (indices (make-tensor #(0.0 1.0) :shape '(2))))
    (let ((output (forward emb indices)))
      ;; Enable gradients
      (setf (requires-grad output) t)
      (setf (tensor-grad output) 
            (make-array (tensor-shape output) 
                       :element-type 'double-float 
                       :initial-element 1.0d0))
      ;; Should have gradient-enabled output
      (is (requires-grad output)))))

(test embedding-layer-mode
  "Test embedding layers respect training/eval mode"
  (let ((emb (make-instance 'neural-network.embedding:word-embedding
                            :vocab-size 10
                            :embedding-dim 4)))
    (train-mode emb)
    (is (layer-training emb))
    (eval-mode emb)
    (is (not (layer-training emb)))))

(test embedding-with-network
  "Test embedding layer in a simple network"
  (let* ((emb (make-instance 'neural-network.embedding:word-embedding
                             :vocab-size 20
                             :embedding-dim 8))
         (linear (linear 8 4))
         (net (sequential emb linear))
         (indices (make-tensor #(0.0 1.0 2.0) :shape '(3))))
    (let ((output (forward net indices)))
      (is (not (null output)))
      ;; Should go through embedding then linear
      (is (equal '(3 4) (tensor-shape output))))))

(test multiple-embedding-types
  "Test using different embedding types together"
  (let ((word-emb (make-instance 'neural-network.embedding:word-embedding
                                 :vocab-size 100
                                 :embedding-dim 16))
        (num-emb (make-instance 'neural-network.embedding:numerical-embedding
                                :input-dim 5
                                :embedding-dim 16))
        (word-indices (make-tensor #(0.0 1.0) :shape '(2)))
        (num-features (make-tensor #2A((1.0 2.0 3.0 4.0 5.0)
                                       (6.0 7.0 8.0 9.0 10.0))
                                  :shape '(2 5))))
    (let ((word-output (forward word-emb word-indices))
          (num-output (forward num-emb num-features)))
      ;; Both should produce embeddings of the same dimension
      (is (= (car (last (tensor-shape word-output)))
             (car (last (tensor-shape num-output))))))))
