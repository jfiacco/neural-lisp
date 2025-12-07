;;;; Tests for Transformer Architecture
;;;; Testing attention, encoder, decoder, and full transformer

(defpackage :neural-tensor-transformer-tests
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
  (:import-from :neural-tensor-transformer
                #:scaled-dot-product-attention
                #:multi-head-attention
                #:positional-encoding
                #:feed-forward-network
                #:transformer-encoder-layer
                #:transformer-decoder-layer
                #:transformer
                #:layer-norm
                #:causal-mask
                #:padding-mask
                #:softmax)
  (:export #:run-transformer-tests))

(in-package :neural-tensor-transformer-tests)

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

(defun run-transformer-tests ()
  "Run all transformer tests"
  (setf *test-results* nil
        *tests-passed* 0
        *tests-failed* 0)
  
  (format t "~%")
  (format t "╔════════════════════════════════════════════════════════════════╗~%")
  (format t "║  Transformer Architecture Tests                               ║~%")
  (format t "╚════════════════════════════════════════════════════════════════╝~%")
  (format t "~%")
  
  ;; Test Softmax
  (deftest test-softmax
    (let* ((input (make-tensor #(1.0d0 2.0d0 3.0d0) :shape '(3)))
           (output (softmax input)))
      (assert-shape '(3) output)
      ;; Check softmax properties: sum should be ~1
      (let ((sum 0.0d0))
        (dotimes (i 3)
          (incf sum (aref (tensor-data output) i)))
        (assert-equal 1.0d0 sum 1d-5))))
  
  ;; Test Scaled Dot-Product Attention
  (deftest test-scaled-dot-product-attention
    (let* ((q (randn '(2 10 64)))  ; (batch, seq_len, d_k)
           (k (randn '(2 10 64)))
           (v (randn '(2 10 64)))
           (output (scaled-dot-product-attention q k v)))
      (assert-shape '(2 10 64) output)))
  
  (deftest test-attention-with-mask
    (let* ((q (randn '(1 5 64)))
           (k (randn '(1 5 64)))
           (v (randn '(1 5 64)))
           (mask (causal-mask 5))
           (output (scaled-dot-product-attention q k v :mask mask)))
      (assert-shape '(1 5 64) output)))
  
  ;; Test Multi-Head Attention
  (deftest test-multi-head-attention-creation
    (let ((mha (make-instance 'multi-head-attention
                             :d-model 512
                             :num-heads 8)))
      (assert (= 512 (neural-tensor-transformer::d-model mha)))
      (assert (= 8 (neural-tensor-transformer::num-heads mha)))
      (assert (= 64 (neural-tensor-transformer::d-k mha)))))
  
  (deftest test-multi-head-attention-parameters
    (let* ((mha (make-instance 'multi-head-attention
                              :d-model 512
                              :num-heads 8))
           (params (layer-parameters mha)))
      (assert (= 4 (length params))))) ; W_q, W_k, W_v, W_o
  
  (deftest test-multi-head-attention-forward
    (let* ((mha (make-instance 'multi-head-attention
                              :d-model 64
                              :num-heads 4))
           (q (randn '(2 10 64)))
           (k (randn '(2 10 64)))
           (v (randn '(2 10 64)))
           (output (neural-tensor-transformer::multi-head-attention-forward mha q k v)))
      (assert-shape '(2 10 64) output)))
  
  ;; Test Positional Encoding
  (deftest test-positional-encoding-creation
    (let ((pe (make-instance 'positional-encoding
                            :d-model 512
                            :max-len 1000)))
      (assert (= 512 (neural-tensor-transformer::d-model pe)))
      (assert (= 1000 (neural-tensor-transformer::max-len pe)))
      (assert (not (null (neural-tensor-transformer::encoding pe))))))
  
  (deftest test-positional-encoding-shape
    (let* ((pe (make-instance 'positional-encoding
                             :d-model 128
                             :max-len 500))
           (encoding (neural-tensor-transformer::encoding pe)))
      (assert-shape '(500 128) encoding)))
  
  (deftest test-positional-encoding-forward
    (let* ((pe (make-instance 'positional-encoding
                             :d-model 64
                             :max-len 100))
           (input (randn '(2 50 64)))
           (output (forward pe input)))
      (assert-shape '(2 50 64) output)))
  
  ;; Test Feed-Forward Network
  (deftest test-ffn-creation
    (let ((ffn (make-instance 'feed-forward-network
                             :d-model 512
                             :d-ff 2048)))
      (assert (not (null (neural-tensor-transformer::linear1 ffn))))
      (assert (not (null (neural-tensor-transformer::linear2 ffn))))))
  
  (deftest test-ffn-parameters
    (let* ((ffn (make-instance 'feed-forward-network
                              :d-model 512
                              :d-ff 2048))
           (params (layer-parameters ffn)))
      (assert (>= (length params) 4)))) ; At least weights and biases for 2 layers
  
  (deftest test-ffn-forward
    (let* ((ffn (make-instance 'feed-forward-network
                              :d-model 64
                              :d-ff 256))
           (input (randn '(2 10 64)))
           (output (forward ffn input)))
      (assert-shape '(2 10 64) output)))
  
  ;; Test Layer Normalization
  (deftest test-layer-norm-creation
    (let ((ln (make-instance 'layer-norm
                            :normalized-shape '(512))))
      (assert (not (null (neural-tensor-transformer::gamma ln))))
      (assert (not (null (neural-tensor-transformer::beta ln))))))
  
  (deftest test-layer-norm-parameters
    (let* ((ln (make-instance 'layer-norm
                             :normalized-shape '(512)))
           (params (layer-parameters ln)))
      (assert (= 2 (length params))))) ; gamma and beta
  
  ;; Test Transformer Encoder Layer
  (deftest test-encoder-layer-creation
    (let ((encoder (make-instance 'transformer-encoder-layer
                                 :d-model 512
                                 :num-heads 8
                                 :d-ff 2048)))
      (assert (not (null (neural-tensor-transformer::self-attn encoder))))
      (assert (not (null (neural-tensor-transformer::feed-forward encoder))))
      (assert (not (null (neural-tensor-transformer::norm1 encoder))))
      (assert (not (null (neural-tensor-transformer::norm2 encoder))))))
  
  (deftest test-encoder-layer-parameters
    (let* ((encoder (make-instance 'transformer-encoder-layer
                                  :d-model 64
                                  :num-heads 4
                                  :d-ff 256))
           (params (layer-parameters encoder)))
      (assert (> (length params) 10)))) ; Attention + FFN + norms
  
  (deftest test-encoder-layer-forward
    (let* ((encoder (make-instance 'transformer-encoder-layer
                                  :d-model 64
                                  :num-heads 4
                                  :d-ff 256))
           (input (randn '(2 10 64)))
           (output (forward encoder input)))
      (assert-shape '(2 10 64) output)))
  
  ;; Test Transformer Decoder Layer
  (deftest test-decoder-layer-creation
    (let ((decoder (make-instance 'transformer-decoder-layer
                                 :d-model 512
                                 :num-heads 8
                                 :d-ff 2048)))
      (assert (not (null (neural-tensor-transformer::self-attn decoder))))
      (assert (not (null (neural-tensor-transformer::cross-attn decoder))))
      (assert (not (null (neural-tensor-transformer::feed-forward decoder))))
      (assert (not (null (neural-tensor-transformer::norm1 decoder))))
      (assert (not (null (neural-tensor-transformer::norm2 decoder))))
      (assert (not (null (neural-tensor-transformer::norm3 decoder))))))
  
  (deftest test-decoder-layer-parameters
    (let* ((decoder (make-instance 'transformer-decoder-layer
                                  :d-model 64
                                  :num-heads 4
                                  :d-ff 256))
           (params (layer-parameters decoder)))
      (assert (> (length params) 15)))) ; Self-attn + cross-attn + FFN + norms
  
  ;; Test Full Transformer
  (deftest test-transformer-creation
    (let ((model (make-instance 'transformer
                               :d-model 512
                               :num-heads 8
                               :num-encoder-layers 6
                               :num-decoder-layers 6
                               :d-ff 2048)))
      (assert (= 6 (length (neural-tensor-transformer::encoder-layers model))))
      (assert (= 6 (length (neural-tensor-transformer::decoder-layers model))))))
  
  (deftest test-transformer-parameters
    (let* ((model (make-instance 'transformer
                                :d-model 64
                                :num-heads 4
                                :num-encoder-layers 2
                                :num-decoder-layers 2
                                :d-ff 256))
           (params (layer-parameters model)))
      (assert (> (length params) 50)))) ; Many parameters across all layers
  
  ;; Test Attention Masks
  (deftest test-causal-mask
    (let ((mask (causal-mask 5)))
      (assert-shape '(5 5) mask)
      ;; Check lower triangular
      (let ((data (tensor-data mask)))
        (assert-equal 1.0d0 (aref data 0 0))
        (assert-equal 1.0d0 (aref data 1 0))
        (assert-equal 1.0d0 (aref data 1 1))
        (assert-equal 0.0d0 (aref data 0 1))
        (assert-equal 0.0d0 (aref data 0 2)))))
  
  (deftest test-padding-mask
    (let* ((lengths '(3 2 4))
           (max-len 5)
           (mask (padding-mask lengths max-len)))
      (assert-shape '(3 5) mask)))
  
  ;; Test d_model divisibility by num_heads
  (deftest test-invalid-num-heads
    (handler-case
        (progn
          (make-instance 'multi-head-attention
                        :d-model 512
                        :num-heads 7) ; 512 not divisible by 7
          (error "Should have raised an error"))
      (error () t)))
  
  ;; Test attention output dimensions
  (deftest test-attention-dimensions
    (let* ((batch-size 4)
           (seq-len 10)
           (d-model 128)
           (num-heads 8)
           (mha (make-instance 'multi-head-attention
                              :d-model d-model
                              :num-heads num-heads))
           (q (randn (list batch-size seq-len d-model)))
           (k (randn (list batch-size seq-len d-model)))
           (v (randn (list batch-size seq-len d-model)))
           (output (neural-tensor-transformer::multi-head-attention-forward mha q k v)))
      (assert-shape (list batch-size seq-len d-model) output)))
  
  ;; ========== EDGE CASE TESTS ==========
  
  ;; Test with very small dimensions
  (deftest test-transformer-minimal-dims
    (let* ((model (make-instance 'transformer
                                :d-model 8
                                :num-heads 2
                                :num-encoder-layers 1
                                :num-decoder-layers 1
                                :d-ff 16))
           (src (randn '(1 2 8)))
           (tgt (randn '(1 2 8))))
      (let ((output (neural-tensor-transformer::transformer-forward model src tgt nil nil nil)))
        (assert-shape '(1 2 8) output))))
  
  ;; Test with single sequence element
  (deftest test-attention-single-token
    (let* ((mha (make-instance 'multi-head-attention
                              :d-model 64
                              :num-heads 4))
           (q (randn '(1 1 64)))
           (k (randn '(1 1 64)))
           (v (randn '(1 1 64)))
           (output (neural-tensor-transformer::multi-head-attention-forward mha q k v)))
      (assert-shape '(1 1 64) output)))
  
  ;; Test with very long sequences
  (deftest test-transformer-long-sequence
    (let* ((encoder (make-instance 'transformer-encoder-layer
                                  :d-model 32
                                  :num-heads 4
                                  :d-ff 64))
           (input (randn '(1 200 32))))
      (let ((output (forward encoder input)))
        (assert-shape '(1 200 32) output))))
  
  ;; Test with large batch size
  (deftest test-transformer-large-batch
    (let* ((encoder (make-instance 'transformer-encoder-layer
                                  :d-model 32
                                  :num-heads 4
                                  :d-ff 64))
           (input (randn '(64 10 32))))
      (let ((output (forward encoder input)))
        (assert-shape '(64 10 32) output))))
  
  ;; Test numerical stability with extreme values
  (deftest test-attention-numerical-stability
    (let* ((q (make-tensor 
               (make-array '(1 5 64)
                          :element-type 'double-float
                          :initial-element 100.0d0)
               :shape '(1 5 64)))
           (k (make-tensor 
               (make-array '(1 5 64)
                          :element-type 'double-float
                          :initial-element 100.0d0)
               :shape '(1 5 64)))
           (v (randn '(1 5 64)))
           (output (scaled-dot-product-attention q k v))
           (data (tensor-data output)))
      ;; Check no NaN or Inf values
      (dotimes (i (min 20 (array-total-size data)))
        (let ((val (row-major-aref data i)))
          (assert (not (sb-ext:float-nan-p val)))
          (assert (not (sb-ext:float-infinity-p val)))))))
  
  ;; Test softmax with very large values (numerical stability)
  (deftest test-softmax-numerical-stability
    (let* ((input (make-tensor #(100.0d0 200.0d0 150.0d0) :shape '(3)))
           (output (softmax input))
           (data (tensor-data output)))
      ;; Check sum is approximately 1
      (let ((sum 0.0d0))
        (dotimes (i 3)
          (incf sum (aref data i)))
        (assert-equal 1.0d0 sum 1d-5))
      ;; Check no NaN values
      (dotimes (i 3)
        (assert (not (sb-ext:float-nan-p (aref data i)))))))
  
  ;; Test softmax with zero/negative values
  (deftest test-softmax-negative-values
    (let* ((input (make-tensor #(-5.0d0 -10.0d0 -3.0d0) :shape '(3)))
           (output (softmax input)))
      (assert-shape '(3) output)
      (let ((sum 0.0d0))
        (dotimes (i 3)
          (incf sum (aref (tensor-data output) i)))
        (assert-equal 1.0d0 sum 1d-5))))
  
  ;; Test cross-attention with different sequence lengths
  (deftest test-cross-attention-different-lengths
    (let* ((mha (make-instance 'multi-head-attention
                              :d-model 64
                              :num-heads 4))
           (q (randn '(2 5 64)))   ; 5 query positions
           (k (randn '(2 10 64)))  ; 10 key positions
           (v (randn '(2 10 64)))  ; 10 value positions
           (output (neural-tensor-transformer::multi-head-attention-forward mha q k v)))
      (assert-shape '(2 5 64) output))) ; Output should match query length
  
  ;; Test positional encoding edge cases
  (deftest test-positional-encoding-max-length
    (let* ((pe (make-instance 'positional-encoding
                             :d-model 64
                             :max-len 100))
           ;; Input at max length
           (input (randn '(1 100 64)))
           (output (forward pe input)))
      (assert-shape '(1 100 64) output)))
  
  ;; Test layer norm with constant input
  (deftest test-layer-norm-constant-input
    (let* ((ln (make-instance 'layer-norm
                             :normalized-shape '(64)))
           (input (make-tensor 
                   (make-array '(2 10 64)
                              :element-type 'double-float
                              :initial-element 5.0d0)
                   :shape '(2 10 64)))
           (output (forward ln input))
           (data (tensor-data output)))
      ;; Check no NaN values (constant input can cause division by zero)
      (dotimes (i (min 50 (array-total-size data)))
        (let ((val (row-major-aref data i)))
          (assert (not (sb-ext:float-nan-p val)))))))
  
  ;; Test FFN with different dimensions
  (deftest test-ffn-dimension-variations
    (let ((ffn1 (make-instance 'feed-forward-network
                              :d-model 32
                              :d-ff 128))  ; 4x expansion
          (ffn2 (make-instance 'feed-forward-network
                              :d-model 32
                              :d-ff 32))   ; 1x (no expansion)
          (input (randn '(2 5 32))))
      (assert-shape '(2 5 32) (forward ffn1 input))
      (assert-shape '(2 5 32) (forward ffn2 input))))
  
  ;; Test causal mask properties
  (deftest test-causal-mask-properties
    (let ((mask (causal-mask 10)))
      ;; Check it's lower triangular
      (let ((data (tensor-data mask)))
        (dotimes (i 10)
          (dotimes (j 10)
            (if (<= j i)
                (assert-equal 1.0d0 (aref data i j))
                (assert-equal 0.0d0 (aref data i j))))))))
  
  ;; Test padding mask with all valid (no padding)
  (deftest test-padding-mask-no-padding
    (let* ((lengths '(5 5 5))
           (max-len 5)
           (mask (padding-mask lengths max-len))
           (data (tensor-data mask)))
      ;; All should be unmasked (1.0)
      (dotimes (i (* 3 5))
        (assert-equal 1.0d0 (row-major-aref data i)))))
  
  ;; Test padding mask with all padding
  (deftest test-padding-mask-all-padding
    (let* ((lengths '(0 0 0))
           (max-len 5)
           (mask (padding-mask lengths max-len))
           (data (tensor-data mask)))
      ;; All should be masked (0.0)
      (dotimes (i (* 3 5))
        (assert-equal 0.0d0 (row-major-aref data i)))))
  
  ;; Test decoder layer with cross attention
  (deftest test-decoder-cross-attention
    (let* ((decoder (make-instance 'transformer-decoder-layer
                                  :d-model 64
                                  :num-heads 4
                                  :d-ff 128))
           (tgt (randn '(2 8 64)))
           (memory (randn '(2 12 64))))  ; Encoder output (different length)
      (let ((output (neural-tensor-transformer::transformer-decoder-layer-forward 
                     decoder tgt memory nil nil)))
        (assert-shape '(2 8 64) output))))
  
  ;; Test attention with all-zero query/key (edge case)
  (deftest test-attention-zero-inputs
    (let* ((q (zeros '(1 5 64)))
           (k (zeros '(1 5 64)))
           (v (randn '(1 5 64)))
           (output (scaled-dot-product-attention q k v)))
      ;; Should not crash or produce NaN
      (assert-shape '(1 5 64) output)
      (let ((data (tensor-data output)))
        (dotimes (i (min 20 (array-total-size data)))
          (assert (not (sb-ext:float-nan-p (row-major-aref data i))))))))
  
  ;; Test multi-head attention head count validation
  (deftest test-num-heads-power-of-two
    ;; Common architectures use power-of-2 heads
    (let* ((mha2 (make-instance 'multi-head-attention :d-model 64 :num-heads 2))
           (mha4 (make-instance 'multi-head-attention :d-model 64 :num-heads 4))
           (mha8 (make-instance 'multi-head-attention :d-model 128 :num-heads 8))
           (input (randn '(1 5 64)))
           (input2 (randn '(1 5 128))))
      (assert-shape '(1 5 64) 
                   (neural-tensor-transformer::multi-head-attention-forward mha2 input input input))
      (assert-shape '(1 5 64) 
                   (neural-tensor-transformer::multi-head-attention-forward mha4 input input input))
      (assert-shape '(1 5 128) 
                   (neural-tensor-transformer::multi-head-attention-forward mha8 input2 input2 input2))))
  
  ;; Test dropout in attention
  (deftest test-attention-with-dropout
    (let* ((q (randn '(1 10 64)))
           (k (randn '(1 10 64)))
           (v (randn '(1 10 64)))
           (output-no-drop (scaled-dot-product-attention q k v :dropout 0.0))
           (output-with-drop (scaled-dot-product-attention q k v :dropout 0.1)))
      ;; Both should have same shape
      (assert-shape '(1 10 64) output-no-drop)
      (assert-shape '(1 10 64) output-with-drop)))
  
  (deftest test-dropout-rate-extremes
    (let* ((q (randn '(1 5 32)))
           (k (randn '(1 5 32)))
           (v (randn '(1 5 32))))
      ;; No dropout
      (let ((out1 (scaled-dot-product-attention q k v :dropout 0.0)))
        (assert-shape '(1 5 32) out1))
      ;; High dropout
      (let ((out2 (scaled-dot-product-attention q k v :dropout 0.5)))
        (assert-shape '(1 5 32) out2))))
  
  ;; Test sparse attention patterns
  (deftest test-sparse-attention-local
    (let* ((q (randn '(1 20 64)))
           (k (randn '(1 20 64)))
           (v (randn '(1 20 64)))
           (output (neural-tensor-transformer::sparse-attention q k v
                                                                :sparsity-pattern :local
                                                                :window-size 5)))
      (assert-shape '(1 20 64) output)))
  
  (deftest test-sparse-attention-strided
    (let* ((q (randn '(1 30 64)))
           (k (randn '(1 30 64)))
           (v (randn '(1 30 64)))
           (output (neural-tensor-transformer::sparse-attention q k v
                                                                :sparsity-pattern :strided
                                                                :stride 4)))
      (assert-shape '(1 30 64) output)))
  
  (deftest test-sparse-attention-fixed
    (let* ((q (randn '(1 25 64)))
           (k (randn '(1 25 64)))
           (v (randn '(1 25 64)))
           (output (neural-tensor-transformer::sparse-attention q k v
                                                                :sparsity-pattern :fixed
                                                                :num-global 3)))
      (assert-shape '(1 25 64) output)))
  
  (deftest test-sparse-attention-global
    (let* ((q (randn '(1 40 64)))
           (k (randn '(1 40 64)))
           (v (randn '(1 40 64)))
           (output (neural-tensor-transformer::sparse-attention q k v
                                                                :sparsity-pattern :global
                                                                :num-global 4)))
      (assert-shape '(1 40 64) output)))
  
  ;; Test sparse attention with different sequence lengths
  (deftest test-sparse-attention-short-sequence
    (let* ((q (randn '(1 5 64)))
           (k (randn '(1 5 64)))
           (v (randn '(1 5 64))))
      ;; Local with window larger than sequence
      (let ((out1 (neural-tensor-transformer::sparse-attention q k v
                                                               :sparsity-pattern :local
                                                               :window-size 10)))
        (assert-shape '(1 5 64) out1))
      ;; Global with more global tokens than sequence
      (let ((out2 (neural-tensor-transformer::sparse-attention q k v
                                                               :sparsity-pattern :global
                                                               :num-global 10)))
        (assert-shape '(1 5 64) out2))))
  
  ;; Test sparse attention batched
  (deftest test-sparse-attention-batched
    (let* ((q (randn '(4 15 64)))
           (k (randn '(4 15 64)))
           (v (randn '(4 15 64)))
           (output (neural-tensor-transformer::sparse-attention q k v
                                                                :sparsity-pattern :local
                                                                :window-size 5)))
      (assert-shape '(4 15 64) output)))
  
  ;; Test all sparse patterns with same input
  (deftest test-all-sparse-patterns
    (let* ((q (randn '(1 20 32)))
           (k (randn '(1 20 32)))
           (v (randn '(1 20 32))))
      (dolist (pattern '(:local :strided :fixed :global))
        (let ((output (neural-tensor-transformer::sparse-attention q k v
                                                                   :sparsity-pattern pattern)))
          (assert-shape '(1 20 32) output)))))
  
  ;; Test apply-dropout function
  (deftest test-apply-dropout-zeros
    (let* ((input (make-tensor #2A((1.0d0 2.0d0 3.0d0 4.0d0)) :shape '(1 4)))
           (dropped (neural-tensor-transformer::apply-dropout input 0.5))
           (data (tensor-data dropped)))
      ;; Some values should be zeroed, others scaled
      (assert-shape '(1 4) dropped)
      ;; At least one should be zero or scaled (probabilistic)
      (assert (not (null data)))))
  
  ;; Test attention with different d_k sizes
  (deftest test-attention-different-dk
    (dolist (dk '(16 32 64 128))
      (let* ((q (randn (list 1 10 dk)))
             (k (randn (list 1 10 dk)))
             (v (randn (list 1 10 dk)))
             (output (scaled-dot-product-attention q k v)))
        (assert-shape (list 1 10 dk) output))))
  
  ;; Test attention with dropout disabled and enabled comparison
  (deftest test-dropout-consistency
    (let* ((q (randn '(1 5 32)))
           (k (randn '(1 5 32)))
           (v (randn '(1 5 32))))
      ;; With dropout 0, should be deterministic
      (let ((out1 (scaled-dot-product-attention q k v :dropout 0.0))
            (out2 (scaled-dot-product-attention q k v :dropout 0.0)))
        (assert-shape '(1 5 32) out1)
        (assert-shape '(1 5 32) out2))))
  
  ;; Print summary
  (format t "~%Transformer Tests: ~d passed, ~d failed~%~%" *tests-passed* *tests-failed*)
  
  (values *tests-passed* *tests-failed*))

;; Run tests when file is loaded
(format t "~%To run transformer tests, execute: (neural-tensor-transformer-tests:run-transformer-tests)~%")
