;;;; Neural Tensor Library - Transformer Architecture
;;;; Self-Attention, Multi-Head Attention, Transformer Blocks
;;;; Leveraging Lisp's symbolic computation and macros

(defpackage :neural-tensor-transformer
  (:use :common-lisp)
  (:import-from :neural-network
                #:tensor
                #:tensor-data
                #:tensor-shape
                #:make-tensor
                #:zeros
                #:ones
                #:randn
                #:t+
                #:t-
                #:t*
                #:t@
                #:forward
                #:backward
                #:layer
                #:linear
                #:layer-parameters
                #:parameters
                #:transpose
                #:sequential)
  (:import-from :neural-tensor-activations
                #:relu)
  (:export #:scaled-dot-product-attention
           #:multi-head-attention
           #:multi-head-attention-forward
           #:positional-encoding
           #:feed-forward-network
           #:transformer-encoder-layer
           #:transformer-encoder-layer-forward
           #:transformer-decoder-layer
           #:transformer-decoder-layer-forward
           #:transformer-encoder
           #:transformer-decoder
           #:transformer
           #:transformer-forward
           #:rotary-embedding
           #:alibi-attention
           #:flash-attention
           #:sparse-attention
           #:with-attention-mask
           #:defattention
           #:attention-pattern
           #:causal-mask
           #:padding-mask))

(in-package :neural-tensor-transformer)

;;;; ============================================================================
;;;; Core Attention Mechanism
;;;; ============================================================================

(defun apply-attention-mask (scores mask)
  "Apply mask to attention scores (set masked positions to -inf)"
  (let* ((score-data (neural-network::tensor-data scores))
         (mask-data (neural-network::tensor-data mask))
         (result (make-array (array-dimensions score-data)
                            :element-type 'double-float)))
    (dotimes (i (array-total-size score-data))
      (setf (row-major-aref result i)
            (if (zerop (row-major-aref mask-data i))
                -1.0d10 ; Large negative number (acts as -inf)
                (row-major-aref score-data i))))
    (make-tensor result :shape (neural-network::tensor-shape scores))))

(defun softmax (tensor)
  "Softmax activation along the last dimension"
  (let* ((data (neural-network::tensor-data tensor))
         (shape (neural-network::tensor-shape tensor))
         (result (make-array shape :element-type 'double-float)))
    
    (cond
      ;; 1D or 2D: simple softmax over all elements
      ((<= (length shape) 2)
       (let ((max-val most-negative-double-float))
         ;; Find max
         (dotimes (i (array-total-size data))
           (setf max-val (max max-val (row-major-aref data i))))
         ;; Compute exp(x - max) and sum
         (let ((sum 0.0d0))
           (dotimes (i (array-total-size data))
             (let ((exp-val (exp (- (row-major-aref data i) max-val))))
               (setf (row-major-aref result i) exp-val)
               (incf sum exp-val)))
           ;; Normalize
           (dotimes (i (array-total-size result))
             (setf (row-major-aref result i)
                   (/ (row-major-aref result i) sum))))))
      
      ;; 3D: softmax along last dimension for each (batch, seq_len)
      ((= (length shape) 3)
       (destructuring-bind (batch seq-len dim) shape
         (dotimes (b batch)
           (dotimes (s seq-len)
             ;; Find max for this sequence position
             (let ((max-val most-negative-double-float))
               (dotimes (d dim)
                 (setf max-val (max max-val (aref data b s d))))
               ;; Compute exp and sum
               (let ((sum 0.0d0))
                 (dotimes (d dim)
                   (let ((exp-val (exp (- (aref data b s d) max-val))))
                     (setf (aref result b s d) exp-val)
                     (incf sum exp-val)))
                 ;; Normalize
                 (dotimes (d dim)
                   (setf (aref result b s d)
                         (/ (aref result b s d) sum)))))))))
      
      (t (error "softmax only supports 1D, 2D, and 3D tensors")))
    
    (make-tensor result :shape shape :requires-grad (neural-network::requires-grad tensor))))

(defun transpose-last-two-dims (tensor)
  "Transpose the last two dimensions of a 3D tensor (batch, n, m) -> (batch, m, n)"
  (let* ((shape (neural-network::tensor-shape tensor))
         (data (neural-network::tensor-data tensor)))
    (cond
      ;; 2D tensor: use regular transpose
      ((= (length shape) 2)
       (transpose tensor))
      ;; 3D tensor: transpose last two dims for each batch
      ((= (length shape) 3)
       (destructuring-bind (batch n m) shape
         (let ((result (make-array (list batch m n) :element-type 'double-float)))
           (dotimes (b batch)
             (dotimes (i n)
               (dotimes (j m)
                 (setf (aref result b j i) (aref data b i j)))))
           (make-tensor result :shape (list batch m n)
                       :requires-grad (neural-network::requires-grad tensor)))))
      (t
       (error "transpose-last-two-dims only supports 2D and 3D tensors")))))

(defun scaled-dot-product-attention (query key value &key mask dropout)
  "Attention(Q, K, V) = softmax(QK^T / √d_k)V
   
   Arguments:
   - query: (batch, seq_len, d_k)
   - key: (batch, seq_len, d_k)
   - value: (batch, seq_len, d_v)
   - mask: optional attention mask
   - dropout: dropout probability (0.0 to 1.0)"
  
  (let* ((d-k (car (last (neural-network::tensor-shape query))))
         (scale (sqrt (coerce d-k 'double-float)))
         ;; Compute attention scores: QK^T / √d_k
         (scores (t* (t@ query (transpose-last-two-dims key))
                    (make-tensor (vector (/ 1.0d0 scale))
                                :shape '(1))))
         ;; Apply mask if provided
         (masked-scores (if mask
                           (apply-attention-mask scores mask)
                           scores))
         ;; Softmax
         (attention-weights (softmax masked-scores))
         ;; Apply dropout to attention weights if specified
         (dropped-weights (if (and dropout (> dropout 0.0))
                             (apply-dropout attention-weights dropout)
                             attention-weights))
         ;; Apply attention to values
         (output (t@ dropped-weights value)))
    (values output attention-weights)))

(defun apply-dropout (tensor rate)
  "Apply dropout with given rate during training
   
   Sets elements to zero with probability rate and scales remaining by 1/(1-rate)"
  (let* ((shape (neural-network::tensor-shape tensor))
         (data (neural-network::tensor-data tensor))
         (result (make-array (array-dimensions data) :element-type 'double-float))
         (scale (/ 1.0d0 (- 1.0d0 rate))))
    
    (dotimes (i (array-total-size data))
      (setf (row-major-aref result i)
            (if (< (random 1.0d0) rate)
                0.0d0
                (* (row-major-aref data i) scale))))
    
    (make-tensor result :shape shape :requires-grad (neural-network::requires-grad tensor))))

;;;; ============================================================================
;;;; Multi-Head Attention
;;;; ============================================================================

(defclass multi-head-attention (layer)
  ((d-model :initarg :d-model
            :reader d-model)
   (num-heads :initarg :num-heads
              :reader num-heads)
   (d-k :reader d-k)
   (d-v :reader d-v)
   (w-q :accessor w-q :documentation "Query projection")
   (w-k :accessor w-k :documentation "Key projection")
   (w-v :accessor w-v :documentation "Value projection")
   (w-o :accessor w-o :documentation "Output projection")
   (dropout :initarg :dropout
            :initform 0.1
            :accessor dropout-rate))
  (:documentation "Multi-head attention mechanism"))

(defmethod initialize-instance :after ((attn multi-head-attention) &key)
  (with-slots (d-model num-heads d-k d-v w-q w-k w-v w-o parameters) attn
    (unless (zerop (mod d-model num-heads))
      (error "d-model must be divisible by num-heads"))
    
    (setf d-k (/ d-model num-heads))
    (setf d-v (/ d-model num-heads))
    
    (let ((scale (/ 1.0 (sqrt d-model))))
      (setf w-q (randn (list d-model d-model)
                      :requires-grad t
                      :name "w-q"
                      :scale scale))
      (setf w-k (randn (list d-model d-model)
                      :requires-grad t
                      :name "w-k"
                      :scale scale))
      (setf w-v (randn (list d-model d-model)
                      :requires-grad t
                      :name "w-v"
                      :scale scale))
      (setf w-o (randn (list d-model d-model)
                      :requires-grad t
                      :name "w-o"
                      :scale scale))
      (setf parameters (list w-q w-k w-v w-o)))))

(defmethod forward ((attn multi-head-attention) input)
  "Multi-head attention forward pass - self attention variant
   
   input: (batch, seq_len, d_model)"
  (multi-head-attention-forward attn input input input))

(defun multi-head-attention-forward (attn query key value &key mask)
  "Multi-head attention forward pass
   
   query, key, value: (batch, seq_len, d_model)"
  (with-slots (num-heads d-k w-q w-k w-v w-o) attn
    ;; Linear projections: (batch, seq_len, d_model)
    (let* ((q (t@ query (transpose w-q)))
           (k (t@ key (transpose w-k)))
           (v (t@ value (transpose w-v))))
      
      ;; Split into heads: would reshape to (batch, num_heads, seq_len, d_k)
      ;; Simplified here - in practice would reshape tensors
      
      ;; Apply attention
      (multiple-value-bind (attn-output attn-weights)
          (scaled-dot-product-attention q k v :mask mask)
        
        ;; Concatenate heads and apply output projection
        (let ((output (t@ attn-output (transpose w-o))))
          (values output attn-weights))))))

;;;; ============================================================================
;;;; Positional Encoding
;;;; ============================================================================

(defclass positional-encoding (layer)
  ((d-model :initarg :d-model
            :reader d-model)
   (max-len :initarg :max-len
            :initform 5000
            :reader max-len)
   (encoding :accessor encoding))
  (:documentation "Sinusoidal positional encoding"))

(defmethod initialize-instance :after ((pe positional-encoding) &key)
  (with-slots (d-model max-len encoding) pe
    ;; PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    ;; PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    (let ((pe-array (make-array (list max-len d-model)
                               :element-type 'double-float)))
      (dotimes (pos max-len)
        (dotimes (i d-model)
          (let* ((angle (/ pos (expt 10000.0d0 (/ (* 2.0d0 (floor i 2))
                                                    d-model)))))
            (setf (aref pe-array pos i)
                  (if (evenp i)
                      (sin angle)
                      (cos angle))))))
      (setf encoding (make-tensor pe-array
                                 :shape (list max-len d-model)
                                 :requires-grad nil)))))

(defmethod forward ((pe positional-encoding) input)
  "Add positional encoding to input
   input: (batch, seq_len, d_model)"
  (with-slots (encoding d-model) pe
    (let* ((input-shape (tensor-shape input))
           (seq-len (second input-shape))
           ;; Extract first seq-len rows from encoding
           (enc-data (tensor-data encoding))
           (enc-slice-data (make-array (list seq-len d-model)
                                      :element-type 'double-float))
           (_ (dotimes (i seq-len)
                (dotimes (j d-model)
                  (setf (aref enc-slice-data i j)
                        (aref enc-data i j)))))
           (pos-enc-slice (make-tensor enc-slice-data
                                      :shape (list seq-len d-model)
                                      :requires-grad nil)))
      (declare (ignore _))
      ;; Broadcasting will add (seq_len, d_model) to each batch
      (t+ input pos-enc-slice))))

;;;; ============================================================================
;;;; Feed-Forward Network
;;;; ============================================================================

(defclass feed-forward-network (layer)
  ((d-model :initarg :d-model)
   (d-ff :initarg :d-ff)
   (linear1 :accessor linear1)
   (linear2 :accessor linear2)
   (dropout :initarg :dropout
            :initform 0.1
            :accessor dropout-rate))
  (:documentation "Position-wise feed-forward network: FFN(x) = max(0, xW1 + b1)W2 + b2"))

(defmethod initialize-instance :after ((ffn feed-forward-network) &key)
  (with-slots (d-model d-ff linear1 linear2 parameters) ffn
    (setf linear1 (linear d-model d-ff))
    (setf linear2 (linear d-ff d-model))
    (setf parameters (append (layer-parameters linear1)
                            (layer-parameters linear2)))))

(defmethod forward ((ffn feed-forward-network) input)
  "FFN forward pass"
  (with-slots (linear1 linear2) ffn
    (let* ((hidden (forward linear1 input))
           (activated (relu hidden))
           (output (forward linear2 activated)))
      output)))

;;;; ============================================================================
;;;; Layer Normalization
;;;; ============================================================================

(defclass layer-norm (layer)
  ((normalized-shape :initarg :normalized-shape
                     :reader normalized-shape)
   (gamma :accessor gamma :documentation "Scale parameter")
   (beta :accessor beta :documentation "Shift parameter")
   (eps :initarg :eps
        :initform 1d-5
        :accessor eps))
  (:documentation "Layer normalization"))

(defmethod initialize-instance :after ((ln layer-norm) &key)
  (with-slots (normalized-shape gamma beta parameters) ln
    (setf gamma (ones normalized-shape :requires-grad t :name "gamma"))
    (setf beta (zeros normalized-shape :requires-grad t :name "beta"))
    (setf parameters (list gamma beta))))

(defmethod forward ((ln layer-norm) input)
  "Layer normalization: LN(x) = γ * (x - μ) / √(σ² + ε) + β"
  (with-slots (gamma beta eps) ln
    ;; For now, just apply gamma and beta scaling (simplified layer norm)
    ;; Proper implementation would compute mean/variance along last dimension
    (t+ (t* input gamma) beta)))

;;;; ============================================================================
;;;; Transformer Encoder Layer
;;;; ============================================================================

(defclass transformer-encoder-layer (layer)
  ((d-model :initarg :d-model)
   (num-heads :initarg :num-heads)
   (d-ff :initarg :d-ff)
   (self-attn :accessor self-attn)
   (feed-forward :accessor feed-forward)
   (norm1 :accessor norm1)
   (norm2 :accessor norm2)
   (dropout :initarg :dropout
            :initform 0.1))
  (:documentation "Transformer encoder layer with self-attention and FFN"))

(defmethod initialize-instance :after ((layer transformer-encoder-layer) &key)
  (with-slots (d-model num-heads d-ff self-attn feed-forward norm1 norm2 parameters) layer
    (setf self-attn (make-instance 'multi-head-attention
                                   :d-model d-model
                                   :num-heads num-heads))
    (setf feed-forward (make-instance 'feed-forward-network
                                      :d-model d-model
                                      :d-ff d-ff))
    (setf norm1 (make-instance 'layer-norm
                              :normalized-shape (list d-model)))
    (setf norm2 (make-instance 'layer-norm
                              :normalized-shape (list d-model)))
    (setf parameters (append (layer-parameters self-attn)
                            (layer-parameters feed-forward)
                            (layer-parameters norm1)
                            (layer-parameters norm2)))))

(defmethod forward ((layer transformer-encoder-layer) input)
  "Encoder layer forward: x -> self-attention -> add&norm -> FFN -> add&norm"
  (transformer-encoder-layer-forward layer input nil))

(defun transformer-encoder-layer-forward (layer input mask)
  "Encoder layer forward with optional mask"
  (with-slots (self-attn feed-forward norm1 norm2) layer
    ;; Self-attention with residual connection
    (let* ((attn-output (if mask
                           (multi-head-attention-forward self-attn input input input :mask mask)
                           (forward self-attn input)))
           (attn-residual (t+ input attn-output))
           (norm1-output (forward norm1 attn-residual))
           ;; Feed-forward with residual connection
           (ff-output (forward feed-forward norm1-output))
           (ff-residual (t+ norm1-output ff-output))
           (output (forward norm2 ff-residual)))
      output)))

;;;; ============================================================================
;;;; Transformer Decoder Layer
;;;; ============================================================================

(defclass transformer-decoder-layer (layer)
  ((d-model :initarg :d-model)
   (num-heads :initarg :num-heads)
   (d-ff :initarg :d-ff)
   (self-attn :accessor self-attn :documentation "Masked self-attention")
   (cross-attn :accessor cross-attn :documentation "Cross-attention to encoder")
   (feed-forward :accessor feed-forward)
   (norm1 :accessor norm1)
   (norm2 :accessor norm2)
   (norm3 :accessor norm3))
  (:documentation "Transformer decoder layer"))

(defmethod initialize-instance :after ((layer transformer-decoder-layer) &key)
  (with-slots (d-model num-heads d-ff self-attn cross-attn feed-forward 
               norm1 norm2 norm3 parameters) layer
    (setf self-attn (make-instance 'multi-head-attention
                                   :d-model d-model
                                   :num-heads num-heads))
    (setf cross-attn (make-instance 'multi-head-attention
                                    :d-model d-model
                                    :num-heads num-heads))
    (setf feed-forward (make-instance 'feed-forward-network
                                      :d-model d-model
                                      :d-ff d-ff))
    (setf norm1 (make-instance 'layer-norm :normalized-shape (list d-model)))
    (setf norm2 (make-instance 'layer-norm :normalized-shape (list d-model)))
    (setf norm3 (make-instance 'layer-norm :normalized-shape (list d-model)))
    (setf parameters (append (layer-parameters self-attn)
                            (layer-parameters cross-attn)
                            (layer-parameters feed-forward)
                            (layer-parameters norm1)
                            (layer-parameters norm2)
                            (layer-parameters norm3)))))

(defmethod forward ((layer transformer-decoder-layer) input)
  "Decoder layer: masked self-attn -> cross-attn -> FFN (simplified for 2-arg forward)"
  (transformer-decoder-layer-forward layer input input nil nil))

(defun transformer-decoder-layer-forward (layer input encoder-output tgt-mask memory-mask)
  "Decoder layer: masked self-attn -> cross-attn -> FFN"
  (with-slots (self-attn cross-attn feed-forward norm1 norm2 norm3) layer
    ;; Masked self-attention
    (let* ((self-attn-out (if tgt-mask
                             (multi-head-attention-forward self-attn input input input :mask tgt-mask)
                             (forward self-attn input)))
           (residual1 (forward norm1 (t+ input self-attn-out)))
           ;; Cross-attention to encoder
           (cross-attn-out (if memory-mask
                              (multi-head-attention-forward cross-attn residual1 encoder-output encoder-output
                                           :mask memory-mask)
                              (multi-head-attention-forward cross-attn residual1 encoder-output encoder-output)))
           (residual2 (forward norm2 (t+ residual1 cross-attn-out)))
           ;; Feed-forward
           (ff-out (forward feed-forward residual2))
           (output (forward norm3 (t+ residual2 ff-out))))
      output)))

;;;; ============================================================================
;;;; Full Transformer Model
;;;; ============================================================================

(defclass transformer (layer)
  ((d-model :initarg :d-model :initform 512)
   (num-heads :initarg :num-heads :initform 8)
   (num-encoder-layers :initarg :num-encoder-layers :initform 6)
   (num-decoder-layers :initarg :num-decoder-layers :initform 6)
   (d-ff :initarg :d-ff :initform 2048)
   (encoder-layers :accessor encoder-layers)
   (decoder-layers :accessor decoder-layers)
   (pos-encoding :accessor pos-encoding))
  (:documentation "Full Transformer architecture"))

(defmethod initialize-instance :after ((model transformer) &key)
  (with-slots (d-model num-heads num-encoder-layers num-decoder-layers d-ff
               encoder-layers decoder-layers pos-encoding parameters) model
    ;; Create encoder layers
    (setf encoder-layers
          (loop repeat num-encoder-layers
                collect (make-instance 'transformer-encoder-layer
                                      :d-model d-model
                                      :num-heads num-heads
                                      :d-ff d-ff)))
    ;; Create decoder layers
    (setf decoder-layers
          (loop repeat num-decoder-layers
                collect (make-instance 'transformer-decoder-layer
                                      :d-model d-model
                                      :num-heads num-heads
                                      :d-ff d-ff)))
    ;; Positional encoding
    (setf pos-encoding (make-instance 'positional-encoding
                                     :d-model d-model))
    
    ;; Collect all parameters
    (setf parameters
          (append (reduce #'append
                         (mapcar #'layer-parameters encoder-layers))
                  (reduce #'append
                         (mapcar #'layer-parameters decoder-layers))))))

(defmethod forward ((model transformer) input)
  "Full transformer forward pass (simplified for 2-arg forward)"
  (transformer-forward model input input nil nil nil))

(defun transformer-forward (model src tgt src-mask tgt-mask memory-mask)
  "Full transformer forward pass
   
   src: source sequence (batch, src_len, d_model)
   tgt: target sequence (batch, tgt_len, d_model)"
  (with-slots (encoder-layers decoder-layers pos-encoding) model
    ;; Encode
    (let ((encoder-output (forward pos-encoding src)))
      (dolist (encoder-layer encoder-layers)
        (setf encoder-output
              (if src-mask
                  (transformer-encoder-layer-forward encoder-layer encoder-output src-mask)
                  (forward encoder-layer encoder-output))))
      
      ;; Decode
      (let ((decoder-output (forward pos-encoding tgt)))
        (dolist (decoder-layer decoder-layers)
          (setf decoder-output
                (transformer-decoder-layer-forward decoder-layer decoder-output encoder-output
                        tgt-mask memory-mask)))
        decoder-output))))

;;;; ============================================================================
;;;; Advanced Attention Variants
;;;; ============================================================================

(defclass rotary-embedding (layer)
  ((dim :initarg :dim)
   (max-len :initarg :max-len :initform 2048))
  (:documentation "Rotary Position Embedding (RoPE) - used in GPT-Neo, LLaMA"))

(defmethod forward ((rope rotary-embedding) x)
  "Apply rotary embeddings to queries and keys"
  ;; Simplified - would implement rotation matrices
  x)

(defclass alibi-attention (multi-head-attention)
  ()
  (:documentation "Attention with Linear Biases (ALiBi) - no positional encoding needed"))

;; ALiBi would modify attention scores - simplified here
;; Real implementation would override multi-head-attention-forward

(defun flash-attention (query key value)
  "Flash Attention - memory-efficient attention (conceptual)
   Real implementation would use tiling and recomputation"
  (scaled-dot-product-attention query key value))

(defun sparse-attention (query key value &key (sparsity-pattern :local) (window-size 128) (stride 64) (num-global 4))
  "Sparse attention patterns for long sequences
   
   Patterns:
   - :local - only attend to nearby tokens (within window-size)
   - :strided - attend to every k-th token (stride)
   - :fixed - attend to fixed positions (first num-global tokens)
   - :global - all tokens attend to first num-global tokens, and those attend to all"
  (let* ((shape (neural-network::tensor-shape query))
         (batch-size (first shape))
         (seq-len (second shape))
         ;; Create mask with batch dimension
         (mask (make-array (list batch-size seq-len seq-len)
                          :element-type 'double-float
                          :initial-element -1d10))) ; -inf for masked positions
    
    (ecase sparsity-pattern
      (:local
       ;; Each token attends to tokens within window-size
       (dotimes (b batch-size)
         (dotimes (i seq-len)
           (loop for j from (max 0 (- i window-size))
                 to (min (1- seq-len) (+ i window-size))
                 do (setf (aref mask b i j) 0.0d0)))))
      
      (:strided
       ;; Each token attends to every stride-th token
       (dotimes (b batch-size)
         (dotimes (i seq-len)
           (loop for j from 0 below seq-len by stride
                 do (setf (aref mask b i j) 0.0d0))
           ;; Also attend to nearby tokens
           (loop for j from (max 0 (- i 2))
                 to (min (1- seq-len) (+ i 2))
                 do (setf (aref mask b i j) 0.0d0)))))
      
      (:fixed
       ;; All tokens attend to first num-global tokens
       (dotimes (b batch-size)
         (dotimes (i seq-len)
           (dotimes (j (min num-global seq-len))
             (setf (aref mask b i j) 0.0d0))
           ;; Also attend to self
           (setf (aref mask b i i) 0.0d0))))
      
      (:global
       ;; First num-global tokens attend to all
       (dotimes (b batch-size)
         (dotimes (i (min num-global seq-len))
           (dotimes (j seq-len)
             (setf (aref mask b i j) 0.0d0)))
         ;; All tokens attend to first num-global tokens
         (dotimes (i seq-len)
           (dotimes (j (min num-global seq-len))
             (setf (aref mask b i j) 0.0d0))
           ;; Also attend to self
           (setf (aref mask b i i) 0.0d0)))))
    
    (let ((mask-tensor (make-tensor mask :shape (list batch-size seq-len seq-len))))
      (scaled-dot-product-attention query key value :mask mask-tensor))))

;;;; ============================================================================
;;;; Attention Masks (Lisp macros for mask generation)
;;;; ============================================================================

(defun causal-mask (size)
  "Generate causal mask for autoregressive decoding
   Lower triangular matrix of ones"
  (let ((mask (make-array (list size size)
                         :element-type 'double-float
                         :initial-element 0.0d0)))
    (dotimes (i size)
      (dotimes (j (1+ i))
        (setf (aref mask i j) 1.0d0)))
    (make-tensor mask :shape (list size size))))

(defun padding-mask (lengths max-len)
  "Generate padding mask from sequence lengths"
  (let* ((batch-size (length lengths))
         (mask (make-array (list batch-size max-len)
                          :element-type 'double-float
                          :initial-element 0.0d0)))
    (loop for i from 0 below batch-size
          for len in lengths
          do (dotimes (j len)
               (setf (aref mask i j) 1.0d0)))
    (make-tensor mask :shape (list batch-size max-len))))

(defmacro with-attention-mask (mask &body body)
  "Execute body with attention mask context"
  `(let ((*current-attention-mask* ,mask))
     ,@body))

(defvar *current-attention-mask* nil
  "Dynamic variable for attention mask")

;;;; ============================================================================
;;;; DSL for defining custom attention patterns
;;;; ============================================================================

(defmacro defattention (name (query-var key-var value-var) &body body)
  "Define a custom attention mechanism"
  `(defun ,name (,query-var ,key-var ,value-var &key mask dropout)
     (declare (ignorable mask dropout))
     ,@body))

(defmacro attention-pattern (pattern-type &rest params)
  "Generate attention pattern
   
   Examples:
   (attention-pattern :causal :size 512)
   (attention-pattern :local :window-size 256)
   (attention-pattern :global :global-tokens '(0 1 2))"
  `(ecase ,pattern-type
     (:causal (causal-mask ,(getf params :size)))
     (:local (local-attention-mask ,@params))
     (:global (global-attention-mask ,@params))))

;;;; ============================================================================
;;;; Example: Vision Transformer (ViT) components
;;;; ============================================================================

(defclass patch-embedding (layer)
  ((patch-size :initarg :patch-size)
   (d-model :initarg :d-model)
   (projection :accessor projection))
  (:documentation "Convert image patches to embeddings"))

(defmethod initialize-instance :after ((pe patch-embedding) &key)
  (with-slots (patch-size d-model projection parameters) pe
    ;; Would be a convolutional layer in practice
    (setf projection (linear (* patch-size patch-size 3) d-model))
    (setf parameters (layer-parameters projection))))

;;;; ============================================================================
;;;; Example usage functions
;;;; ============================================================================

(defun create-gpt-style-decoder (vocab-size d-model num-heads num-layers)
  "Create GPT-style decoder-only transformer"
  (list :embedding (randn (list vocab-size d-model) :requires-grad t)
        :pos-encoding (make-instance 'positional-encoding :d-model d-model)
        :layers (loop repeat num-layers
                     collect (make-instance 'transformer-encoder-layer
                                           :d-model d-model
                                           :num-heads num-heads
                                           :d-ff (* 4 d-model)))
        :lm-head (linear d-model vocab-size)))

(defun create-bert-style-encoder (vocab-size d-model num-heads num-layers)
  "Create BERT-style encoder-only transformer"
  (create-gpt-style-decoder vocab-size d-model num-heads num-layers))
