;;;; embedding.lisp - Token Embedding Layers
;;;; Provides various embedding types: word, subword, positional, OOV handling, numerical, and byte embeddings

(defpackage :neural-network.embedding
  (:use :common-lisp :neural-network)
  (:export #:embedding-layer
           #:word-embedding
           #:subword-embedding
           #:positional-encoding
           #:sinusoidal-positional-encoding
           #:learned-positional-encoding
           #:embedding-with-oov
           #:numerical-embedding
           #:byte-embedding
           #:combined-embedding
           ;; Accessors
           #:emb-vocab-size
           #:embedding-dim
           #:embeddings
           #:emb-max-length
           #:oov-token
           #:oov-index
           #:pad-index
           #:padding-idx
           #:max-subwords
           #:aggregation
           #:encodings
           #:position-embeddings
           #:oov-strategy
           #:input-dim
           #:projection
           #:byte-vocab-size
           #:max-bytes
           #:token-embedding
           #:use-normalization
           ;; Functions
           #:lookup
           #:get-embedding))

(in-package :neural-network.embedding)

;;;; ============================================================================
;;;; Base Embedding Layer
;;;; ============================================================================

(defclass embedding-layer (layer)
  ((vocab-size :initarg :vocab-size
               :accessor emb-vocab-size
               :documentation "Size of vocabulary")
   (embedding-dim :initarg :embedding-dim
                  :accessor embedding-dim
                  :documentation "Dimension of embeddings")
   (embeddings :accessor embeddings
               :documentation "Embedding weight matrix"))
  (:documentation "Base class for embedding layers"))

;;;; ============================================================================
;;;; Word Embedding Layer
;;;; ============================================================================

(defclass word-embedding (embedding-layer)
  ((padding-idx :initarg :padding-idx
                :accessor padding-idx
                :initform nil
                :documentation "Index for padding token (gradient will be zero)"))
  (:documentation "Standard word embedding layer - maps token IDs to dense vectors"))

(defmethod initialize-instance :after ((layer word-embedding) &key)
  (with-slots (embeddings vocab-size embedding-dim parameters padding-idx) layer
    ;; Initialize embeddings with scaled random values (Xavier initialization)
    (let ((scale (/ 1.0 (sqrt embedding-dim))))
      (setf embeddings (randn (list vocab-size embedding-dim)
                              :requires-grad t
                              :name "word-embeddings"
                              :scale scale)))
    
    ;; Zero out padding embedding if specified
    (when padding-idx
      (let ((emb-data (tensor-data embeddings)))
        (dotimes (j embedding-dim)
          (setf (aref emb-data padding-idx j) 0.0d0))))
    
    (setf parameters (list embeddings))))

(defmethod forward ((layer word-embedding) indices)
  "Look up embeddings for given token indices
   indices: tensor of shape (batch-size,) or (batch-size, seq-len)"
  (let* ((indices-data (tensor-data indices))
         (indices-shape (tensor-shape indices))
         (emb-data (tensor-data (embeddings layer)))
         (emb-dim (embedding-dim layer))
         (padding-idx (padding-idx layer))
         (result-shape (append indices-shape (list emb-dim)))
         (result-data (make-array result-shape :element-type 'double-float)))
    
    ;; Copy embeddings for each index
    (dotimes (i (array-total-size indices-data))
      (let ((idx (floor (row-major-aref indices-data i))))
        ;; Zero out padding embeddings
        (if (and padding-idx (= idx padding-idx))
            (dotimes (j emb-dim)
              (setf (row-major-aref result-data (+ (* i emb-dim) j)) 0.0d0))
            (dotimes (j emb-dim)
              (setf (row-major-aref result-data (+ (* i emb-dim) j))
                    (aref emb-data idx j))))))
    
    (make-tensor result-data
                 :shape result-shape
                 :requires-grad (requires-grad (embeddings layer)))))

;;;; ============================================================================
;;;; Subword Embedding Layer
;;;; ============================================================================

(defclass subword-embedding (embedding-layer)
  ((max-subwords :initarg :max-subwords
                 :accessor max-subwords
                 :initform 5
                 :documentation "Maximum number of subwords per token")
   (aggregation :initarg :aggregation
                :accessor aggregation
                :initform :mean
                :documentation "How to aggregate subword embeddings (:mean, :sum, :max)"))
  (:documentation "Subword embedding layer (e.g., BPE, WordPiece) with aggregation"))

(defmethod initialize-instance :after ((layer subword-embedding) &key)
  (with-slots (embeddings vocab-size embedding-dim parameters) layer
    (let ((scale (/ 1.0 (sqrt embedding-dim))))
      (setf embeddings (randn (list vocab-size embedding-dim)
                              :requires-grad t
                              :name "subword-embeddings"
                              :scale scale)))
    (setf parameters (list embeddings))))

(defmethod forward ((layer subword-embedding) subword-indices)
  "Look up and aggregate subword embeddings
   subword-indices: tensor of shape (batch-size, max-subwords) where each row contains subword IDs
                    Use -1 or vocab-size for padding"
  (let* ((indices-data (tensor-data subword-indices))
         (indices-shape (tensor-shape subword-indices))
         (batch-size (first indices-shape))
         (max-sw (second indices-shape))
         (emb-data (tensor-data (embeddings layer)))
         (emb-dim (embedding-dim layer))
         (agg-method (aggregation layer))
         (vocab-sz (emb-vocab-size layer))
         (result-shape (list batch-size emb-dim))
         (result-data (make-array result-shape :element-type 'double-float :initial-element 0.0d0)))
    
    ;; For each token in batch, aggregate its subword embeddings
    (dotimes (b batch-size)
      (let ((counts 0))
        (dotimes (s max-sw)
          (let ((idx (floor (aref indices-data b s))))
            (when (and (>= idx 0) (< idx vocab-sz))
              (incf counts)
              (dotimes (d emb-dim)
                (ecase agg-method
                  (:sum (incf (aref result-data b d) (aref emb-data idx d)))
                  (:mean (incf (aref result-data b d) (aref emb-data idx d)))
                  (:max (setf (aref result-data b d)
                             (max (aref result-data b d) (aref emb-data idx d)))))))))
        ;; For mean aggregation, divide by count
        (when (and (eq agg-method :mean) (> counts 0))
          (dotimes (d emb-dim)
            (setf (aref result-data b d) (/ (aref result-data b d) counts))))))
    
    (make-tensor result-data
                 :shape result-shape
                 :requires-grad (requires-grad (embeddings layer)))))

;;;; ============================================================================
;;;; Positional Encoding Layers
;;;; ============================================================================

(defclass positional-encoding (layer)
  ((max-length :initarg :max-length
               :accessor emb-max-length
               :documentation "Maximum sequence length")
   (embedding-dim :initarg :embedding-dim
                  :accessor embedding-dim
                  :documentation "Dimension of position embeddings"))
  (:documentation "Base class for positional encodings"))

;;;; Sinusoidal Positional Encoding (as in "Attention is All You Need")
(defclass sinusoidal-positional-encoding (positional-encoding)
  ((encodings :accessor encodings
              :documentation "Pre-computed sinusoidal encodings"))
  (:documentation "Fixed sinusoidal positional encodings"))

(defmethod initialize-instance :after ((layer sinusoidal-positional-encoding) &key)
  (with-slots (encodings max-length embedding-dim) layer
    ;; Pre-compute sinusoidal encodings
    (let ((pe-data (make-array (list max-length embedding-dim) :element-type 'double-float)))
      (dotimes (pos max-length)
        (dotimes (i embedding-dim)
          (let* ((pair-idx (floor i 2))
                 (div-term (exp (* -1.0d0 pair-idx (/ (log 10000.0d0) (/ embedding-dim 2.0d0)))))
                 (angle (* pos div-term)))
            (setf (aref pe-data pos i)
                  (if (evenp i)
                      (sin angle)
                      (cos angle))))))
      (setf encodings (make-tensor pe-data
                                   :shape (list max-length embedding-dim)
                                   :requires-grad nil)))))

(defmethod forward ((layer sinusoidal-positional-encoding) x)
  "Add sinusoidal positional encodings to input
   x: tensor of shape (batch-size, seq-len, embedding-dim)"
  (let* ((shape (tensor-shape x))
         (seq-len (second shape))
         (pe-data (tensor-data (encodings layer)))
         (x-data (tensor-data x))
         (result-data (make-array shape :element-type 'double-float)))
    
    ;; Add positional encodings
    (dotimes (b (first shape))
      (dotimes (s seq-len)
        (dotimes (d (third shape))
          (setf (aref result-data b s d)
                (+ (aref x-data b s d)
                   (aref pe-data s d))))))
    
    (make-tensor result-data
                 :shape shape
                 :requires-grad (requires-grad x))))

;;;; Learned Positional Encoding
(defclass learned-positional-encoding (positional-encoding)
  ((position-embeddings :accessor position-embeddings
                        :documentation "Learned position embedding matrix"))
  (:documentation "Learned positional embeddings"))

(defmethod initialize-instance :after ((layer learned-positional-encoding) &key)
  (with-slots (position-embeddings max-length embedding-dim parameters) layer
    (let ((scale (/ 1.0 (sqrt embedding-dim))))
      (setf position-embeddings (randn (list max-length embedding-dim)
                                       :requires-grad t
                                       :name "position-embeddings"
                                       :scale scale)))
    (setf parameters (list position-embeddings))))

(defmethod forward ((layer learned-positional-encoding) x)
  "Add learned positional embeddings to input
   x: tensor of shape (batch-size, seq-len, embedding-dim)"
  (let* ((shape (tensor-shape x))
         (seq-len (second shape))
         (pe-data (tensor-data (position-embeddings layer)))
         (x-data (tensor-data x))
         (result-data (make-array shape :element-type 'double-float)))
    
    ;; Add positional embeddings
    (dotimes (b (first shape))
      (dotimes (s seq-len)
        (dotimes (d (third shape))
          (setf (aref result-data b s d)
                (+ (aref x-data b s d)
                   (aref pe-data s d))))))
    
    (make-tensor result-data
                 :shape shape
                 :requires-grad (requires-grad x))))

;;;; ============================================================================
;;;; Embedding with Out-of-Vocabulary (OOV) Handling
;;;; ============================================================================

(defclass embedding-with-oov (embedding-layer)
  ((oov-index :initarg :oov-index
              :accessor oov-index
              :documentation "Index for out-of-vocabulary tokens")
   (pad-index :initarg :pad-index
              :accessor pad-index
              :initform nil
              :documentation "Index for padding tokens")
   (oov-strategy :initarg :oov-strategy
                 :accessor oov-strategy
                 :initform :learned
                 :documentation "Strategy for OOV: :learned, :zero, :random"))
  (:documentation "Embedding layer with explicit OOV handling"))

(defmethod initialize-instance :after ((layer embedding-with-oov) &key)
  (with-slots (embeddings vocab-size embedding-dim parameters oov-index pad-index) layer
    (let ((scale (/ 1.0 (sqrt embedding-dim))))
      (setf embeddings (randn (list vocab-size embedding-dim)
                              :requires-grad t
                              :name "oov-embeddings"
                              :scale scale)))
    
    ;; Initialize OOV embedding specially
    (let ((emb-data (tensor-data embeddings)))
      (dotimes (j embedding-dim)
        (setf (aref emb-data oov-index j) 0.0d0))
      
      ;; Zero out padding if specified
      (when pad-index
        (dotimes (j embedding-dim)
          (setf (aref emb-data pad-index j) 0.0d0))))
    
    (setf parameters (list embeddings))))

(defmethod forward ((layer embedding-with-oov) indices)
  "Look up embeddings, handling OOV tokens
   indices: tensor of shape (batch-size,) or (batch-size, seq-len)"
  (let* ((indices-data (tensor-data indices))
         (indices-shape (tensor-shape indices))
         (emb-data (tensor-data (embeddings layer)))
         (emb-dim (embedding-dim layer))
         (vocab-sz (emb-vocab-size layer))
         (oov-idx (oov-index layer))
         (pad-idx (pad-index layer))
         (result-shape (append indices-shape (list emb-dim)))
         (result-data (make-array result-shape :element-type 'double-float)))
    
    ;; Copy embeddings, replacing invalid indices with OOV
    (dotimes (i (array-total-size indices-data))
      (let ((idx (floor (row-major-aref indices-data i))))
        ;; Check if index is valid
        (when (or (< idx 0) (>= idx vocab-sz))
          (setf idx oov-idx))
        
        ;; Zero out padding
        (if (and pad-idx (= idx pad-idx))
            (dotimes (j emb-dim)
              (setf (row-major-aref result-data (+ (* i emb-dim) j)) 0.0d0))
            (dotimes (j emb-dim)
              (setf (row-major-aref result-data (+ (* i emb-dim) j))
                    (aref emb-data idx j))))))
    
    (make-tensor result-data
                 :shape result-shape
                 :requires-grad (requires-grad (embeddings layer)))))

;;;; ============================================================================
;;;; Numerical Embedding Layer
;;;; ============================================================================

(defclass numerical-embedding (layer)
  ((input-dim :initarg :input-dim
              :accessor input-dim
              :documentation "Number of numerical features")
   (embedding-dim :initarg :embedding-dim
                  :accessor embedding-dim
                  :documentation "Output embedding dimension")
   (projection :accessor projection
               :documentation "Linear projection for numerical features")
   (use-normalization :initarg :use-normalization
                      :accessor use-normalization
                      :initform t
                      :documentation "Whether to apply layer normalization"))
  (:documentation "Embed numerical features into dense vectors"))

(defmethod initialize-instance :after ((layer numerical-embedding) &key)
  (with-slots (projection input-dim embedding-dim parameters) layer
    ;; Create linear projection
    (setf projection (linear input-dim embedding-dim))
    (setf parameters (layer-parameters projection))))

(defmethod forward ((layer numerical-embedding) x)
  "Project numerical features to embedding space
   x: tensor of shape (batch-size, input-dim) or (batch-size, seq-len, input-dim)"
  (let ((projected (forward (projection layer) x)))
    ;; Could add normalization here if use-normalization is true
    projected))

;;;; ============================================================================
;;;; Byte Embedding Layer
;;;; ============================================================================

(defclass byte-embedding (embedding-layer)
  ((byte-vocab-size :initform 256
                    :accessor byte-vocab-size
                    :documentation "Number of byte values (0-255)")
   (max-bytes :initarg :max-bytes
              :accessor max-bytes
              :initform 16
              :documentation "Maximum bytes per token")
   (aggregation :initarg :aggregation
                :accessor aggregation
                :initform :cnn
                :documentation "How to aggregate byte embeddings (:mean, :sum, :cnn)"))
  (:documentation "Byte-level embedding for handling arbitrary Unicode/rare words"))

(defmethod initialize-instance :after ((layer byte-embedding) &key)
  (with-slots (embeddings byte-vocab-size embedding-dim parameters vocab-size) layer
    ;; Override vocab-size with byte vocab size
    (setf vocab-size byte-vocab-size)
    
    (let ((scale (/ 1.0 (sqrt embedding-dim))))
      (setf embeddings (randn (list byte-vocab-size embedding-dim)
                              :requires-grad t
                              :name "byte-embeddings"
                              :scale scale)))
    (setf parameters (list embeddings))))

(defmethod forward ((layer byte-embedding) byte-indices)
  "Look up and aggregate byte embeddings
   byte-indices: tensor of shape (batch-size, max-bytes) where each row contains byte values (0-255)
                Use -1 for padding"
  (let* ((indices-data (tensor-data byte-indices))
         (indices-shape (tensor-shape byte-indices))
         (batch-size (first indices-shape))
         (max-b (second indices-shape))
         (emb-data (tensor-data (embeddings layer)))
         (emb-dim (embedding-dim layer))
         (agg-method (aggregation layer))
         (result-shape (list batch-size emb-dim))
         (result-data (make-array result-shape :element-type 'double-float :initial-element 0.0d0)))
    
    ;; For each token in batch, aggregate its byte embeddings
    (dotimes (b batch-size)
      (let ((counts 0))
        (dotimes (s max-b)
          (let ((idx (floor (aref indices-data b s))))
            (when (and (>= idx 0) (< idx 256))
              (incf counts)
              (dotimes (d emb-dim)
                (ecase agg-method
                  (:sum (incf (aref result-data b d) (aref emb-data idx d)))
                  (:mean (incf (aref result-data b d) (aref emb-data idx d)))
                  ;; :cnn would require convolution layers - simplified to max for now
                  (:cnn (setf (aref result-data b d)
                             (max (aref result-data b d) (aref emb-data idx d)))))))))
        ;; For mean aggregation, divide by count
        (when (and (eq agg-method :mean) (> counts 0))
          (dotimes (d emb-dim)
            (setf (aref result-data b d) (/ (aref result-data b d) counts))))))
    
    (make-tensor result-data
                 :shape result-shape
                 :requires-grad (requires-grad (embeddings layer)))))

;;;; ============================================================================
;;;; Combined Embedding Layer
;;;; ============================================================================

(defclass combined-embedding (layer)
  ((token-embedding :initarg :token-embedding
                    :accessor token-embedding
                    :documentation "Primary token embedding layer")
   (positional-encoding :initarg :positional-encoding
                        :accessor positional-encoding
                        :initform nil
                        :documentation "Optional positional encoding layer")
   (dropout-rate :initarg :dropout-rate
                 :accessor dropout-rate
                 :initform 0.1
                 :documentation "Dropout rate after embedding"))
  (:documentation "Combined embedding with tokens + positions + dropout"))

(defmethod initialize-instance :after ((layer combined-embedding) &key)
  (with-slots (token-embedding positional-encoding parameters) layer
    (setf parameters (layer-parameters token-embedding))
    (when positional-encoding
      (setf parameters (append parameters (layer-parameters positional-encoding))))))

(defmethod forward ((layer combined-embedding) indices)
  "Combine token embeddings with positional encodings
   indices: tensor of shape (batch-size, seq-len)"
  (let* ((token-emb (forward (token-embedding layer) indices))
         (result (if (positional-encoding layer)
                     (forward (positional-encoding layer) token-emb)
                     token-emb)))
    ;; TODO: Add dropout when in training mode
    result))

;;;; ============================================================================
;;;; Utility Functions
;;;; ============================================================================

(defun lookup (embedding-layer indices)
  "Convenience function for embedding lookup"
  (forward embedding-layer indices))

(defun get-embedding (embedding-layer index)
  "Get a single embedding vector by index"
  (let* ((emb-data (tensor-data (embeddings embedding-layer)))
         (emb-dim (embedding-dim embedding-layer))
         (result-data (make-array (list emb-dim) :element-type 'double-float)))
    (dotimes (i emb-dim)
      (setf (aref result-data i) (aref emb-data index i)))
    (make-tensor result-data :shape (list emb-dim))))
