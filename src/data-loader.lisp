;;;; Neural Tensor Library - Generic Data Loader Interface
;;;; Provides standard interface for loading and batching data for all architectures
;;;; Compatible with HuggingFace and PyTorch dataset formats

(defpackage :neural-data-loader
  (:use :common-lisp)
  (:import-from :neural-network
                #:tensor
                #:make-tensor
                #:zeros
                #:randn
                #:tensor-shape)
  (:export #:data-loader
           #:dataset
           #:make-data-loader
           #:get-batch
           #:dataset-length
           #:num-batches
           #:reset-loader
           #:shuffle-dataset
           #:collate-fn
           #:get-item
           #:get-item-by-key
           #:dataset-features
           #:dataset-split
           ;; Dataset types
           #:sequence-dataset
           #:image-dataset
           #:text-dataset
           #:tensor-dataset
           #:huggingface-dataset
           #:pytorch-dataset
           #:map-style-dataset
           #:iterable-dataset
           #:csv-dataset
           #:json-dataset
           #:arrow-dataset
           ;; Dataset accessors
           #:max-length
           #:vocab-size
           #:batch-size
           #:dataset-cache
           #:dataset-split-name
           #:dataset-features-dict
           ;; Format loaders
           #:load-csv-dataset
           #:load-json-dataset
           #:load-huggingface-format
           #:load-pytorch-format
           #:load-arrow-dataset
           #:from-dict
           #:from-list
           ;; Data processing
           #:map-dataset
           #:filter-dataset
           #:select-columns
           #:rename-column
           #:train-test-split
           #:concatenate-datasets
           ;; Utilities
           #:pad-sequences
           #:create-attention-mask
           #:normalize-batch
           #:tokenize-batch
           #:compute-dataset-statistics
           ;; Samplers
           #:sampler
           #:sequential-sampler
           #:random-sampler
           #:weighted-sampler
           #:distributed-sampler))

(in-package :neural-data-loader)

;;;; ============================================================================
;;;; Base Dataset Protocol
;;;; ============================================================================

(defclass dataset ()
  ((data :initarg :data
         :accessor dataset-data
         :documentation "Raw data samples")
   (labels :initarg :labels
           :initform nil
           :accessor dataset-labels
           :documentation "Labels/targets for supervised learning")
   (features :initarg :features
             :initform nil
             :accessor dataset-features
             :documentation "Feature schema/metadata (HuggingFace-style)")
   (cache :initarg :cache
          :initform (make-hash-table :test 'equal)
          :accessor dataset-cache
          :documentation "Cache for processed items")
   (transform :initarg :transform
              :initform nil
              :accessor dataset-transform
              :documentation "Optional transformation function"))
  (:documentation "Base dataset class following PyTorch-like interface"))

(defgeneric dataset-length (dataset)
  (:documentation "Return the number of samples in the dataset"))

(defgeneric get-item (dataset index)
  (:documentation "Get a single item from the dataset at index"))

(defgeneric get-item-by-key (dataset index key)
  (:documentation "Get specific feature/column from dataset item (HuggingFace-style)"))

(defmethod dataset-length ((ds dataset))
  "Default implementation: length of data slot"
  (length (dataset-data ds)))

(defmethod get-item ((ds dataset) index)
  "Default implementation: get item with optional transform and caching"
  (let ((cache-key (format nil "~A" index))
        (cache (dataset-cache ds)))
    ;; Check cache first
    (multiple-value-bind (cached-value present)
        (gethash cache-key cache)
      (if present
          (if (dataset-labels ds)
              (values (car cached-value) (cdr cached-value))
              cached-value)
          ;; Not cached, fetch and process
          (let ((item (elt (dataset-data ds) index))
                (label (when (dataset-labels ds)
                         (elt (dataset-labels ds) index))))
            (when (dataset-transform ds)
              (setf item (funcall (dataset-transform ds) item)))
            ;; Cache the result
            (if label
                (progn
                  (setf (gethash cache-key cache) (cons item label))
                  (values item label))
                (progn
                  (setf (gethash cache-key cache) item)
                  item)))))))

(defmethod get-item-by-key ((ds dataset) index key)
  "Get specific feature from dataset (for dict-like datasets)"
  (let ((item (get-item ds index)))
    (cond
      ((hash-table-p item) (gethash key item))
      ((listp item) (cdr (assoc key item :test #'equal)))
      (t item))))

;;;; ============================================================================
;;;; Specialized Dataset Types
;;;; ============================================================================

(defclass sequence-dataset (dataset)
  ((max-length :initarg :max-length
               :initform nil
               :accessor max-length
               :documentation "Maximum sequence length (for padding)")
   (vocab-size :initarg :vocab-size
               :initform nil
               :accessor vocab-size
               :documentation "Vocabulary size for text/token data"))
  (:documentation "Dataset for sequential data (text, time series, etc.)"))

(defclass image-dataset (dataset)
  ((height :initarg :height
           :accessor image-height)
   (width :initarg :width
          :accessor image-width)
   (channels :initarg :channels
             :initform 3
             :accessor image-channels))
  (:documentation "Dataset for image data"))

(defclass text-dataset (sequence-dataset)
  ((tokenizer :initarg :tokenizer
              :initform #'identity
              :accessor tokenizer
              :documentation "Function to tokenize text"))
  (:documentation "Dataset specifically for text data"))

(defclass tensor-dataset (dataset)
  ((tensor-shape :initarg :tensor-shape
                 :accessor tensor-shape-slot
                 :documentation "Expected shape of each tensor sample"))
  (:documentation "Dataset for pre-tensorized data"))

;;;; HuggingFace-compatible Dataset
(defclass huggingface-dataset (dataset)
  ((split-name :initarg :split
               :initform "train"
               :accessor dataset-split-name
               :documentation "Split name (train/validation/test)")
   (features-dict :initarg :features
                  :initform nil
                  :accessor dataset-features-dict
                  :documentation "Dictionary mapping column names to feature types")
   (format :initarg :format
           :initform :dict
           :accessor dataset-format
           :documentation "Output format (:dict, :list, :tensor)")
   (columns :initarg :columns
            :initform nil
            :accessor dataset-columns
            :documentation "List of column names"))
  (:documentation "HuggingFace Datasets-compatible interface"))

(defmethod get-item ((ds huggingface-dataset) index)
  "HuggingFace-style item access returning dict-like structure"
  (let ((item (call-next-method))
        (cols (dataset-columns ds)))
    ;; If item is already structured, return it
    (if (or (hash-table-p item) (and (listp item) (listp (car item))))
        item
        ;; Otherwise wrap in appropriate format
        (ecase (dataset-format ds)
          (:dict (let ((dict (make-hash-table :test 'equal)))
                   (if cols
                       (loop for col in cols
                             for val in (if (listp item) item (list item))
                             do (setf (gethash col dict) val))
                       (setf (gethash "data" dict) item))
                   dict))
          (:list (if (listp item) item (list item)))
          (:tensor item)))))

;;;; PyTorch-compatible Dataset Types
(defclass pytorch-dataset (dataset)
  ((dataset-type :initarg :dataset-type
                 :initform :map-style
                 :accessor dataset-type
                 :documentation "Map-style or iterable"))
  (:documentation "PyTorch-compatible dataset"))

(defclass map-style-dataset (pytorch-dataset)
  ()
  (:documentation "Map-style dataset (random access via index)"))

(defclass iterable-dataset (pytorch-dataset)
  ((iterator-fn :initarg :iterator-fn
                :accessor iterator-fn
                :documentation "Function that returns iterator")
   (current-iterator :initform nil
                     :accessor current-iterator))
  (:documentation "Iterable-style dataset (sequential access)"))

;;;; File Format Datasets
(defclass csv-dataset (dataset)
  ((filepath :initarg :filepath
             :accessor csv-filepath
             :documentation "Path to CSV file")
   (delimiter :initarg :delimiter
              :initform #\,
              :accessor csv-delimiter)
   (header :initarg :header
           :initform t
           :accessor csv-header
           :documentation "Whether first row is header"))
  (:documentation "Dataset loaded from CSV file"))

(defclass json-dataset (dataset)
  ((filepath :initarg :filepath
             :accessor json-filepath
             :documentation "Path to JSON/JSONL file")
   (format :initarg :format
           :initform :json
           :accessor json-format
           :documentation ":json or :jsonl"))
  (:documentation "Dataset loaded from JSON file"))

(defclass arrow-dataset (dataset)
  ((filepath :initarg :filepath
             :accessor arrow-filepath
             :documentation "Path to Arrow/Parquet file")
   (schema :initarg :schema
           :accessor arrow-schema
           :documentation "Arrow schema"))
  (:documentation "Dataset loaded from Apache Arrow/Parquet format"))

;;;; ============================================================================
;;;; Samplers for Custom Sampling Strategies
;;;; ============================================================================

(defclass sampler ()
  ((dataset-length :initarg :dataset-length
                   :accessor sampler-dataset-length
                   :documentation "Length of dataset to sample from"))
  (:documentation "Base sampler class"))

(defgeneric get-sampler-indices (sampler n)
  (:documentation "Get indices for sampling from dataset of length n"))

(defclass sequential-sampler (sampler)
  ()
  (:documentation "Sequential sampler (no shuffling)"))

(defmethod get-sampler-indices ((s sequential-sampler) n)
  (loop for i from 0 below n collect i))

(defclass random-sampler (sampler)
  ((replacement :initarg :replacement
                :initform nil
                :accessor sampler-replacement
                :documentation "Whether to sample with replacement")
   (num-samples :initarg :num-samples
                :initform nil
                :accessor sampler-num-samples
                :documentation "Number of samples to draw"))
  (:documentation "Random sampler"))

(defmethod get-sampler-indices ((s random-sampler) n)
  (let ((num-samples (or (sampler-num-samples s) n)))
    (if (sampler-replacement s)
        ;; With replacement
        (loop repeat num-samples collect (random n))
        ;; Without replacement (shuffle)
        (let ((indices (shuffle-indices n)))
          (subseq indices 0 (min num-samples n))))))

(defclass weighted-sampler (sampler)
  ((weights :initarg :weights
            :accessor sampler-weights
            :documentation "Weights for each sample")
   (num-samples :initarg :num-samples
                :accessor sampler-num-samples
                :documentation "Number of samples to draw")
   (replacement :initarg :replacement
                :initform t
                :accessor sampler-replacement))
  (:documentation "Weighted random sampler"))

(defmethod get-sampler-indices ((s weighted-sampler) n)
  (let* ((weights (sampler-weights s))
         (num-samples (sampler-num-samples s))
         (total-weight (reduce #'+ weights)))
    (loop repeat num-samples
          collect (weighted-random-choice weights total-weight))))

(defun weighted-random-choice (weights total-weight)
  "Select random index based on weights"
  (let ((r (* (random 1.0) total-weight))
        (cumsum 0.0))
    (loop for w in weights
          for i from 0
          do (incf cumsum w)
             (when (>= cumsum r)
               (return i))
          finally (return (1- (length weights))))))

(defclass distributed-sampler (sampler)
  ((num-replicas :initarg :num-replicas
                 :initform 1
                 :accessor sampler-num-replicas
                 :documentation "Number of distributed processes")
   (rank :initarg :rank
         :initform 0
         :accessor sampler-rank
         :documentation "Rank of current process")
   (shuffle :initarg :shuffle
            :initform t
            :accessor sampler-shuffle))
  (:documentation "Sampler for distributed training"))

(defmethod get-sampler-indices ((s distributed-sampler) n)
  (let* ((num-replicas (sampler-num-replicas s))
         (rank (sampler-rank s))
         (indices (if (sampler-shuffle s)
                      (shuffle-indices n)
                      (loop for i from 0 below n collect i)))
         ;; Partition indices across replicas
         (per-replica (ceiling n num-replicas))
         (start (* rank per-replica))
         (end (min (+ start per-replica) n)))
    (subseq indices start end)))

;;;; ============================================================================
;;;; Data Loader
;;;; ============================================================================

(defclass data-loader ()
  ((dataset :initarg :dataset
            :reader loader-dataset
            :documentation "Dataset to load from")
   (batch-size :initarg :batch-size
               :initform 32
               :reader batch-size
               :documentation "Number of samples per batch")
   (shuffle :initarg :shuffle
            :initform nil
            :accessor shuffle-p
            :documentation "Whether to shuffle data each epoch")
   (drop-last :initarg :drop-last
              :initform nil
              :accessor drop-last-p
              :documentation "Drop last incomplete batch")
   (collate-fn :initarg :collate-fn
               :initform #'default-collate
               :accessor collate-fn
               :documentation "Function to collate samples into batches")
   (sampler :initarg :sampler
            :initform nil
            :accessor loader-sampler
            :documentation "Optional sampler for custom sampling strategy")
   (num-workers :initarg :num-workers
                :initform 0
                :accessor num-workers
                :documentation "Number of worker threads (future: parallel loading)")
   (pin-memory :initarg :pin-memory
               :initform nil
               :accessor pin-memory-p
               :documentation "Pin memory for GPU transfer (future feature)")
   (current-index :initform 0
                  :accessor current-index
                  :documentation "Current position in dataset")
   (indices :initform nil
            :accessor loader-indices
            :documentation "Shuffled indices for current epoch")
   (prefetch-buffer :initform nil
                    :accessor prefetch-buffer
                    :documentation "Prefetched batches for performance"))
  (:documentation "Data loader for batching and iterating over datasets"))

(defun make-data-loader (dataset &key (batch-size 32) (shuffle nil) 
                                     (drop-last nil) (collate-fn nil)
                                     (sampler nil) (num-workers 0)
                                     (pin-memory nil))
  "Create a data loader for the given dataset
   
   Arguments:
   - dataset: Dataset instance
   - batch-size: Number of samples per batch (default: 32)
   - shuffle: Whether to shuffle data (default: nil)
   - drop-last: Drop incomplete last batch (default: nil)
   - collate-fn: Custom collation function (default: default-collate)
   - sampler: Custom sampler for sampling strategy (default: nil)
   - num-workers: Number of worker threads for parallel loading (default: 0)
   - pin-memory: Pin memory for GPU transfer (default: nil)
   
   Example:
   (let* ((data '((1 2 3) (4 5 6) (7 8 9)))
          (labels '(0 1 0))
          (ds (make-instance 'dataset :data data :labels labels))
          (loader (make-data-loader ds :batch-size 2 :shuffle t)))
     (loop for batch = (get-batch loader)
           while batch
           do (process-batch batch)))"
  (let ((loader (make-instance 'data-loader
                              :dataset dataset
                              :batch-size batch-size
                              :shuffle shuffle
                              :drop-last drop-last
                              :sampler sampler
                              :num-workers num-workers
                              :pin-memory pin-memory)))
    (when collate-fn
      (setf (collate-fn loader) collate-fn))
    (reset-loader loader)
    loader))

(defun reset-loader (loader)
  "Reset loader to beginning, optionally shuffling or using sampler"
  (setf (current-index loader) 0)
  (let ((n (dataset-length (loader-dataset loader)))
        (sampler (loader-sampler loader)))
    (setf (loader-indices loader)
          (cond
            ;; Use custom sampler if provided
            (sampler (get-sampler-indices sampler n))
            ;; Shuffle if requested
            ((shuffle-p loader) (shuffle-indices n))
            ;; Sequential by default
            (t (loop for i from 0 below n collect i))))))

(defun shuffle-indices (n)
  "Create shuffled list of indices from 0 to n-1 using Fisher-Yates shuffle"
  (let ((indices (loop for i from 0 below n collect i)))
    (loop for i from (1- n) downto 1
          do (let ((j (random (1+ i))))
               (rotatef (nth i indices) (nth j indices))))
    indices))

(defun num-batches (loader)
  "Calculate number of batches in the loader"
  (let* ((dataset-len (dataset-length (loader-dataset loader)))
         (batch-size (batch-size loader)))
    (if (drop-last-p loader)
        (floor dataset-len batch-size)
        (ceiling dataset-len batch-size))))

(defun get-batch (loader)
  "Get next batch from loader, returns NIL when exhausted
   
   Returns: (values batch-data batch-labels) or NIL"
  (with-slots (dataset batch-size current-index indices collate-fn drop-last) loader
    (let ((dataset-len (dataset-length dataset)))
      (when (>= current-index dataset-len)
        (return-from get-batch nil))
      
      (let* ((end-index (min (+ current-index batch-size) dataset-len))
             (batch-indices (subseq indices current-index end-index))
             (batch-size-actual (length batch-indices)))
        
        ;; Check if we should drop this batch
        (when (and drop-last (< batch-size-actual batch-size))
          (return-from get-batch nil))
        
        ;; Collect samples
        (let ((samples nil)
              (labels nil))
          (dolist (idx batch-indices)
            (multiple-value-bind (item label)
                (get-item dataset idx)
              (push item samples)
              (when label
                (push labels label))))
          
          (setf samples (nreverse samples))
          (setf labels (when labels (nreverse labels)))
          
          ;; Update index
          (incf current-index batch-size-actual)
          
          ;; Collate into batch
          (if labels
              (values (funcall collate-fn samples labels t)
                      (funcall collate-fn labels nil nil))
              (funcall collate-fn samples nil nil)))))))

;;;; ============================================================================
;;;; Collation Functions
;;;; ============================================================================

(defun default-collate (items labels-p has-labels)
  "Default collation: stack items into batched tensor
   
   For lists of numbers: creates (batch-size,) tensor
   For lists of lists: creates (batch-size, seq-len) tensor
   For tensors: stacks along batch dimension"
  (declare (ignore has-labels))
  
  (cond
    ;; Empty batch
    ((null items) nil)
    
    ;; Single numbers -> 1D tensor
    ((every #'numberp items)
     (let ((data (make-array (length items) :element-type 'double-float)))
       (loop for item in items
             for i from 0
             do (setf (aref data i) (coerce item 'double-float)))
       (make-tensor data :shape (list (length items)))))
    
    ;; Lists of lists -> 2D tensor (sequences)
    ((and (every #'listp items) (not (null (first items))))
     (collate-sequences items labels-p))
    
    ;; Lists of tensors -> stack tensors
    ((every #'(lambda (x) (typep x 'tensor)) items)
     (stack-tensors items))
    
    ;; Default: return as list
    (t items)))

(defun collate-sequences (sequences pad-value-p)
  "Collate variable-length sequences into padded batch
   
   Returns: (batch-size, max-seq-len) tensor"
  (let* ((batch-size (length sequences))
         (max-len (reduce #'max sequences :key #'length))
         (pad-value (if pad-value-p 0.0d0 0.0d0))
         (data (make-array (list batch-size max-len) 
                          :element-type 'double-float
                          :initial-element pad-value)))
    (loop for seq in sequences
          for i from 0
          do (loop for val in seq
                   for j from 0
                   do (setf (aref data i j) (coerce val 'double-float))))
    (make-tensor data :shape (list batch-size max-len))))

(defun stack-tensors (tensors)
  "Stack list of tensors along new batch dimension"
  (let* ((batch-size (length tensors))
         (first-shape (tensor-shape (first tensors)))
         (result-shape (cons batch-size first-shape))
         (data (make-array result-shape :element-type 'double-float)))
    
    ;; Copy each tensor into the batch
    (loop for tensor in tensors
          for b from 0
          do (let ((tensor-data (neural-network::tensor-data tensor)))
               (dotimes (i (array-total-size tensor-data))
                 (setf (row-major-aref data (+ (* b (array-total-size tensor-data)) i))
                       (row-major-aref tensor-data i)))))
    
    (make-tensor data :shape result-shape)))

;;;; ============================================================================
;;;; Utility Functions
;;;; ============================================================================

(defun pad-sequences (sequences &key (max-length nil) (pad-value 0))
  "Pad sequences to same length
   
   Arguments:
   - sequences: List of sequences (lists or arrays)
   - max-length: Target length (default: max length in batch)
   - pad-value: Value to use for padding (default: 0)
   
   Returns: List of padded sequences"
  (let ((target-length (or max-length
                          (reduce #'max sequences :key #'length))))
    (mapcar (lambda (seq)
              (let ((padded (make-list target-length :initial-element pad-value)))
                (loop for item in seq
                      for i from 0
                      do (setf (nth i padded) item))
                padded))
            sequences)))

(defun create-attention-mask (lengths max-length)
  "Create attention mask from sequence lengths
   
   Returns: (batch-size, max-length) tensor with 1 for valid positions"
  (let* ((batch-size (length lengths))
         (mask (make-array (list batch-size max-length)
                          :element-type 'double-float
                          :initial-element 0.0d0)))
    (loop for len in lengths
          for i from 0
          do (dotimes (j len)
               (setf (aref mask i j) 1.0d0)))
    (make-tensor mask :shape (list batch-size max-length))))

(defun normalize-batch (batch &key (mean 0.0d0) (std 1.0d0))
  "Normalize batch: (x - mean) / std
   
   Arguments:
   - batch: Tensor to normalize
   - mean: Mean value (default: 0.0)
   - std: Standard deviation (default: 1.0)
   
   Returns: Normalized tensor"
  (let* ((data (neural-network::tensor-data batch))
         (shape (tensor-shape batch))
         (result (make-array shape :element-type 'double-float)))
    (dotimes (i (array-total-size data))
      (setf (row-major-aref result i)
            (/ (- (row-major-aref data i) mean) std)))
    (make-tensor result :shape shape)))

(defun shuffle-dataset (dataset)
  "Shuffle dataset in-place (modifies data and labels)"
  (let* ((n (dataset-length dataset))
         (indices (shuffle-indices n))
         (data (dataset-data dataset))
         (labels (dataset-labels dataset)))
    (setf (dataset-data dataset)
          (mapcar (lambda (i) (elt data i)) indices))
    (when labels
      (setf (dataset-labels dataset)
            (mapcar (lambda (i) (elt labels i)) indices))))
  dataset)

;;;; ============================================================================
;;;; Format Loaders - Load datasets from standard formats
;;;; ============================================================================

(defun load-csv-dataset (filepath &key (delimiter #\,) (header t) 
                                      (label-column nil) (skip-rows 0))
  "Load dataset from CSV file
   
   Arguments:
   - filepath: Path to CSV file
   - delimiter: Column delimiter (default: comma)
   - header: Whether first row is header (default: t)
   - label-column: Name or index of label column (default: nil)
   - skip-rows: Number of rows to skip (default: 0)
   
   Returns: dataset instance"
  (with-open-file (stream filepath :direction :input)
    ;; Skip initial rows
    (dotimes (i skip-rows)
      (read-line stream nil nil))
    
    ;; Read header if present
    (let ((headers (when header
                    (parse-csv-line (read-line stream) delimiter)))
          (data '())
          (labels '()))
      
      ;; Determine label column index
      (let ((label-idx (cond
                         ((null label-column) nil)
                         ((numberp label-column) label-column)
                         ((stringp label-column)
                          (position label-column headers :test #'string=))
                         (t nil))))
        
        ;; Read data rows
        (loop for line = (read-line stream nil nil)
              while line
              do (let ((values (parse-csv-line line delimiter)))
                   (when label-idx
                     (push (parse-number (nth label-idx values)) labels)
                     (setf values (remove-nth label-idx values)))
                   (push (mapcar #'parse-number values) data)))
        
        (make-instance 'csv-dataset
                      :filepath filepath
                      :delimiter delimiter
                      :header header
                      :data (nreverse data)
                      :labels (when labels (nreverse labels))
                      :features (when headers
                                 (if label-idx
                                     (remove-nth label-idx headers)
                                     headers)))))))

(defun parse-csv-line (line delimiter)
  "Parse CSV line into list of strings"
  (let ((result '())
        (current "")
        (in-quotes nil))
    (loop for char across line
          do (cond
               ((and (char= char #\") (not in-quotes))
                (setf in-quotes t))
               ((and (char= char #\") in-quotes)
                (setf in-quotes nil))
               ((and (char= char delimiter) (not in-quotes))
                (push (string-trim '(#\Space #\Tab) current) result)
                (setf current ""))
               (t
                (setf current (concatenate 'string current (string char))))))
    (push (string-trim '(#\Space #\Tab) current) result)
    (nreverse result)))

(defun parse-number (str)
  "Parse string to number, return string if not a number"
  (let ((trimmed (string-trim '(#\Space #\Tab) str)))
    (handler-case
        (let ((num (read-from-string trimmed)))
          (if (numberp num) num trimmed))
      (error () trimmed))))

(defun remove-nth (n list)
  "Remove nth element from list"
  (append (subseq list 0 n) (subseq list (1+ n))))

(defun load-json-dataset (filepath &key (format :json) (label-key "label"))
  "Load dataset from JSON or JSONL file
   
   Arguments:
   - filepath: Path to JSON/JSONL file
   - format: :json (single object/array) or :jsonl (line-delimited)
   - label-key: Key for label in each object
   
   Returns: dataset instance"
  (with-open-file (stream filepath :direction :input)
    (let ((data '())
          (labels '()))
      (ecase format
        (:json
         ;; Read entire JSON file
         (let ((json-data (json-parse-stream stream)))
           (if (listp json-data)
               (dolist (item json-data)
                 (multiple-value-bind (item-data item-label)
                     (extract-label-from-dict item label-key)
                   (push item-data data)
                   (when item-label (push item-label labels))))
               (push json-data data))))
        (:jsonl
         ;; Read line-by-line JSON
         (loop for line = (read-line stream nil nil)
               while line
               do (let ((item (json-parse-string line)))
                    (multiple-value-bind (item-data item-label)
                        (extract-label-from-dict item label-key)
                      (push item-data data)
                      (when item-label (push item-label labels)))))))
      
      (make-instance 'json-dataset
                    :filepath filepath
                    :format format
                    :data (nreverse data)
                    :labels (when labels (nreverse labels))))))

(defun json-parse-stream (stream)
  "Simple JSON parser (placeholder - use proper JSON library in production)"
  ;; This is a simplified stub - integrate with cl-json or similar
  (read-from-string (with-output-to-string (s)
                     (loop for line = (read-line stream nil nil)
                           while line
                           do (write-line line s)))))

(defun json-parse-string (str)
  "Parse JSON string (placeholder)"
  (read-from-string str))

(defun extract-label-from-dict (dict label-key)
  "Extract label from dict/hash-table, return (values data label)"
  (cond
    ((hash-table-p dict)
     (let ((label (gethash label-key dict)))
       (remhash label-key dict)
       (values dict label)))
    ((listp dict)
     (let ((label (cdr (assoc label-key dict :test #'equal))))
       (values (remove label-key dict :key #'car :test #'equal) label)))
    (t (values dict nil))))

(defun load-huggingface-format (dirpath &key (split "train") (cache t))
  "Load dataset from HuggingFace Datasets format (Arrow/Parquet files)
   
   Arguments:
   - dirpath: Path to dataset directory
   - split: Split name to load (train/validation/test)
   - cache: Whether to cache processed items
   
   Returns: huggingface-dataset instance
   
   Note: This is a simplified loader. Full HF integration requires:
   - Apache Arrow bindings
   - Parquet file support
   - Dataset info JSON parsing"
  (let* ((split-path (merge-pathnames 
                      (make-pathname :directory (list :relative split))
                      (pathname dirpath)))
         (data '())
         (features nil))
    
    ;; Try to load dataset_info.json for metadata
    (let ((info-path (merge-pathnames "dataset_info.json" (pathname dirpath))))
      (when (probe-file info-path)
        (setf features (parse-dataset-info info-path))))
    
    ;; Load data files (simplified - would normally parse Arrow files)
    ;; This is a placeholder that looks for .jsonl or .csv fallbacks
    (let ((jsonl-path (merge-pathnames "data.jsonl" split-path))
          (csv-path (merge-pathnames "data.csv" split-path)))
      (cond
        ((probe-file jsonl-path)
         (let ((ds (load-json-dataset jsonl-path :format :jsonl)))
           (setf data (dataset-data ds))))
        ((probe-file csv-path)
         (let ((ds (load-csv-dataset csv-path)))
           (setf data (dataset-data ds))))
        (t
         (warn "No compatible data files found in ~A" split-path))))
    
    (make-instance 'huggingface-dataset
                  :data data
                  :split split
                  :features features
                  :cache (if cache (make-hash-table :test 'equal) nil))))

(defun parse-dataset-info (filepath)
  "Parse HuggingFace dataset_info.json"
  (with-open-file (stream filepath)
    (json-parse-stream stream)))

(defun load-pytorch-format (filepath &key (map-style t))
  "Load dataset from PyTorch-compatible format
   
   Arguments:
   - filepath: Path to serialized dataset
   - map-style: Whether dataset is map-style (vs iterable)
   
   Returns: pytorch-dataset instance"
  ;; Placeholder for PyTorch format loading
  ;; Would require pickle/torch file parsing
  (make-instance (if map-style 'map-style-dataset 'iterable-dataset)
                :filepath filepath))

(defun from-dict (dict &key (features nil))
  "Create dataset from dictionary/hash-table (HuggingFace-style)
   
   Arguments:
   - dict: Dictionary mapping column names to lists of values
   - features: Optional feature schema
   
   Example:
   (from-dict (alexandria:plist-hash-table 
               '(\"text\" (\"hello\" \"world\")
                 \"label\" (0 1))
               :test 'equal))"
  (let* ((keys (cond
                 ((hash-table-p dict) 
                  (loop for k being the hash-keys of dict collect k))
                 ((listp dict)
                  (mapcar #'car dict))
                 (t nil)))
         (first-key (first keys))
         (length (length (cond
                          ((hash-table-p dict) (gethash first-key dict))
                          ((listp dict) (cdr (assoc first-key dict)))
                          (t nil))))
         (data '()))
    
    ;; Convert column-wise dict to row-wise list
    (dotimes (i length)
      (let ((row (make-hash-table :test 'equal)))
        (dolist (key keys)
          (let ((values (cond
                         ((hash-table-p dict) (gethash key dict))
                         ((listp dict) (cdr (assoc key dict)))
                         (t nil))))
            (setf (gethash key row) (nth i values))))
        (push row data)))
    
    (make-instance 'huggingface-dataset
                  :data (nreverse data)
                  :features features
                  :columns keys
                  :format :dict)))

(defun from-list (list &key (features nil))
  "Create dataset from list of items
   
   Arguments:
   - list: List of data items
   - features: Optional feature schema
   
   Example:
   (from-list '((\"text\" \"hello\" \"label\" 0)
                (\"text\" \"world\" \"label\" 1)))"
  (make-instance 'dataset
                :data list
                :features features))

;;;; ============================================================================
;;;; Dataset Operations (HuggingFace/PyTorch-style)
;;;; ============================================================================

(defun map-dataset (dataset function &key (batched nil) (batch-size 1000))
  "Apply function to each item in dataset
   
   Arguments:
   - dataset: Dataset to transform
   - function: Function to apply (takes item, returns transformed item)
   - batched: Whether to process in batches
   - batch-size: Batch size for batched processing
   
   Returns: New dataset with transformed data"
  (let ((new-data (if batched
                      (map-batched dataset function batch-size)
                      (mapcar function (dataset-data dataset)))))
    (make-instance (class-of dataset)
                  :data new-data
                  :labels (dataset-labels dataset)
                  :features (dataset-features dataset)
                  :transform (dataset-transform dataset))))

(defun map-batched (dataset function batch-size)
  "Apply function to dataset in batches"
  (let ((data (dataset-data dataset))
        (result '()))
    (loop for i from 0 below (length data) by batch-size
          do (let ((batch (subseq data i (min (+ i batch-size) (length data)))))
               (setf result (append result (funcall function batch)))))
    result))

(defun filter-dataset (dataset predicate)
  "Filter dataset by predicate
   
   Arguments:
   - dataset: Dataset to filter
   - predicate: Function returning t for items to keep
   
   Returns: New dataset with filtered data"
  (let ((filtered-data '())
        (filtered-labels '()))
    (loop for item in (dataset-data dataset)
          for label in (or (dataset-labels dataset) (make-list (length (dataset-data dataset))))
          when (funcall predicate item)
          do (push item filtered-data)
             (when label (push label filtered-labels)))
    (make-instance (class-of dataset)
                  :data (nreverse filtered-data)
                  :labels (when (dataset-labels dataset) (nreverse filtered-labels))
                  :features (dataset-features dataset))))

(defun select-columns (dataset columns)
  "Select specific columns from dataset (for dict-style datasets)
   
   Arguments:
   - dataset: Dataset with dict-like items
   - columns: List of column names to keep
   
   Returns: New dataset with only selected columns"
  (let ((new-data (mapcar (lambda (item)
                           (if (hash-table-p item)
                               (let ((new-item (make-hash-table :test 'equal)))
                                 (dolist (col columns)
                                   (setf (gethash col new-item) (gethash col item)))
                                 new-item)
                               item))
                         (dataset-data dataset))))
    (make-instance (class-of dataset)
                  :data new-data
                  :features (dataset-features dataset))))

(defun rename-column (dataset old-name new-name)
  "Rename column in dataset
   
   Arguments:
   - dataset: Dataset to modify
   - old-name: Current column name
   - new-name: New column name
   
   Returns: New dataset with renamed column"
  (let ((new-data (mapcar (lambda (item)
                           (if (hash-table-p item)
                               (let ((new-item (make-hash-table :test 'equal)))
                                 (maphash (lambda (k v)
                                           (setf (gethash (if (equal k old-name) new-name k) 
                                                         new-item)
                                                 v))
                                         item)
                                 new-item)
                               item))
                         (dataset-data dataset))))
    (make-instance (class-of dataset)
                  :data new-data
                  :features (dataset-features dataset))))

(defun train-test-split (dataset &key (test-size 0.2) (shuffle t) (random-state nil))
  "Split dataset into train and test sets
   
   Arguments:
   - dataset: Dataset to split
   - test-size: Fraction for test set (default: 0.2)
   - shuffle: Whether to shuffle before splitting
   - random-state: Random seed (not implemented yet)
   
   Returns: (values train-dataset test-dataset)"
  (declare (ignore random-state))
  (let* ((data (dataset-data dataset))
         (labels (dataset-labels dataset))
         (n (length data))
         (indices (if shuffle (shuffle-indices n) (loop for i from 0 below n collect i)))
         (split-point (floor (* n (- 1 test-size))))
         (train-indices (subseq indices 0 split-point))
         (test-indices (subseq indices split-point)))
    
    (values
     (make-instance (class-of dataset)
                   :data (mapcar (lambda (i) (nth i data)) train-indices)
                   :labels (when labels (mapcar (lambda (i) (nth i labels)) train-indices))
                   :features (dataset-features dataset))
     (make-instance (class-of dataset)
                   :data (mapcar (lambda (i) (nth i data)) test-indices)
                   :labels (when labels (mapcar (lambda (i) (nth i labels)) test-indices))
                   :features (dataset-features dataset)))))

(defun concatenate-datasets (datasets)
  "Concatenate multiple datasets
   
   Arguments:
   - datasets: List of datasets to concatenate
   
   Returns: New dataset with combined data"
  (let ((all-data '())
        (all-labels '())
        (has-labels (dataset-labels (first datasets))))
    (dolist (ds datasets)
      (setf all-data (append all-data (dataset-data ds)))
      (when has-labels
        (setf all-labels (append all-labels (dataset-labels ds)))))
    (make-instance (class-of (first datasets))
                  :data all-data
                  :labels (when has-labels all-labels)
                  :features (dataset-features (first datasets)))))

(defun dataset-split (dataset split-name)
  "Get specific split from dataset (HuggingFace-style)
   
   This is a convenience method for datasets that contain multiple splits"
  (if (typep dataset 'huggingface-dataset)
      (if (equal (dataset-split-name dataset) split-name)
          dataset
          (error "Dataset split ~A not available" split-name))
      dataset))

;;;; ============================================================================
;;;; Additional Utilities
;;;; ============================================================================

(defun tokenize-batch (texts tokenizer &key (max-length nil) (padding t) (truncation t))
  "Tokenize batch of texts (HuggingFace-style)
   
   Arguments:
   - texts: List of text strings
   - tokenizer: Tokenization function
   - max-length: Maximum sequence length
   - padding: Whether to pad sequences
   - truncation: Whether to truncate long sequences
   
   Returns: Hash-table with 'input_ids' and 'attention_mask'"
  (let* ((tokenized (mapcar tokenizer texts))
         (lengths (mapcar #'length tokenized))
         (max-len (if max-length
                      max-length
                      (reduce #'max lengths)))
         (result (make-hash-table :test 'equal)))
    
    ;; Truncate if needed
    (when truncation
      (setf tokenized (mapcar (lambda (seq)
                               (if (> (length seq) max-len)
                                   (subseq seq 0 max-len)
                                   seq))
                             tokenized)))
    
    ;; Pad if needed
    (when padding
      (setf tokenized (pad-sequences tokenized :max-length max-len :pad-value 0)))
    
    ;; Create attention mask
    (setf (gethash "input_ids" result) tokenized)
    (setf (gethash "attention_mask" result)
          (mapcar (lambda (len)
                   (append (make-list (min len max-len) :initial-element 1)
                          (make-list (- max-len (min len max-len)) :initial-element 0)))
                 lengths))
    
    result))

(defun compute-dataset-statistics (dataset &key (columns nil))
  "Compute statistics for dataset (mean, std, min, max)
   
   Arguments:
   - dataset: Dataset to analyze
   - columns: Specific columns to analyze (for dict datasets)
   
   Returns: Hash-table with statistics"
  (let ((stats (make-hash-table :test 'equal))
        (data (dataset-data dataset)))
    
    (flet ((compute-stats (values)
             (let* ((n (length values))
                    (sum (reduce #'+ values))
                    (mean (/ sum n))
                    (variance (/ (reduce #'+ (mapcar (lambda (x) (expt (- x mean) 2)) values)) n))
                    (std (sqrt variance)))
               (list :mean mean
                     :std std
                     :min (reduce #'min values)
                     :max (reduce #'max values)
                     :count n))))
      
      (if columns
          ;; For dict-style datasets with columns
          (dolist (col columns)
            (let ((values (mapcar (lambda (item)
                                   (if (hash-table-p item)
                                       (gethash col item)
                                       (cdr (assoc col item))))
                                 data)))
              (when (every #'numberp values)
                (setf (gethash col stats) (compute-stats values)))))
          ;; For simple numeric datasets
          (when (every #'numberp data)
            (setf (gethash "data" stats) (compute-stats data)))))
    
    stats))

;;;; ============================================================================
;;;; ============================================================================
;;;; Example Usage and Documentation
;;;; ============================================================================

#|
EXAMPLES:

1. Basic Usage:
   (let* ((data '((1 2 3) (4 5 6) (7 8 9) (10 11 12)))
          (labels '(0 1 0 1))
          (ds (make-instance 'dataset :data data :labels labels))
          (loader (make-data-loader ds :batch-size 2)))
     (loop for batch = (get-batch loader)
           while batch
           do (print batch)))

2. Sequence Dataset with Padding:
   (let* ((sequences '((1 2 3) (4 5) (6 7 8 9)))
          (labels '(0 1 0))
          (ds (make-instance 'sequence-dataset 
                            :data sequences 
                            :labels labels
                            :max-length 10))
          (loader (make-data-loader ds :batch-size 2)))
     (loop for batch = (get-batch loader)
           while batch
           collect batch))

3. Custom Collation:
   (defun my-collate (items labels has-labels)
     ;; Custom batching logic
     (default-collate items labels has-labels))
   
   (let* ((ds (make-instance 'dataset :data data))
          (loader (make-data-loader ds 
                                   :batch-size 32
                                   :collate-fn #'my-collate)))
     (get-batch loader))

4. Training Loop:
   (defun train-epoch (model loader optimizer loss-fn)
     (reset-loader loader)
     (loop for (batch-x batch-y) = (multiple-value-list (get-batch loader))
           while batch-x
           do (let* ((output (forward model batch-x))
                     (loss (funcall loss-fn output batch-y)))
                (backward loss)
                (step optimizer))))

5. Loading from CSV (PyTorch/Pandas-style):
   (let* ((ds (load-csv-dataset "data/train.csv" 
                                :header t 
                                :label-column "target"))
          (loader (make-data-loader ds :batch-size 32 :shuffle t)))
     (get-batch loader))

6. Loading from JSON Lines (HuggingFace-style):
   (let ((ds (load-json-dataset "data/dataset.jsonl" 
                                :format :jsonl
                                :label-key "label")))
     (dataset-length ds))

7. Creating from Dictionary (HuggingFace Datasets API):
   (let ((ds (from-dict 
              (alexandria:plist-hash-table
               '("text" ("The cat sat" "Dogs are great" "Birds fly")
                 "label" (0 1 2))
               :test 'equal))))
     (loop for i from 0 below (dataset-length ds)
           collect (get-item ds i)))

8. Dataset Mapping and Filtering:
   (let* ((ds (from-list '(1 2 3 4 5 6 7 8 9 10)))
          ;; Filter even numbers
          (filtered (filter-dataset ds #'evenp))
          ;; Map to squares
          (squared (map-dataset filtered (lambda (x) (* x x)))))
     (dataset-data squared))
   ;; => (4 16 36 64 100)

9. Train/Test Split:
   (let ((ds (make-instance 'dataset :data data :labels labels)))
     (multiple-value-bind (train-ds test-ds)
         (train-test-split ds :test-size 0.2 :shuffle t)
       (format t "Train size: ~A, Test size: ~A~%"
               (dataset-length train-ds)
               (dataset-length test-ds))))

10. Using Custom Samplers:
    ;; Weighted sampling for imbalanced datasets
    (let* ((ds (make-instance 'dataset :data data :labels labels))
           (weights '(0.1 0.1 0.8))  ; Oversample rare classes
           (sampler (make-instance 'weighted-sampler
                                  :weights weights
                                  :num-samples 100))
           (loader (make-data-loader ds 
                                    :batch-size 10
                                    :sampler sampler)))
      (get-batch loader))

11. Distributed Training Sampler:
    (let* ((ds (make-instance 'dataset :data data))
           (sampler (make-instance 'distributed-sampler
                                  :num-replicas 4
                                  :rank 0
                                  :shuffle t))
           (loader (make-data-loader ds :sampler sampler)))
      ;; Each process gets different subset
      (get-batch loader))

12. HuggingFace-style Dataset Operations:
    (let* ((ds (from-dict (alexandria:plist-hash-table
                           '("text" ("hello" "world" "foo" "bar")
                             "label" (0 1 0 1)
                             "score" (0.5 0.8 0.3 0.9))
                           :test 'equal)))
           ;; Select specific columns
           (text-only (select-columns ds '("text" "label")))
           ;; Rename column
           (renamed (rename-column text-only "label" "target"))
           ;; Compute statistics
           (stats (compute-dataset-statistics ds :columns '("score"))))
      stats)

13. Text Tokenization (HuggingFace-style):
    (let* ((texts '("Hello world" "Common Lisp is great"))
           (tokenizer (lambda (text) 
                       (mapcar #'char-code (coerce text 'list))))
           (tokenized (tokenize-batch texts tokenizer
                                     :max-length 20
                                     :padding t
                                     :truncation t)))
      (gethash "input_ids" tokenized))

14. Caching for Expensive Transforms:
    (let* ((ds (make-instance 'dataset 
                             :data raw-data
                             :transform #'expensive-preprocessing))
           (loader (make-data-loader ds :batch-size 32)))
      ;; First access: processes and caches
      ;; Subsequent accesses: returns cached result
      (get-batch loader))

15. Concatenating Multiple Datasets:
    (let* ((ds1 (load-csv-dataset "train1.csv"))
           (ds2 (load-csv-dataset "train2.csv"))
           (ds3 (load-csv-dataset "train3.csv"))
           (combined (concatenate-datasets (list ds1 ds2 ds3)))
           (loader (make-data-loader combined :batch-size 64 :shuffle t)))
      (dataset-length combined))

INTERFACE GUARANTEES:

- dataset-length: Always returns integer >= 0
- get-item: Returns (values item label) or item if no labels
- get-item-by-key: Access specific features from dict-style datasets
- get-batch: Returns (values batch-data batch-labels) or NIL when exhausted
- Shapes: 
  * 1D data: (batch-size,)
  * Sequences: (batch-size, seq-len)
  * Images: (batch-size, channels, height, width)
  * Generic: (batch-size, ...) where ... matches item shape

COMPATIBILITY:

PyTorch Dataset API:
- __len__ -> dataset-length
- __getitem__ -> get-item
- DataLoader -> make-data-loader
- map, filter, collate_fn -> equivalent functions

HuggingFace Datasets API:
- Dataset.from_dict -> from-dict
- Dataset.from_list -> from-list
- Dataset.map -> map-dataset
- Dataset.filter -> filter-dataset
- Dataset.select_columns -> select-columns
- Dataset.rename_column -> rename-column
- Dataset.train_test_split -> train-test-split
- concatenate_datasets -> concatenate-datasets

Standard Formats:
- CSV files (with Pandas-like interface)
- JSON/JSONL files
- Arrow/Parquet files (placeholder for full implementation)
- Dictionary-based datasets
- List-based datasets

Samplers:
- SequentialSampler -> sequential-sampler
- RandomSampler -> random-sampler
- WeightedRandomSampler -> weighted-sampler
- DistributedSampler -> distributed-sampler

NOTES:

1. Caching: Dataset items are cached after first access for expensive transforms
2. Lazy Loading: Future versions will support lazy loading for large datasets
3. Arrow Support: Full Apache Arrow support requires additional bindings
4. JSON Parsing: Use cl-json or jonathan library for production JSON parsing
5. Parallel Loading: num-workers parameter reserved for future multi-threading
6. GPU Pinning: pin-memory parameter reserved for future GPU optimization
|#
