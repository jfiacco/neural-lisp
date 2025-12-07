;;;; Neural Tensor Library V2 - Leveraging Lisp's Strengths
;;;; Using CLOS, Macros, Symbolic Computation, and Metaprogramming

(defpackage :neural-network
  (:use :common-lisp)
  (:export #:tensor
           #:tensor-data
           #:tensor-shape
           #:tensor-name
           #:tensor-grad
           #:tensor-requires-grad
           #:tensor-backward-fn
           #:grad-fn
           #:make-tensor
           #:zeros
           #:ones
           #:randn
           #:requires-grad
           ;; Operators that work with CLOS
           #:t+
           #:t-
           #:t*
           #:t@
           #:transpose
           #:tsum
           #:tmean
           ;; Autograd
           #:with-grad
           #:backward
           #:zero-grad!
           ;; Neural network DSL
           #:defnetwork
           #:deflayer
           #:=>
           ;; Symbolic differentiation
           #:symbolic-grad
           #:simplify
           #:compile-graph
           #:graph->sexp
           ;; Checkpointing
           #:checkpoint-parameters
           #:save-checkpoint
           #:load-checkpoint
           ;; Layer abstractions
           #:layer
           #:linear
           #:linear-layer
           #:layer-weight
           #:layer-bias
           #:layer-training
           #:layer-parameters
           #:weights
           #:bias
           #:sequential
           #:seq-layers
           #:forward
           #:parameters
           #:train-mode
           #:eval-mode
           ;; Internal utilities for other modules
           #:children))

(in-package :neural-network)

;;;; ============================================================================
;;;; CLOS-based Tensor System
;;;; ============================================================================

(defclass tensor ()
  ((data :initarg :data
         :accessor tensor-data
         :type array)
   (shape :initarg :shape
          :accessor tensor-shape
          :type list)
   (grad :initarg :grad
         :accessor tensor-grad
         :initform nil)
   (requires-grad :initarg :requires-grad
                  :accessor requires-grad
                  :initform nil
                  :type boolean)
   (grad-fn :accessor grad-fn
            :initform nil)
   (children :accessor children
             :initform nil)
   (name :initarg :name
         :accessor tensor-name
         :initform nil))
  (:documentation "Tensor class with automatic differentiation"))

(defmethod print-object ((obj tensor) stream)
  "Pretty printing for tensors"
  (print-unreadable-object (obj stream :type t)
    (format stream "~@[~a ~]~a~@[ grad=~a~]"
            (tensor-name obj)
            (tensor-shape obj)
            (if (requires-grad obj) "enabled" nil))))

;;;; Constructor functions
(defun make-tensor (data &key requires-grad shape name)
  "Create a tensor - works with lists or arrays"
  (let* ((arr (if (arrayp data) 
                  ;; Convert existing array to double-float array
                  (let* ((dims (array-dimensions data))
                         (new-arr (make-array dims :element-type 'double-float)))
                    (if (null dims) ; 0-dimensional array
                        (let ((val (row-major-aref data 0)))
                          ;; Handle quirk where #0A(x) stores (x) instead of x
                          (setf (row-major-aref new-arr 0) 
                                (coerce (if (listp val) (first val) val) 'double-float)))
                        (dotimes (i (array-total-size data))
                          (setf (row-major-aref new-arr i)
                                (coerce (row-major-aref data i) 'double-float))))
                    new-arr)
                  ;; Create new double-float array from list
                  (let ((new-arr (make-array (or shape (list (length data)))
                                             :element-type 'double-float)))
                    (dotimes (i (length data))
                      (setf (aref new-arr i) (coerce (elt data i) 'double-float)))
                    new-arr)))
         (actual-shape (array-dimensions arr)))
    (make-instance 'tensor
                   :data arr
                   :shape actual-shape
                   :requires-grad requires-grad
                   :grad (when requires-grad
                           (make-array actual-shape 
                                      :element-type 'double-float
                                      :initial-element 0.0d0))
                   :name name)))

(defun zeros (shape &key requires-grad name)
  (make-tensor (make-array shape :element-type 'double-float :initial-element 0.0d0)
               :requires-grad requires-grad
               :shape shape
               :name name))

(defun ones (shape &key requires-grad name)
  (make-tensor (make-array shape :element-type 'double-float :initial-element 1.0d0)
               :requires-grad requires-grad
               :shape shape
               :name name))

(defun randn (shape &key requires-grad name (scale 1.0))
  "Random tensor with normal distribution (mean=0, std=scale)"
  (let ((arr (make-array shape :element-type 'double-float)))
    ;; Box-Muller transform for normal distribution
    (let ((size (array-total-size arr)))
      (loop for i from 0 below size by 2 do
        (let* ((u1 (loop for x = (random 1.0d0) while (= x 0.0d0) finally (return x)))
               (u2 (random 1.0d0))
               (mag (* scale (sqrt (* -2.0d0 (log u1)))))
               (z0 (* mag (cos (* 2.0d0 pi u2))))
               (z1 (* mag (sin (* 2.0d0 pi u2)))))
          (setf (row-major-aref arr i) (coerce z0 'double-float))
          (when (< (+ i 1) size)
            (setf (row-major-aref arr (+ i 1)) (coerce z1 'double-float))))))
    (make-tensor arr :requires-grad requires-grad :shape shape :name name)))

;;;; ============================================================================
;;;; Generic Functions for Operations (Operator Overloading via CLOS)
;;;; ============================================================================

(defgeneric t+ (a b)
  (:documentation "Generic addition - works on tensors and numbers"))

(defgeneric t- (a b)
  (:documentation "Generic subtraction"))

(defgeneric t* (a b)
  (:documentation "Generic element-wise multiplication"))

(defgeneric t@ (a b)
  (:documentation "Generic matrix multiplication"))

(defgeneric tsum (tensor)
  (:documentation "Sum all elements"))

(defgeneric tmean (tensor)
  (:documentation "Mean of all elements"))

(defgeneric backward (tensor &optional grad)
  (:documentation "Compute gradients via backpropagation"))

(defgeneric zero-grad! (tensor)
  (:documentation "Zero out gradients"))

;;;; Helper for elementwise operations
(defun elementwise-op (op t1 t2)
  (let* ((arr1 (tensor-data t1))
         (arr2 (tensor-data t2))
         (result (make-array (tensor-shape t1) :element-type 'double-float)))
    (dotimes (i (array-total-size arr1))
      (setf (row-major-aref result i)
            (funcall op
                     (row-major-aref arr1 i)
                     (row-major-aref arr2 i))))
    result))

;;;; Broadcasting helpers
(defun broadcast-shape (shape1 shape2)
  "Compute the broadcast shape for two shapes following NumPy rules"
  (let* ((len1 (length shape1))
         (len2 (length shape2))
         (max-len (max len1 len2))
         (padded1 (append (make-list (- max-len len1) :initial-element 1) shape1))
         (padded2 (append (make-list (- max-len len2) :initial-element 1) shape2))
         (result-shape nil))
    (dotimes (i max-len)
      (let ((d1 (nth i padded1))
            (d2 (nth i padded2)))
        (cond
          ((= d1 d2) (push d1 result-shape))
          ((= d1 1) (push d2 result-shape))
          ((= d2 1) (push d1 result-shape))
          (t (error "Shapes ~a and ~a are not broadcast compatible" shape1 shape2)))))
    (nreverse result-shape)))

(defun broadcast-index (index shape original-shape)
  "Map a broadcasted index to the original tensor index"
  (let* ((len (length shape))
         (orig-len (length original-shape))
         (offset (- len orig-len))
         (result nil))
    (dotimes (i orig-len)
      (let ((dim-idx (+ i offset))
            (orig-dim (nth i original-shape)))
        (push (if (= orig-dim 1) 0 (nth dim-idx index)) result)))
    (nreverse result)))

(defun multi-dim-index-to-subscripts (index shape)
  "Convert flat index to multi-dimensional subscripts"
  (let ((subscripts nil)
        (remaining index))
    (dolist (dim (reverse shape))
      (multiple-value-bind (quot rem) (floor remaining dim)
        (push rem subscripts)
        (setf remaining quot)))
    subscripts))

(defun subscripts-to-multi-dim-index (subscripts shape)
  "Convert multi-dimensional subscripts to flat index"
  (let ((index 0)
        (multiplier 1))
    (loop for sub in (reverse subscripts)
          for dim in (reverse shape)
          do (incf index (* sub multiplier))
             (setf multiplier (* multiplier dim)))
    index))

(defun broadcast-compatible-p (shape1 shape2)
  "Check if two shapes are broadcast compatible"
  (handler-case
      (progn (broadcast-shape shape1 shape2) t)
    (error () nil)))

(defun broadcast-add (t1 t2)
  "Add tensors with broadcasting support following NumPy rules"
  (let* ((shape1 (tensor-shape t1))
         (shape2 (tensor-shape t2))
         (arr1 (tensor-data t1))
         (arr2 (tensor-data t2)))
    (cond
      ;; Same shape - direct addition
      ((equal shape1 shape2)
       (elementwise-op #'+ t1 t2))
      
      ;; General broadcasting
      (t
       (let* ((result-shape (broadcast-shape shape1 shape2))
              (result (make-array result-shape :element-type 'double-float))
              (total-size (array-total-size result)))
         (dotimes (i total-size)
           (let* ((subscripts (multi-dim-index-to-subscripts i result-shape))
                  (idx1 (subscripts-to-multi-dim-index 
                         (broadcast-index subscripts result-shape shape1) shape1))
                  (idx2 (subscripts-to-multi-dim-index 
                         (broadcast-index subscripts result-shape shape2) shape2)))
             (setf (row-major-aref result i)
                   (+ (row-major-aref arr1 idx1)
                      (row-major-aref arr2 idx2)))))
         result)))))

;;;; Method implementations
(defmethod t+ ((t1 tensor) (t2 tensor))
  (let* ((shape1 (tensor-shape t1))
         (shape2 (tensor-shape t2))
         (result-data (broadcast-add t1 t2))
         (result-shape (array-dimensions result-data))
         (result (make-tensor result-data
                              :shape result-shape
                              :requires-grad (or (requires-grad t1)
                                                (requires-grad t2)))))
    (when (requires-grad result)
      (setf (grad-fn result)
            (lambda ()
              (when (requires-grad t1)
                (let ((g1 (tensor-grad t1))
                      (gr (tensor-grad result)))
                  (cond
                    ;; No broadcasting or t1 is larger
                    ((equal shape1 result-shape)
                     (dotimes (i (array-total-size g1))
                       (incf (row-major-aref g1 i)
                             (row-major-aref gr i))))
                    ;; t1 was broadcast from (1 N) to (M N): sum over first dimension
                    ((and (= (length shape1) 2) (= (first shape1) 1))
                     (let ((n (second shape1)))
                       (dotimes (j n)
                         (let ((sum 0.0d0))
                           (dotimes (i (first result-shape))
                             (incf sum (aref gr i j)))
                           (incf (aref g1 0 j) sum))))))))
              (when (requires-grad t2)
                (let ((g2 (tensor-grad t2))
                      (gr (tensor-grad result)))
                  ;; Handle broadcasting in gradient
                  (cond
                    ;; No broadcasting or t2 is larger
                    ((equal shape2 result-shape)
                     (dotimes (i (array-total-size g2))
                       (incf (row-major-aref g2 i)
                             (row-major-aref gr i))))
                    ;; t2 was broadcast from (1 N) to (M N): sum over first dimension
                    ((and (= (length shape2) 2) (= (first shape2) 1))
                     (let ((n (second shape2)))
                       (dotimes (j n)
                         (let ((sum 0.0d0))
                           (dotimes (i (first result-shape))
                             (incf sum (aref gr i j)))
                           (incf (aref g2 0 j) sum)))))
                    ;; Broadcast (N,) -> (M N): sum over first dimension
                    ((= (length shape2) 1)
                     (let ((n (first shape2)))
                       (dotimes (j n)
                         (let ((sum 0.0d0))
                           (dotimes (i (first result-shape))
                             (incf sum (aref gr i j)))
                           (incf (aref g2 j) sum))))))))))
      (setf (children result) (list t1 t2)))
    result))

(defmethod t- ((t1 tensor) (t2 tensor))
  (unless (equal (tensor-shape t1) (tensor-shape t2))
    (error "Shape mismatch: ~a vs ~a" (tensor-shape t1) (tensor-shape t2)))
  (let ((result (make-tensor (elementwise-op #'- t1 t2)
                             :shape (tensor-shape t1)
                             :requires-grad (or (requires-grad t1)
                                               (requires-grad t2)))))
    (when (requires-grad result)
      (setf (grad-fn result)
            (lambda ()
              (when (requires-grad t1)
                (let ((g1 (tensor-grad t1))
                      (gr (tensor-grad result)))
                  (dotimes (i (array-total-size g1))
                    (incf (row-major-aref g1 i)
                          (row-major-aref gr i)))))
              (when (requires-grad t2)
                (let ((g2 (tensor-grad t2))
                      (gr (tensor-grad result)))
                  (dotimes (i (array-total-size g2))
                    (decf (row-major-aref g2 i)
                          (row-major-aref gr i)))))))
      (setf (children result) (list t1 t2)))
    result))

(defun broadcast-mul (t1 t2)
  "Multiply tensors with broadcasting support following NumPy rules"
  (let* ((shape1 (tensor-shape t1))
         (shape2 (tensor-shape t2))
         (arr1 (tensor-data t1))
         (arr2 (tensor-data t2)))
    (cond
      ;; Same shape - direct multiplication
      ((equal shape1 shape2)
       (elementwise-op #'* t1 t2))
      
      ;; General broadcasting
      (t
       (let* ((result-shape (broadcast-shape shape1 shape2))
              (result (make-array result-shape :element-type 'double-float))
              (total-size (array-total-size result)))
         (dotimes (i total-size)
           (let* ((subscripts (multi-dim-index-to-subscripts i result-shape))
                  (idx1 (subscripts-to-multi-dim-index 
                         (broadcast-index subscripts result-shape shape1) shape1))
                  (idx2 (subscripts-to-multi-dim-index 
                         (broadcast-index subscripts result-shape shape2) shape2)))
             (setf (row-major-aref result i)
                   (* (row-major-aref arr1 idx1)
                      (row-major-aref arr2 idx2)))))
         result)))))

(defmethod t* ((t1 tensor) (t2 tensor))
  (let* ((shape1 (tensor-shape t1))
         (shape2 (tensor-shape t2))
         (result-data (broadcast-mul t1 t2))
         (result-shape (array-dimensions result-data))
         (result (make-tensor result-data
                              :shape result-shape
                              :requires-grad (or (requires-grad t1)
                                                (requires-grad t2)))))
    (when (requires-grad result)
      (setf (grad-fn result)
            (lambda ()
              (when (requires-grad t1)
                (let ((g1 (tensor-grad t1))
                      (gr (tensor-grad result))
                      (d2 (tensor-data t2)))
                  (if (equal shape1 shape2)
                      (dotimes (i (array-total-size g1))
                        (incf (row-major-aref g1 i)
                              (* (row-major-aref gr i)
                                 (row-major-aref d2 i))))
                      ;; Broadcast case - multiply by scalar
                      (when (equal shape2 '(1))
                        (let ((scalar (row-major-aref d2 0)))
                          (dotimes (i (array-total-size g1))
                            (incf (row-major-aref g1 i)
                                  (* (row-major-aref gr i) scalar))))))))
              (when (requires-grad t2)
                (let ((g2 (tensor-grad t2))
                      (gr (tensor-grad result))
                      (d1 (tensor-data t1)))
                  (if (equal shape1 shape2)
                      (dotimes (i (array-total-size g2))
                        (incf (row-major-aref g2 i)
                              (* (row-major-aref gr i)
                                 (row-major-aref d1 i))))
                      ;; Broadcast case - sum gradients for scalar
                      (when (equal shape2 '(1))
                        (let ((sum 0.0))
                          (dotimes (i (array-total-size d1))
                            (incf sum (* (row-major-aref gr i)
                                        (row-major-aref d1 i))))
                          (incf (row-major-aref g2 0) sum))))))))
      (setf (children result) (list t1 t2)))
    result))

(defmethod t@ ((t1 tensor) (t2 tensor))
  "Matrix multiplication with support for batch operations"
  (let ((shape1 (tensor-shape t1))
        (shape2 (tensor-shape t2)))
    (cond
      ;; 2D @ 2D: standard matrix multiplication
      ((and (= (length shape1) 2) (= (length shape2) 2))
       (unless (= (second shape1) (first shape2))
         (error "Incompatible shapes: ~a @ ~a" shape1 shape2))
       (let* ((m (first shape1))
              (n (second shape2))
              (k (second shape1))
              (result-data (make-array (list m n) :element-type 'double-float :initial-element 0.0d0))
              (arr1 (tensor-data t1))
              (arr2 (tensor-data t2)))
         (dotimes (i m)
           (dotimes (j n)
             (dotimes (p k)
               (incf (aref result-data i j)
                     (* (aref arr1 i p) (aref arr2 p j))))))
         (let ((result (make-tensor result-data
                                    :shape (list m n)
                                    :requires-grad (or (requires-grad t1)
                                                      (requires-grad t2)))))
           (when (requires-grad result)
             (setf (grad-fn result)
                   (lambda ()
                     (when (requires-grad t1)
                       (let ((g1 (tensor-grad t1))
                             (gr (tensor-grad result))
                             (d2 (tensor-data t2)))
                         (dotimes (i m)
                           (dotimes (p k)
                             (dotimes (j n)
                               (incf (aref g1 i p)
                                     (* (aref gr i j) (aref d2 p j))))))))
                     (when (requires-grad t2)
                       (let ((g2 (tensor-grad t2))
                             (gr (tensor-grad result))
                             (d1 (tensor-data t1)))
                         (dotimes (p k)
                           (dotimes (j n)
                             (dotimes (i m)
                               (incf (aref g2 p j)
                                     (* (aref d1 i p) (aref gr i j))))))))))
             (setf (children result) (list t1 t2)))
           result)))
      
      ;; 3D @ 3D: batch matrix multiplication
      ((and (= (length shape1) 3) (= (length shape2) 3))
       (destructuring-bind (batch1 m k1) shape1
         (destructuring-bind (batch2 k2 n) shape2
           (unless (= batch1 batch2)
             (error "Batch sizes must match: ~a vs ~a" batch1 batch2))
           (unless (= k1 k2)
             (error "Incompatible shapes: ~a @ ~a" shape1 shape2))
           (let* ((batch batch1)
                  (k k1)
                  (result-data (make-array (list batch m n) :element-type 'double-float :initial-element 0.0d0))
                  (arr1 (tensor-data t1))
                  (arr2 (tensor-data t2)))
             (dotimes (b batch)
               (dotimes (i m)
                 (dotimes (j n)
                   (dotimes (p k)
                     (incf (aref result-data b i j)
                           (* (aref arr1 b i p) (aref arr2 b p j)))))))
             (make-tensor result-data
                         :shape (list batch m n)
                         :requires-grad (or (requires-grad t1)
                                           (requires-grad t2)))))))
      
      ;; 3D @ 2D: broadcast 2D across batch dimension
      ((and (= (length shape1) 3) (= (length shape2) 2))
       (destructuring-bind (batch m k1) shape1
         (destructuring-bind (k2 n) shape2
           (unless (= k1 k2)
             (error "Incompatible shapes: ~a @ ~a" shape1 shape2))
           (let* ((k k1)
                  (result-data (make-array (list batch m n) :element-type 'double-float :initial-element 0.0d0))
                  (arr1 (tensor-data t1))
                  (arr2 (tensor-data t2)))
             (dotimes (b batch)
               (dotimes (i m)
                 (dotimes (j n)
                   (dotimes (p k)
                     (incf (aref result-data b i j)
                           (* (aref arr1 b i p) (aref arr2 p j)))))))
             (make-tensor result-data
                         :shape (list batch m n)
                         :requires-grad (or (requires-grad t1)
                                           (requires-grad t2)))))))
      
      ;; Unsupported combination
      (t
       (error "Unsupported tensor shapes for multiplication: ~a @ ~a" shape1 shape2)))))

(defun transpose (tensor)
  "Transpose a 2D tensor"
  (let* ((shape (tensor-shape tensor))
         (m (first shape))
         (n (second shape))
         (data (tensor-data tensor))
         (result-data (make-array (list n m) :element-type 'double-float)))
    (dotimes (i m)
      (dotimes (j n)
        (setf (aref result-data j i) (aref data i j))))
    (let ((result (make-tensor result-data
                               :shape (list n m)
                               :requires-grad (requires-grad tensor))))
      (when (requires-grad result)
        (setf (grad-fn result)
              (lambda ()
                (when (requires-grad tensor)
                  (let ((grad-out (tensor-grad result))
                        (grad-in (tensor-grad tensor)))
                    ;; Gradient of transpose is transpose of gradient
                    (dotimes (i m)
                      (dotimes (j n)
                        (incf (aref grad-in i j) (aref grad-out j i))))))))
        (setf (children result) (list tensor)))
      result)))

(defmethod tsum ((tensor tensor))
  (let* ((data (tensor-data tensor))
         (total 0.0d0))
    (dotimes (i (array-total-size data))
      (incf total (row-major-aref data i)))
    (let ((result (make-tensor (make-array '(1) :element-type 'double-float :initial-element total)
                               :shape '(1)
                               :requires-grad (requires-grad tensor))))
      (when (requires-grad result)
        (setf (grad-fn result)
              (lambda ()
                (when (requires-grad tensor)
                  (let ((grad (tensor-grad tensor))
                        (grad-val (aref (tensor-grad result) 0)))
                    (dotimes (i (array-total-size grad))
                      (incf (row-major-aref grad i) grad-val))))))
        (setf (children result) (list tensor)))
      result)))

(defmethod tmean ((tensor tensor))
  (let* ((data (tensor-data tensor))
         (n (array-total-size data))
         (total 0.0d0))
    (dotimes (i n)
      (incf total (row-major-aref data i)))
    (let ((result (make-tensor (make-array '(1) :element-type 'double-float :initial-element (/ total n))
                               :shape '(1)
                               :requires-grad (requires-grad tensor))))
      (when (requires-grad result)
        (setf (grad-fn result)
              (lambda ()
                (when (requires-grad tensor)
                  (let ((grad (tensor-grad tensor))
                        (grad-val (/ (aref (tensor-grad result) 0) n)))
                    (dotimes (i (array-total-size grad))
                      (incf (row-major-aref grad i) grad-val))))))
        (setf (children result) (list tensor)))
      result)))

(defmethod backward ((tensor tensor) &optional (grad 1.0d0))
  ;; If this tensor doesn't require grad, just return (no-op)
  (unless (requires-grad tensor)
    (return-from backward))
  
  ;; Initialize gradient
  (when (equal (tensor-shape tensor) '(1))
    (setf (aref (tensor-grad tensor) 0) (coerce grad 'double-float)))
  ;; Topological sort and backward
  (labels ((topo-sort (node visited order)
             (unless (member node visited)
               (push node visited)
               (dolist (child (children node))
                 (multiple-value-setq (visited order)
                   (topo-sort child visited order)))
               (push node order))
             (values visited order)))
    (let ((sorted (second (multiple-value-list (topo-sort tensor nil nil)))))
      (dolist (node sorted)
        (when (grad-fn node)
          (funcall (grad-fn node)))))))

(defmethod zero-grad! ((tensor tensor))
  (when (tensor-grad tensor)
    (dotimes (i (array-total-size (tensor-grad tensor)))
      (setf (row-major-aref (tensor-grad tensor) i) 0.0d0))))

;;;; ============================================================================
;;;; Macro Magic: Gradient Computation Context
;;;; ============================================================================

(defmacro with-grad (vars &body body)
  "Execute body with gradient tracking enabled for specified variables"
  `(progn
     ,@(loop for var in vars
             collect `(setf (requires-grad ,var) t
                           (tensor-grad ,var) 
                           (make-array (tensor-shape ,var) 
                                      :element-type 'double-float
                                      :initial-element 0.0d0)))
     ,@body))

;;;; ============================================================================
;;;; Layer Abstractions (Higher-Order Functions)
;;;; ============================================================================

(defclass layer ()
  ((parameters :initform nil
               :accessor layer-parameters)
   (training :initform t
             :accessor layer-training
             :documentation "Whether layer is in training mode"))
  (:documentation "Base class for neural network layers"))

(defgeneric forward (layer input)
  (:documentation "Forward pass through a layer"))

(defclass linear-layer (layer)
  ((weights :accessor weights)
   (bias :accessor bias)
   (in-features :initarg :in-features)
   (out-features :initarg :out-features)))

(defmethod initialize-instance :after ((layer linear-layer) &key)
  (with-slots (weights bias in-features out-features parameters) layer
    (setf weights (randn (list out-features in-features)
                        :requires-grad t
                        :name "weights"
                        :scale (/ 1.0 (sqrt in-features))))
    (setf bias (zeros (list out-features)
                     :requires-grad t
                     :name "bias"))
    (setf parameters (list weights bias))))

(defmethod forward ((layer linear-layer) input)
  (t+ (t@ input (transpose (weights layer)))
      (bias layer)))

(defun linear (in-features out-features)
  "Create a linear layer"
  (make-instance 'linear-layer
                 :in-features in-features
                 :out-features out-features))

;;;; Sequential container (function composition!)
(defclass sequential (layer)
  ((layers :initarg :layers
           :accessor seq-layers)))

(defmethod initialize-instance :after ((seq sequential) &key)
  (with-slots (layers parameters) seq
    (setf parameters
          (apply #'append
                 (mapcar #'layer-parameters layers)))))

(defmethod forward ((seq sequential) input)
  (reduce (lambda (x layer) (forward layer x))
          (seq-layers seq)
          :initial-value input))

(defun sequential (&rest layers)
  "Create a sequential model - function composition!"
  (make-instance 'sequential :layers layers))

;;;; ============================================================================
;;;; DSL for Neural Networks (The Real Lisp Magic!)
;;;; ============================================================================

(defmacro deflayer (name params &body body)
  "Define a custom layer type"
  `(progn
     (defclass ,name (layer)
       ,(loop for (param init) in params
              collect `(,param :initform ,init
                              :accessor ,param)))
     
     (defmethod forward ((layer ,name) input)
       (with-slots ,(mapcar #'first params) layer
         ,@body))))

(defmacro defnetwork (name layers &rest options)
  "Define a neural network architecture using a beautiful DSL"
  (declare (ignore options))  ; Options reserved for future use
  (let ((layer-list (gensym "LAYERS")))
    `(progn
       (defclass ,name (layer)
         ((network :accessor network-layers)))
       
       (defmethod initialize-instance :after ((net ,name) &key)
         (let ((,layer-list (list ,@layers)))
           (setf (network-layers net) 
                 (apply #'sequential ,layer-list))
           (setf (layer-parameters net)
                 (layer-parameters (network-layers net)))))
       
       (defmethod forward ((net ,name) input)
         (forward (network-layers net) input))
       
       (defun ,(intern (format nil "MAKE-~a" name)) ()
         (make-instance ',name)))))

;;;; ============================================================================
;;;; Helper Functions and Aliases
;;;; ============================================================================

(defun tensor-backward-fn (tensor)
  "Get the backward function of a tensor"
  (grad-fn tensor))

(defun layer-weight (layer)
  "Get the weight of a linear layer"
  (weights layer))

(defun layer-bias (layer)
  "Get the bias of a linear layer"
  (bias layer))

(defun parameters (layer)
  "Get parameters of a layer (alias for layer-parameters)"
  (layer-parameters layer))

(defun train-mode (layer)
  "Set layer to training mode"
  (setf (layer-training layer) t)
  ;; If it's a sequential, set all sublayers too
  (when (typep layer 'sequential)
    (dolist (sublayer (seq-layers layer))
      (train-mode sublayer)))
  layer)

(defun eval-mode (layer)
  "Set layer to evaluation mode"
  (setf (layer-training layer) nil)
  ;; If it's a sequential, set all sublayers too
  (when (typep layer 'sequential)
    (dolist (sublayer (seq-layers layer))
      (eval-mode sublayer)))
  layer)

;;;; ============================================================================
;;;; Checkpointing & Serialization
;;;; ============================================================================

(defparameter +checkpoint-format-id+ "neural-lisp-checkpoint"
  "Identifier stored in checkpoint files to verify compatibility.")

(defparameter +checkpoint-format-version+ 1
  "Current checkpoint format version.")

(defun %tensor-checkpoint-name (tensor)
  "Return the stable name for TENSOR if one exists, otherwise NIL."
  (let ((name (tensor-name tensor)))
    (cond
      ((stringp name) name)
      ((symbolp name) (symbol-name name))
      (name (format nil "~a" name))
      (t nil))))

(defun %tensor-display-name (tensor index)
  "Return a human-readable label for TENSOR, used in diagnostics."
  (or (%tensor-checkpoint-name tensor)
      (format nil "param-~d" index)))

(defun %collect-checkpoint-tensors (object)
  "Return a flat list of tensors contained in OBJECT.

OBJECT can be a tensor, a layer, a list of layers/tensors, or any nested
combination thereof."
  (cond
    ((null object) nil)
    ((typep object 'tensor) (list object))
    ((typep object 'layer) (remove nil (parameters object)))
    ((listp object)
     (loop for entry in object append (%collect-checkpoint-tensors entry)))
    ((vectorp object)
     (loop for entry across object append (%collect-checkpoint-tensors entry)))
    (t (error "Cannot extract tensors from object of type ~a" (type-of object)))))

(defun checkpoint-parameters (object)
  "Return all tensors that should be checkpointed for OBJECT.

Signals an error if no tensors can be located to prevent writing empty
checkpoints."
  (let ((tensors (%collect-checkpoint-tensors object)))
    (unless tensors
      (error "No tensors available for checkpointing in ~a" object))
    tensors))

(defun %tensor-data->list (array)
  "Convert ARRAY into a flat list of double floats."
  (loop for i from 0 below (array-total-size array)
        collect (row-major-aref array i)))

(defun %tensor->checkpoint-entry (tensor index include-grad)
  "Serialize TENSOR into a plist saved inside the checkpoint file."
  (let ((data (tensor-data tensor)))
    (list :name (%tensor-checkpoint-name tensor)
          :position index
          :shape (copy-list (tensor-shape tensor))
          :requires-grad (requires-grad tensor)
          :data (%tensor-data->list data)
          :grad (when (and include-grad (tensor-grad tensor))
                  (%tensor-data->list (tensor-grad tensor))))))

(defun %serialize-tensors (tensors include-grad)
  "Create checkpoint entries for each tensor."
  (loop for tensor in tensors
        for idx from 0
        collect (%tensor->checkpoint-entry tensor idx include-grad)))

(defun save-checkpoint (target path &key metadata include-grad)
  "Persist TARGET's tensors to PATH.

TARGET can be a tensor, a layer, or a (possibly nested) list/vector of such
objects. The checkpoint stores tensor data, shapes, requires-grad flags, and
optional gradient values. METADATA can be any Lisp object that will be saved in
the file for downstream consumers. Returns the pathname that was written."
  (let* ((tensors (checkpoint-parameters target))
         (entries (%serialize-tensors tensors include-grad))
         (payload (list :format +checkpoint-format-id+
                        :version +checkpoint-format-version+
                        :saved-at (get-universal-time)
                        :metadata metadata
                        :include-grad include-grad
                        :entries entries))
         (pathname (pathname path)))
    (with-open-file (stream pathname
                            :direction :output
                            :if-exists :supersede
                            :if-does-not-exist :create)
      (with-standard-io-syntax
        (let ((*print-circle* nil)
              (*print-pretty* t))
          (print payload stream))))
    pathname))

(defun %read-checkpoint (path)
  "Read PATH and return the raw checkpoint plist, ensuring format sanity."
  (let* ((pathname (pathname path))
         (payload (with-open-file (stream pathname :direction :input)
                    (with-standard-io-syntax
                      (read stream nil nil)))))
    (unless (and (listp payload)
                 (string= (getf payload :format) +checkpoint-format-id+))
      (error "~a is not a valid neural-lisp checkpoint" pathname))
    payload))

(defun %entries->table (entries)
  "Return a hash table of named ENTRIES (preserving order) and unnamed ones."
  (let ((table (make-hash-table :test 'equal))
        (unnamed '()))
    (dolist (entry entries)
      (let ((name (getf entry :name)))
        (if name
            (push entry (gethash name table))
            (push entry unnamed))))
    (maphash (lambda (key value)
               (declare (ignore key))
               (setf (gethash key table) (nreverse value)))
             table)
      (values table (nreverse unnamed))))

(defun %pop-named-entry (table name)
  "Pop the next entry associated with NAME from TABLE."
  (let ((queue (gethash name table)))
    (when queue
      (let ((entry (car queue))
            (rest (cdr queue)))
        (if rest
            (setf (gethash name table) rest)
            (remhash name table))
          entry))))

(defun %remaining-named-count (table)
  "Return the number of unused named entries left in TABLE."
  (let ((count 0))
    (maphash (lambda (key value)
               (declare (ignore key))
               (incf count (length value)))
             table)
    count))

(defun %ensure-sequence-length (seq expected what)
  (let ((len (length seq)))
    (unless (= len expected)
      (error "Checkpoint entry for ~a has ~d values but expected ~d"
             what len expected))))

(defun %restore-array-from-data (array data what)
  "Copy DATA into ARRAY, ensuring dimensional alignment."
  (let* ((flat (coerce data 'list))
         (size (array-total-size array)))
    (%ensure-sequence-length flat size what)
    (loop for value in flat
          for idx from 0
          do (setf (row-major-aref array idx)
                   (coerce value 'double-float)))
    array))

(defun %ensure-grad-buffer (tensor)
  (or (tensor-grad tensor)
      (setf (tensor-grad tensor)
            (make-array (tensor-shape tensor)
                        :element-type 'double-float
                        :initial-element 0.0d0))))

(defun %apply-checkpoint-entry (tensor entry)
  "Load ENTRY contents into TENSOR."
  (let* ((shape (getf entry :shape))
         (data (getf entry :data)))
    (unless (and shape data)
      (error "Malformed checkpoint entry: missing shape/data"))
    (unless (equal shape (tensor-shape tensor))
      (error "Shape mismatch when loading checkpoint.~%  Tensor: ~a~%  Entry: ~a"
             (tensor-shape tensor) shape))
    (%restore-array-from-data (tensor-data tensor) data (tensor-name tensor))
    (let ((requires-flag (getf entry :requires-grad :not-found)))
      (unless (eq requires-flag :not-found)
        (setf (requires-grad tensor) (not (null requires-flag)))))
    (let ((grad-data (getf entry :grad)))
      (cond
        (grad-data
         (%restore-array-from-data (%ensure-grad-buffer tensor)
                                   grad-data
                                   (format nil "grad(~a)" (tensor-name tensor))))
        ((requires-grad tensor)
         (%ensure-grad-buffer tensor))))
    tensor))

(defun load-checkpoint (target path &key (strict t))
  "Load tensor data from PATH into TARGET.

When STRICT is true, signals if there are missing or extra parameters compared
to the checkpoint. When false, missing tensors are ignored and surplus entries
are skipped. Returns TARGET for convenience."
  (let* ((payload (%read-checkpoint path))
         (entries (getf payload :entries))
         (tensors (checkpoint-parameters target)))
    (multiple-value-bind (named-table unnamed) (%entries->table entries)
      (loop for tensor in tensors
            for idx from 0
            for preferred-name = (%tensor-checkpoint-name tensor)
            for display-name = (%tensor-display-name tensor idx)
            for entry = (or (and preferred-name
                                 (%pop-named-entry named-table preferred-name))
                            (when unnamed
                              (let ((item (car unnamed)))
                                (setf unnamed (cdr unnamed))
                                item)))
            do (cond
                 (entry
                  (%apply-checkpoint-entry tensor entry))
                 (strict
                  (error "Missing checkpoint data for parameter ~a"
                         display-name))))
      (when (and strict
                 (or unnamed (> (%remaining-named-count named-table) 0)))
        (error "Checkpoint contains extra parameters not present in target"))
      target)))

;;;; ============================================================================
;;;; Symbolic Differentiation (Classic Lisp!)
;;;; ============================================================================

(defun symbolic-grad (expr var)
  "Symbolic differentiation - shows off Lisp's symbolic computation!"
  (cond
    ;; Constant
    ((numberp expr) 0)
    ;; Variable
    ((eq expr var) 1)
    ;; Different variable
    ((symbolp expr) 0)
    ;; Compound expression
    ((listp expr)
     (let ((op (first expr))
           (args (rest expr)))
       (case op
         (+ `(+ ,@(mapcar (lambda (arg) (symbolic-grad arg var)) args)))
         (- (if (= (length args) 1)
                `(- ,(symbolic-grad (first args) var))
                `(- ,(symbolic-grad (first args) var)
                    ,(symbolic-grad (second args) var))))
         (* (if (= (length args) 2)
                `(+ (* ,(first args) ,(symbolic-grad (second args) var))
                    (* ,(symbolic-grad (first args) var) ,(second args)))
                (error "Multiplication of ~a arguments not supported" (length args))))
         (/ `(/ (- (* ,(second args) ,(symbolic-grad (first args) var))
                   (* ,(first args) ,(symbolic-grad (second args) var)))
                (* ,(second args) ,(second args))))
         (exp `(* (exp ,@args) ,(symbolic-grad (first args) var)))
         (log `(/ ,(symbolic-grad (first args) var) ,@args))
         (expt (if (numberp (second args))
                   `(* ,(second args)
                       (expt ,(first args) ,(1- (second args)))
                       ,(symbolic-grad (first args) var))
                   (error "Power rule only for numeric exponents")))
         (otherwise (error "Unknown operator: ~a" op)))))))

(defun simplify (expr)
  "Basic symbolic simplification"
  (if (not (listp expr))
      expr
      (let ((op (first expr))
            (args (mapcar #'simplify (rest expr))))
        (case op
          (+ (cond
               ((every #'numberp args) (apply #'+ args))
               ((member 0 args) (simplify `(+ ,@(remove 0 args))))
               (t `(+ ,@args))))
          (* (cond
               ((every #'numberp args) (apply #'* args))
               ((member 0 args) 0)
               ((member 1 args) (simplify `(* ,@(remove 1 args))))
               (t `(* ,@args))))
          (otherwise `(,op ,@args))))))

;;;; ============================================================================
;;;; Code Generation: Compile Computational Graphs to Lisp Functions
;;;; ============================================================================

(defun compile-graph (operations)
  "Compile a sequence of tensor operations into an optimized Lisp function"
  (eval `(lambda (inputs)
           ,@operations)))

;;;; Export a function to visualize the computation graph as S-expressions
(defun graph->sexp (tensor)
  "Convert computational graph to S-expression"
  (if (null (children tensor))
      (or (tensor-name tensor) 'input)
      (list (type-of tensor)
            (mapcar #'graph->sexp (children tensor)))))

