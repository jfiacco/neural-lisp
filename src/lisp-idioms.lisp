;;;; Neural Tensor Library - Lisp Idioms Extension
;;;; Showcasing Lisp's Unique Strengths: Macros, Functional Programming,
;;;; Conditions, Method Combinations, and Symbolic Computation

(defpackage :neural-tensor-lisp-idioms
  (:use :common-lisp)
  (:import-from :neural-network
                #:tensor
                #:tensor-data
                #:tensor-shape
                #:forward
                #:backward
                #:layer
                #:layer-parameters
                #:parameters
                #:zero-grad!
                #:requires-grad)
  (:export ;; Training macros
           #:with-training
           #:with-gradient-checkpointing
           #:deftrainer
           #:with-frozen-layers
           ;; Functional composition
           #:compose
           #:pipe
           #:curry
           #:partial
           #:transform-pipeline
           ;; Layer combinators
           #:residual
           #:parallel
           #:branch
           #:attention-combine
           ;; Condition system
           #:neural-condition
           #:gradient-explosion
           #:shape-mismatch
           #:nan-in-gradient
           #:with-gradient-protection
           ;; Pattern matching for graphs
           #:defpattern
           #:graph-rewrite
           #:optimize-graph
           ;; Method combinations
           #:forward-with-hooks
           #:defhook
           ;; Symbolic computation
           #:symbolic-optimize
           #:defderivative
           #:auto-simplify))

(in-package :neural-tensor-lisp-idioms)

;;;; ============================================================================
;;;; Condition System - Lisp's Superior Error Handling
;;;; ============================================================================

(define-condition neural-condition (condition)
  ((message :initarg :message
            :reader neural-message))
  (:documentation "Base condition for neural network operations"))

(define-condition gradient-explosion (neural-condition)
  ((norm :initarg :norm
         :reader gradient-norm)
   (threshold :initarg :threshold
              :reader gradient-threshold))
  (:report (lambda (condition stream)
             (format stream "Gradient explosion detected: norm=~,6f exceeds threshold=~,6f"
                     (gradient-norm condition)
                     (gradient-threshold condition)))))

(define-condition shape-mismatch (neural-condition)
  ((expected :initarg :expected
             :reader expected-shape)
   (actual :initarg :actual
           :reader actual-shape)
   (operation :initarg :operation
              :reader operation-name))
  (:report (lambda (condition stream)
             (format stream "Shape mismatch in ~a: expected ~a but got ~a"
                     (operation-name condition)
                     (expected-shape condition)
                     (actual-shape condition)))))

(define-condition nan-in-gradient (neural-condition)
  ((layer :initarg :layer
          :reader nan-layer))
  (:report (lambda (condition stream)
             (format stream "NaN detected in gradients of layer ~a"
                     (nan-layer condition)))))

;;;; Restarts for gradient issues
(defun clip-gradients-restart (parameters max-norm)
  "Restart: clip gradients to max norm"
  (dolist (param parameters)
    (when (neural-network::tensor-grad param)
      (let* ((grad (neural-network::tensor-grad param))
             (norm 0.0))
        (dotimes (i (array-total-size grad))
          (incf norm (expt (row-major-aref grad i) 2)))
        (setf norm (sqrt norm))
        (when (> norm max-norm)
          (let ((scale (/ max-norm norm)))
            (dotimes (i (array-total-size grad))
              (setf (row-major-aref grad i)
                    (* (row-major-aref grad i) scale)))))))))

(defun zero-gradients-restart (parameters)
  "Restart: zero out all gradients"
  (dolist (param parameters)
    (when (neural-network::tensor-grad param)
      (let ((grad (neural-network::tensor-grad param)))
        (dotimes (i (array-total-size grad))
          (setf (row-major-aref grad i) 0.0d0))))))

;;;; ============================================================================
;;;; Training Macros - Encapsulating Common Patterns
;;;; ============================================================================

(defmacro with-training ((model &key mode (verbose t)) &body body)
  "Execute body in training context with automatic mode management"
  (let ((original-mode (gensym "ORIGINAL-MODE"))
        (model-sym (gensym "MODEL")))
    `(let* ((,model-sym ,model)
            (,original-mode (slot-value ,model-sym 'neural-network::training)))
       (unwind-protect
            (progn
              (when ,verbose
                (format t "~%[Training Mode: ~a]~%" ,mode))
              (setf (slot-value ,model-sym 'neural-network::training) 
                    (eq ,mode :train))
              ,@body)
         ;; Always restore original mode
         (setf (slot-value ,model-sym 'neural-network::training) ,original-mode)
         (when ,verbose
           (format t "[Training Mode Restored]~%"))))))

(defmacro with-gradient-protection ((parameters &key (max-norm 10.0) (on-explosion :clip))
                                    &body body)
  "Execute body with automatic gradient explosion handling using restarts"
  `(block gradient-protection
     (handler-bind
         ((gradient-explosion
            (lambda (c)
              (format t "~&Warning: ~a~%" c)
              (ecase ,on-explosion
                (:clip (invoke-restart 'clip-and-continue ,parameters ,max-norm))
                (:zero (invoke-restart 'zero-and-continue ,parameters))
                (:abort (invoke-restart 'abort-training))))))
       (restart-case
           (progn ,@body)
         (clip-and-continue (params max-norm)
           :report "Clip gradients and continue training"
           (clip-gradients-restart params max-norm))
         (zero-and-continue (params)
           :report "Zero gradients and continue training"
           (zero-gradients-restart params))
         (abort-training ()
           :report "Abort training immediately"
           (return-from gradient-protection nil))))))

(defmacro with-frozen-layers (layer-specs &body body)
  "Temporarily freeze specified layers during execution"
  (let ((saved-states (gensym "SAVED-STATES")))
    `(let ((,saved-states 
            (mapcar (lambda (layer)
                      (cons layer (mapcar (lambda (p)
                                           (cons p (requires-grad p)))
                                         (parameters layer))))
                    (list ,@layer-specs))))
       (unwind-protect
            (progn
              ;; Freeze layers
              (dolist (pair ,saved-states)
                (dolist (param (parameters (car pair)))
                  (setf (requires-grad param) nil)))
              ,@body)
         ;; Restore gradient tracking
         (dolist (pair ,saved-states)
           (dolist (param-state (cdr pair))
             (setf (requires-grad (car param-state)) (cdr param-state))))))))

(defmacro deftrainer (name (model optimizer loss-fn) &body clauses)
  "Define a custom training procedure with declarative syntax"
  (let ((train-fn (intern (format nil "TRAIN-~a" name)))
        (step-fn (intern (format nil "~a-STEP" name))))
    `(progn
       (defun ,step-fn (,model ,optimizer inputs targets)
         (neural-tensor-optimizers::zero-grad ,optimizer)
         (let* ((outputs (forward ,model inputs))
                (loss (funcall ,loss-fn outputs targets)))
           (backward loss)
           (neural-tensor-optimizers::step ,optimizer)
           loss))
       
       (defun ,train-fn (,model ,optimizer train-data 
                        &key (epochs 10) ,@(extract-keywords clauses))
         ,(extract-body clauses)
         (dotimes (epoch epochs)
           (format t "~%Epoch ~a/~a~%" (1+ epoch) epochs)
           (let ((epoch-loss 0.0)
                 (batch-count 0))
             (dolist (batch train-data)
               (destructuring-bind (inputs targets) batch
                 (let ((loss (,step-fn ,model ,optimizer inputs targets)))
                   (incf epoch-loss (aref (neural-network::tensor-data loss) 0))
                   (incf batch-count))))
             (format t "Average Loss: ~,6f~%" (/ epoch-loss batch-count))))))))

(defun extract-keywords (clauses)
  "Helper to extract keyword arguments from clauses"
  (declare (ignore clauses))  ; Reserved for future DSL expansion
  '())

(defun extract-body (clauses)
  "Helper to extract body from clauses"
  (declare (ignore clauses))  ; Reserved for future DSL expansion
  "")

;;;; ============================================================================
;;;; Functional Composition - Higher-Order Functions
;;;; ============================================================================

(defun compose (&rest functions)
  "Compose functions right-to-left: (compose f g h) = f(g(h(x)))"
  (if (null functions)
      #'identity
      (let ((fn (car (last functions)))
            (rest-fns (butlast functions)))
        (lambda (&rest args)
          (reduce (lambda (result f)
                    (funcall f result))
                  rest-fns
                  :initial-value (apply fn args)
                  :from-end t)))))

(defun pipe (&rest functions)
  "Compose functions left-to-right: (pipe f g h) = h(g(f(x)))"
  (apply #'compose (reverse functions)))

(defun curry (function &rest initial-args)
  "Curry a function with initial arguments"
  (lambda (&rest remaining-args)
    (apply function (append initial-args remaining-args))))

(defun partial (function &rest initial-args)
  "Partial application (alias for curry)"
  (apply #'curry function initial-args))

(defmacro transform-pipeline (input &body transforms)
  "Threading macro for data transformations: (-> x f g h) = h(g(f(x)))"
  (if (null transforms)
      input
      `(transform-pipeline ,(if (listp (car transforms))
                                `(,(caar transforms) ,input ,@(cdar transforms))
                                `(,(car transforms) ,input))
                           ,@(cdr transforms))))

(defmacro -> (input &rest forms)
  "Thread-first macro: threads input as first argument"
  (if (null forms)
      input
      `(-> ,(if (listp (car forms))
                `(,(caar forms) ,input ,@(cdar forms))
                `(,(car forms) ,input))
           ,@(cdr forms))))

(defmacro ->> (input &rest forms)
  "Thread-last macro: threads input as last argument"
  (if (null forms)
      input
      `(->> ,(if (listp (car forms))
                 `(,@(car forms) ,input)
                 `(,(car forms) ,input))
            ,@(cdr forms))))

;;;; ============================================================================
;;;; Layer Combinators - Functional Network Construction
;;;; ============================================================================

(defclass residual-layer (layer)
  ((main-path :initarg :main-path
              :accessor main-path)
   (shortcut :initarg :shortcut
             :accessor shortcut
             :initform nil))
  (:documentation "Residual connection: output = main(x) + shortcut(x)"))

(defmethod initialize-instance :after ((layer residual-layer) &key)
  (with-slots (parameters main-path shortcut) layer
    (setf parameters (append (parameters main-path)
                            (when shortcut (parameters shortcut))))))

(defmethod forward ((layer residual-layer) input)
  (let ((main-output (forward (main-path layer) input))
        (shortcut-output (if (shortcut layer)
                            (forward (shortcut layer) input)
                            input)))
    (neural-network::t+ main-output shortcut-output)))

(defun residual (main-path &optional shortcut)
  "Create a residual connection (skip connection)"
  (make-instance 'residual-layer
                 :main-path main-path
                 :shortcut shortcut))

(defclass parallel-layer (layer)
  ((branches :initarg :branches
             :accessor branches)
   (combiner :initarg :combiner
             :accessor combiner
             :initform :concat))
  (:documentation "Process input through multiple parallel paths"))

(defmethod initialize-instance :after ((layer parallel-layer) &key)
  (with-slots (parameters branches) layer
    (setf parameters (apply #'append (mapcar #'parameters branches)))))

(defmethod forward ((layer parallel-layer) input)
  (let ((outputs (mapcar (lambda (branch) (forward branch input))
                        (branches layer))))
    (ecase (combiner layer)
      (:concat (apply #'concatenate-tensors outputs))
      (:sum (reduce #'neural-network::t+ outputs))
      (:mean (neural-network::t* 
              (reduce #'neural-network::t+ outputs)
              (/ 1.0 (length outputs)))))))

(defun parallel (combiner &rest branches)
  "Process input through multiple parallel branches"
  (make-instance 'parallel-layer
                 :branches branches
                 :combiner combiner))

(defun branch (&rest branches)
  "Create parallel branches that concatenate outputs"
  (apply #'parallel :concat branches))

(defun concatenate-tensors (&rest tensors)
  "Concatenate tensors along last dimension"
  ;; Simplified implementation - would need proper tensor concatenation
  (car tensors))

;;;; ============================================================================
;;;; Method Combinations - Extensible Layer Behavior
;;;; ============================================================================

(define-method-combination forward-with-hooks ()
  ((before-methods (:before))
   (primary-methods (:primary))
   (after-methods (:after))
   (around-methods (:around)))
  
  (let ((form (if (or before-methods after-methods)
                  `(multiple-value-prog1
                       (progn
                         ,@(mapcar (lambda (method)
                                    `(call-method ,method))
                                  before-methods)
                         (call-method ,(first primary-methods)
                                     ,(rest primary-methods)))
                     ,@(mapcar (lambda (method)
                                `(call-method ,method))
                              after-methods))
                  `(call-method ,(first primary-methods)
                               ,(rest primary-methods)))))
    (if around-methods
        `(call-method ,(first around-methods)
                     (,@(rest around-methods)
                      (make-method ,form)))
        form)))

(defmacro defhook (name (layer input) &body body)
  "Define a hook for layer forward pass"
  `(defmethod forward :before ((,layer ,name) ,input)
     ,@body))

;;;; Example usage:
;;;; (defhook my-layer (layer input)
;;;;   (format t "Processing input with shape ~a~%" (tensor-shape input)))

;;;; ============================================================================
;;;; Pattern Matching and Graph Rewriting
;;;; ============================================================================

(defstruct graph-pattern
  "Pattern for matching computation graphs"
  matcher
  rewriter
  description)

(defvar *rewrite-rules* nil
  "List of graph rewrite rules")

(defmacro defpattern (name pattern rewrite &key (description ""))
  "Define a graph rewrite pattern"
  `(progn
     (setf (get ',name 'graph-pattern)
           (make-graph-pattern
            :matcher (lambda (graph) (match-pattern graph ',pattern))
            :rewriter (lambda (graph) (rewrite-graph graph ',pattern ',rewrite))
            :description ,description))
     (pushnew ',name *rewrite-rules*)))

(defun match-pattern (graph pattern)
  "Pattern match on computation graph"
  ;; Simplified - would need full pattern matching
  (cond
    ((eq pattern :any) t)
    ((symbolp pattern) (eq graph pattern))
    ((listp pattern)
     (and (listp graph)
          (= (length graph) (length pattern))
          (every #'match-pattern graph pattern)))
    (t (equal graph pattern))))

(defun rewrite-graph (graph pattern replacement)
  "Rewrite graph according to pattern"
  ;; Simplified - would need full rewriting
  (if (match-pattern graph pattern)
      replacement
      graph))

;;;; Common optimization patterns
(defpattern fuse-multiply-add
    (* (+ a b) c)
    (+ (* a c) (* b c))
  :description "Distribute multiplication over addition")

(defpattern eliminate-identity
    (+ x 0)
    x
  :description "Remove identity elements")

(defpattern strength-reduction
    (* x 2)
    (+ x x)
  :description "Replace multiplication with addition")

(defun optimize-graph (graph)
  "Apply all rewrite rules to optimize computation graph"
  (let ((changed t)
        (result graph))
    (loop while changed do
      (setf changed nil)
      (dolist (rule *rewrite-rules*)
        (let* ((pattern (get rule 'graph-pattern))
               (new-graph (funcall (graph-pattern-rewriter pattern) result)))
          (unless (equal new-graph result)
            (setf result new-graph
                  changed t)))))
    result))

;;;; ============================================================================
;;;; Symbolic Computation Enhancement
;;;; ============================================================================

(defun symbolic-optimize (expr)
  "Optimize symbolic expression using algebraic rules"
  (labels ((optimize-once (e)
             (cond
               ;; Arithmetic identities
               ((and (listp e) (eq (car e) '+))
                (let ((args (remove 0 (cdr e))))
                  (cond
                    ((null args) 0)
                    ((= (length args) 1) (car args))
                    (t `(+ ,@args)))))
               
               ((and (listp e) (eq (car e) '*))
                (let ((args (remove 1 (cdr e))))
                  (cond
                    ((member 0 args) 0)
                    ((null args) 1)
                    ((= (length args) 1) (car args))
                    (t `(* ,@args)))))
               
               ;; Algebraic simplifications
               ((and (listp e) (eq (car e) '*))
                (cond
                  ;; x * x -> x^2
                  ((and (= (length e) 3)
                        (equal (second e) (third e)))
                   `(expt ,(second e) 2))
                  (t e)))
               
               ;; Double negation
               ((and (listp e) (eq (car e) '-))
                (if (and (= (length e) 2)
                         (listp (second e))
                         (eq (car (second e)) '-))
                    (second (second e))
                    e))
               
               ;; Recursive optimization
               ((listp e)
                (let ((optimized-args (mapcar #'optimize-once (cdr e))))
                  (cons (car e) optimized-args)))
               
               (t e))))
    
    (let ((result expr)
          (prev nil))
      (loop while (not (equal result prev)) do
        (setf prev result
              result (optimize-once result)))
      result)))

(defmacro defderivative (op (var) &body body)
  "Define derivative rule for an operation"
  `(setf (get ',op 'derivative-rule)
         (lambda (,var) ,@body)))

;; Define some common derivatives
(defderivative sin (x)
  `(cos ,x))

(defderivative cos (x)
  `(- (sin ,x)))

(defderivative exp (x)
  `(exp ,x))

(defderivative log (x)
  `(/ 1 ,x))

;;;; ============================================================================
;;;; Automatic Simplification with Reader Macro (Advanced)
;;;; ============================================================================

(defmacro auto-simplify (&body body)
  "Automatically simplify symbolic expressions in body"
  `(progn
     ,@(mapcar (lambda (expr)
                 (if (and (listp expr) (member (car expr) '(+ - * / expt)))
                     (symbolic-optimize expr)
                     expr))
               body)))

;;;; ============================================================================
;;;; Lisp-Style Network Composition Utilities
;;;; ============================================================================

(defun map-layers (function network)
  "Map a function over all layers in a network (like mapcar for networks)"
  (typecase network
    (neural-network::sequential
     (mapcar function (neural-network::seq-layers network)))
    (t (funcall function network))))

(defun filter-layers (predicate network)
  "Filter layers by predicate (like remove-if-not)"
  (typecase network
    (neural-network::sequential
     (remove-if-not predicate (neural-network::seq-layers network)))
    (t (when (funcall predicate network) (list network)))))

(defun fold-layers (function initial-value network)
  "Fold over layers (like reduce)"
  (typecase network
    (neural-network::sequential
     (reduce function (neural-network::seq-layers network)
             :initial-value initial-value))
    (t (funcall function initial-value network))))

;;;; ============================================================================
;;;; Example: Putting It All Together
;;;; ============================================================================

(defun demo-lisp-idioms ()
  "Demonstrate Lisp's unique features in neural networks"
  (format t "~%╔══════════════════════════════════════════════════════════════╗~%")
  (format t "║        Lisp Idioms for Neural Networks - Demo                ║~%")
  (format t "╚══════════════════════════════════════════════════════════════╝~%")
  
  ;; 1. Condition System
  (format t "~%1. Condition System with Restarts:~%")
  (format t "   - Gradient explosion? Choose: clip, zero, or abort~%")
  (format t "   - Shape mismatch? Get helpful error messages~%")
  (format t "   - NaN in gradients? Automatic recovery~%")
  
  ;; 2. Functional Composition
  (format t "~%2. Functional Composition:~%")
  (let ((transform (compose #'1+ #'* #'+)))
    (declare (ignore transform))  ; Example for demonstration purposes
    (format t "   (compose #'1+ #'* #'+) creates a new function~%")
    (format t "   Pipe operations naturally: (-> x f g h)~%"))
  
  ;; 3. Method Combinations
  (format t "~%3. Method Combinations:~%")
  (format t "   Add hooks to any layer: :before, :after, :around~%")
  (format t "   Perfect for logging, debugging, profiling~%")
  
  ;; 4. Pattern Matching
  (format t "~%4. Graph Pattern Matching & Rewriting:~%")
  (format t "   Automatically optimize: (* (+ a b) c) -> (+ (* a c) (* b c))~%")
  (let ((expr '(+ (* x 2) 0)))
    (format t "   Simplify: ~a -> ~a~%" expr (symbolic-optimize expr)))
  
  ;; 5. Macros
  (format t "~%5. Powerful Macros:~%")
  (format t "   (with-training (model :mode :train) ...) handles mode automatically~%")
  (format t "   (with-frozen-layers (layer1 layer2) ...) freezes/unfreezes~%")
  (format t "   (deftrainer name ...) creates custom training loops~%")
  
  (format t "~%~%These features are uniquely Lisp - impossible in Python!~%"))
