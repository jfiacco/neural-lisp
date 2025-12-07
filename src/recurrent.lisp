;;;; Neural Tensor Library - Recurrent Neural Networks
;;;; RNN, LSTM, GRU, and Bidirectional variants
;;;; Showcasing Lisp's metaprogramming for sequence processing

(defpackage :neural-tensor-recurrent
  (:use :common-lisp)
  (:import-from :neural-network
                #:tensor
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
                #:transpose)
  (:import-from :neural-tensor-activations
                #:sigmoid
                #:tanh-activation
                #:relu)
  (:export #:rnn-cell
           #:lstm-cell
           #:gru-cell
           #:rnn-layer
           #:lstm-layer
           #:gru-layer
           #:bidirectional-rnn
           #:bidirectional-lstm
           #:bidirectional-gru
           #:stack-rnn
           #:attention-rnn
           #:reset-hidden-state
           #:get-hidden-state
           #:with-sequence-processing
           #:defrecurrent
           #:sequence-map
           #:sequence-fold))

(in-package :neural-tensor-recurrent)

;;;; ============================================================================
;;;; Base Recurrent Cell Protocol
;;;; ============================================================================

(defclass recurrent-cell (layer)
  ((input-size :initarg :input-size
               :reader input-size)
   (hidden-size :initarg :hidden-size
                :reader hidden-size)
   (hidden-state :initform nil
                 :accessor hidden-state
                 :documentation "Current hidden state"))
  (:documentation "Base class for recurrent cells"))

(defgeneric cell-forward (cell input hidden)
  (:documentation "Process single timestep"))

(defgeneric init-hidden (cell batch-size)
  (:documentation "Initialize hidden state"))

;;;; ============================================================================
;;;; Vanilla RNN Cell
;;;; ============================================================================

(defclass rnn-cell (recurrent-cell)
  ((w-ih :accessor w-ih :documentation "Input to hidden weights")
   (w-hh :accessor w-hh :documentation "Hidden to hidden weights")
   (b-ih :accessor b-ih :documentation "Input bias")
   (b-hh :accessor b-hh :documentation "Hidden bias")
   (activation :initarg :activation
               :initform :tanh
               :reader activation))
  (:documentation "Vanilla RNN cell: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)"))

(defmethod initialize-instance :after ((cell rnn-cell) &key)
  (with-slots (input-size hidden-size w-ih w-hh b-ih b-hh parameters) cell
    (let ((scale (/ 1.0 (sqrt hidden-size))))
      (setf w-ih (randn (list hidden-size input-size)
                       :requires-grad t
                       :name "w-ih"
                       :scale scale))
      (setf w-hh (randn (list hidden-size hidden-size)
                       :requires-grad t
                       :name "w-hh"
                       :scale scale))
      (setf b-ih (zeros (list hidden-size)
                       :requires-grad t
                       :name "b-ih"))
      (setf b-hh (zeros (list hidden-size)
                       :requires-grad t
                       :name "b-hh"))
      (setf parameters (list w-ih w-hh b-ih b-hh)))))

(defmethod cell-forward ((cell rnn-cell) input hidden)
  (with-slots (w-ih w-hh b-ih b-hh activation) cell
    (let* ((ih (t+ (t@ input (transpose w-ih)) b-ih))
           (hh (t+ (t@ hidden (transpose w-hh)) b-hh))
           (pre-activation (t+ ih hh)))
      (ecase activation
  (:tanh (tanh-activation pre-activation))
  (:relu (relu pre-activation))))))

(defmethod init-hidden ((cell rnn-cell) batch-size)
  (with-slots (hidden-size) cell
    (zeros (list batch-size hidden-size))))

;;;; ============================================================================
;;;; LSTM Cell (Long Short-Term Memory)
;;;; ============================================================================

(defclass lstm-cell (recurrent-cell)
  ((w-ii :accessor w-ii :documentation "Input gate - input weights")
   (w-hi :accessor w-hi :documentation "Input gate - hidden weights")
   (b-ii :accessor b-ii)
   (b-hi :accessor b-hi)
   (w-if :accessor w-if :documentation "Forget gate weights")
   (w-hf :accessor w-hf)
   (b-if :accessor b-if)
   (b-hf :accessor b-hf)
   (w-ig :accessor w-ig :documentation "Cell gate weights")
   (w-hg :accessor w-hg)
   (b-ig :accessor b-ig)
   (b-hg :accessor b-hg)
   (w-io :accessor w-io :documentation "Output gate weights")
   (w-ho :accessor w-ho)
   (b-io :accessor b-io)
   (b-ho :accessor b-ho)
   (cell-state :initform nil
               :accessor cell-state
               :documentation "LSTM cell state"))
  (:documentation "LSTM cell with forget, input, and output gates"))

(defmethod initialize-instance :after ((cell lstm-cell) &key)
  (with-slots (input-size hidden-size parameters
               w-ii w-hi b-ii b-hi
               w-if w-hf b-if b-hf
               w-ig w-hg b-ig b-hg
               w-io w-ho b-io b-ho) cell
    (let ((scale (/ 1.0 (sqrt hidden-size))))
      ;; Input gate
      (setf w-ii (randn (list hidden-size input-size) :requires-grad t :scale scale))
      (setf w-hi (randn (list hidden-size hidden-size) :requires-grad t :scale scale))
      (setf b-ii (zeros (list hidden-size) :requires-grad t))
      (setf b-hi (zeros (list hidden-size) :requires-grad t))
      ;; Forget gate
      (setf w-if (randn (list hidden-size input-size) :requires-grad t :scale scale))
      (setf w-hf (randn (list hidden-size hidden-size) :requires-grad t :scale scale))
      (setf b-if (ones (list hidden-size) :requires-grad t)) ; Bias to 1 for forget gate
      (setf b-hf (zeros (list hidden-size) :requires-grad t))
      ;; Cell gate
      (setf w-ig (randn (list hidden-size input-size) :requires-grad t :scale scale))
      (setf w-hg (randn (list hidden-size hidden-size) :requires-grad t :scale scale))
      (setf b-ig (zeros (list hidden-size) :requires-grad t))
      (setf b-hg (zeros (list hidden-size) :requires-grad t))
      ;; Output gate
      (setf w-io (randn (list hidden-size input-size) :requires-grad t :scale scale))
      (setf w-ho (randn (list hidden-size hidden-size) :requires-grad t :scale scale))
      (setf b-io (zeros (list hidden-size) :requires-grad t))
      (setf b-ho (zeros (list hidden-size) :requires-grad t))
      
      (setf parameters (list w-ii w-hi b-ii b-hi
                            w-if w-hf b-if b-hf
                            w-ig w-hg b-ig b-hg
                            w-io w-ho b-io b-ho)))))

(defmethod cell-forward ((cell lstm-cell) input hidden-and-cell)
  "LSTM forward pass
   hidden-and-cell is a list: (hidden-state cell-state)"
  (destructuring-bind (hidden cell-state) hidden-and-cell
    (with-slots (w-ii w-hi b-ii b-hi
                 w-if w-hf b-if b-hf
                 w-ig w-hg b-ig b-hg
                 w-io w-ho b-io b-ho) cell
      ;; Input gate: i_t = σ(W_ii * x_t + W_hi * h_{t-1} + b_i)
      (let* ((i-gate (sigmoid (t+ (t+ (t@ input (transpose w-ii)) b-ii)
                                  (t+ (t@ hidden (transpose w-hi)) b-hi))))
             ;; Forget gate: f_t = σ(W_if * x_t + W_hf * h_{t-1} + b_f)
             (f-gate (sigmoid (t+ (t+ (t@ input (transpose w-if)) b-if)
                                  (t+ (t@ hidden (transpose w-hf)) b-hf))))
             ;; Cell gate: g_t = tanh(W_ig * x_t + W_hg * h_{t-1} + b_g)
             (g-gate (tanh-activation (t+ (t+ (t@ input (transpose w-ig)) b-ig)
                                          (t+ (t@ hidden (transpose w-hg)) b-hg))))
             ;; Output gate: o_t = σ(W_io * x_t + W_ho * h_{t-1} + b_o)
             (o-gate (sigmoid (t+ (t+ (t@ input (transpose w-io)) b-io)
                                  (t+ (t@ hidden (transpose w-ho)) b-ho))))
             ;; New cell state: c_t = f_t * c_{t-1} + i_t * g_t
             (new-cell-state (t+ (t* f-gate cell-state) (t* i-gate g-gate)))
             ;; New hidden state: h_t = o_t * tanh(c_t)
             (new-hidden (t* o-gate (tanh-activation new-cell-state))))
        (list new-hidden new-cell-state)))))

(defmethod init-hidden ((cell lstm-cell) batch-size)
  (with-slots (hidden-size) cell
    (list (zeros (list batch-size hidden-size))
          (zeros (list batch-size hidden-size)))))

;;;; ============================================================================
;;;; GRU Cell (Gated Recurrent Unit)
;;;; ============================================================================

(defclass gru-cell (recurrent-cell)
  ((w-ir :accessor w-ir :documentation "Reset gate - input weights")
   (w-hr :accessor w-hr :documentation "Reset gate - hidden weights")
   (b-ir :accessor b-ir)
   (b-hr :accessor b-hr)
   (w-iz :accessor w-iz :documentation "Update gate weights")
   (w-hz :accessor w-hz)
   (b-iz :accessor b-iz)
   (b-hz :accessor b-hz)
   (w-in :accessor w-in :documentation "New gate weights")
   (w-hn :accessor w-hn)
   (b-in :accessor b-in)
   (b-hn :accessor b-hn))
  (:documentation "GRU cell - fewer parameters than LSTM"))

(defmethod initialize-instance :after ((cell gru-cell) &key)
  (with-slots (input-size hidden-size parameters
               w-ir w-hr b-ir b-hr
               w-iz w-hz b-iz b-hz
               w-in w-hn b-in b-hn) cell
    (let ((scale (/ 1.0 (sqrt hidden-size))))
      ;; Reset gate
      (setf w-ir (randn (list hidden-size input-size) :requires-grad t :scale scale))
      (setf w-hr (randn (list hidden-size hidden-size) :requires-grad t :scale scale))
      (setf b-ir (zeros (list hidden-size) :requires-grad t))
      (setf b-hr (zeros (list hidden-size) :requires-grad t))
      ;; Update gate
      (setf w-iz (randn (list hidden-size input-size) :requires-grad t :scale scale))
      (setf w-hz (randn (list hidden-size hidden-size) :requires-grad t :scale scale))
      (setf b-iz (zeros (list hidden-size) :requires-grad t))
      (setf b-hz (zeros (list hidden-size) :requires-grad t))
      ;; New gate
      (setf w-in (randn (list hidden-size input-size) :requires-grad t :scale scale))
      (setf w-hn (randn (list hidden-size hidden-size) :requires-grad t :scale scale))
      (setf b-in (zeros (list hidden-size) :requires-grad t))
      (setf b-hn (zeros (list hidden-size) :requires-grad t))
      
      (setf parameters (list w-ir w-hr b-ir b-hr
                            w-iz w-hz b-iz b-hz
                            w-in w-hn b-in b-hn)))))

(defmethod cell-forward ((cell gru-cell) input hidden)
  "GRU forward pass"
  (with-slots (w-ir w-hr b-ir b-hr
               w-iz w-hz b-iz b-hz
               w-in w-hn b-in b-hn) cell
    ;; Reset gate: r_t = σ(W_ir * x_t + W_hr * h_{t-1} + b_r)
    (let* ((r-gate (sigmoid (t+ (t+ (t@ input (transpose w-ir)) b-ir)
                                (t+ (t@ hidden (transpose w-hr)) b-hr))))
           ;; Update gate: z_t = σ(W_iz * x_t + W_hz * h_{t-1} + b_z)
           (z-gate (sigmoid (t+ (t+ (t@ input (transpose w-iz)) b-iz)
                                (t+ (t@ hidden (transpose w-hz)) b-hz))))
           ;; New gate: n_t = tanh(W_in * x_t + r_t * (W_hn * h_{t-1}) + b_n)
           (n-gate (tanh-activation
                    (t+ (t+ (t@ input (transpose w-in)) b-in)
                        (t* r-gate (t+ (t@ hidden (transpose w-hn)) b-hn)))))
           ;; New hidden: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
           (one (make-tensor (make-array (neural-network::tensor-shape z-gate)
                                        :element-type 'double-float
                                        :initial-element 1.0d0)
                            :shape (neural-network::tensor-shape z-gate)))
           (new-hidden (t+ (t* (t- one z-gate) n-gate)
                          (t* z-gate hidden))))
      new-hidden)))

(defmethod init-hidden ((cell gru-cell) batch-size)
  (with-slots (hidden-size) cell
    (zeros (list batch-size hidden-size))))

;;;; ============================================================================
;;;; Recurrent Layer (processes sequences)
;;;; ============================================================================

(defclass rnn-layer (layer)
  ((cell :initarg :cell
         :accessor rnn-cell
         :documentation "The recurrent cell")
   (return-sequences :initarg :return-sequences
                     :initform nil
                     :accessor return-sequences
                     :documentation "Return all timesteps or just last")
   (stateful :initarg :stateful
             :initform nil
             :accessor stateful
             :documentation "Maintain state between batches"))
  (:documentation "RNN layer that processes sequences"))

(defmethod initialize-instance :after ((layer rnn-layer) &key)
  (with-slots (cell parameters) layer
    (setf parameters (layer-parameters cell))))

(defmethod forward ((layer rnn-layer) inputs)
  "Forward pass through sequence
   inputs: (batch-size, seq-len, input-size) tensor"
  (with-slots (cell return-sequences stateful) layer
    (let* ((shape (neural-network::tensor-shape inputs))
           (batch-size (first shape))
           (seq-len (second shape))
           (input-size (third shape))
           (hidden (if (and stateful (hidden-state cell))
                      (hidden-state cell)
                      (init-hidden cell batch-size)))
           (outputs nil))
      
      ;; Process each timestep
      (dotimes (t-idx seq-len)
        ;; Extract input at time t: inputs[:, t, :]
        (let* ((input-data (neural-network::tensor-data inputs))
               (input-t-data (make-array (list batch-size input-size)
                                        :element-type 'double-float)))
          ;; Copy data for timestep t
          (dotimes (b batch-size)
            (dotimes (i input-size)
              (setf (aref input-t-data b i)
                    (aref input-data b t-idx i))))
          
          ;; Forward through cell
          (let* ((input-t (make-tensor input-t-data))
                 (new-hidden (cell-forward cell input-t hidden)))
            (push new-hidden outputs)
            (setf hidden new-hidden))))
      
      ;; Store hidden state if stateful
      (when stateful
        (setf (hidden-state cell) hidden))
      
      ;; Return based on mode
      (if return-sequences
          ;; Stack all outputs into (batch, seq_len, hidden_size)
          (let* ((hidden-size (hidden-size cell))
                 (result-data (make-array (list batch-size seq-len hidden-size)
                                         :element-type 'double-float))
                 (outputs-list (nreverse outputs)))
            (dotimes (t-idx seq-len)
              (let ((output-t (nth t-idx outputs-list)))
                (dotimes (b batch-size)
                  (dotimes (h hidden-size)
                    (setf (aref result-data b t-idx h)
                          (aref (neural-network::tensor-data 
                                 (if (listp output-t) (first output-t) output-t))
                                b h))))))
            (make-tensor result-data :shape (list batch-size seq-len hidden-size)))
          ;; Just return last hidden state
          (if (listp hidden) (first hidden) hidden)))))

;;;; ============================================================================
;;;; Convenience constructors
;;;; ============================================================================

(defun rnn-layer (input-size hidden-size &key (return-sequences nil) (activation :tanh))
  "Create vanilla RNN layer"
  (make-instance 'rnn-layer
                 :cell (make-instance 'rnn-cell
                                     :input-size input-size
                                     :hidden-size hidden-size
                                     :activation activation)
                 :return-sequences return-sequences))

(defun lstm-layer (input-size hidden-size &key (return-sequences nil))
  "Create LSTM layer"
  (make-instance 'rnn-layer
                 :cell (make-instance 'lstm-cell
                                     :input-size input-size
                                     :hidden-size hidden-size)
                 :return-sequences return-sequences))

(defun gru-layer (input-size hidden-size &key (return-sequences nil))
  "Create GRU layer"
  (make-instance 'rnn-layer
                 :cell (make-instance 'gru-cell
                                     :input-size input-size
                                     :hidden-size hidden-size)
                 :return-sequences return-sequences))

;;;; ============================================================================
;;;; Bidirectional RNN (Lisp's functional composition!)
;;;; ============================================================================

(defclass bidirectional-rnn (layer)
  ((forward-cell :accessor forward-cell)
   (backward-cell :accessor backward-cell)
   (merge-mode :initarg :merge-mode
               :initform :concat
               :accessor merge-mode
               :documentation "How to combine: :concat, :sum, :mul, :ave"))
  (:documentation "Bidirectional RNN layer"))

(defmethod initialize-instance :after ((layer bidirectional-rnn) &key cell-type input-size hidden-size)
  (with-slots (forward-cell backward-cell parameters) layer
    (setf forward-cell
          (ecase cell-type
            (:rnn (make-instance 'rnn-cell :input-size input-size :hidden-size hidden-size))
            (:lstm (make-instance 'lstm-cell :input-size input-size :hidden-size hidden-size))
            (:gru (make-instance 'gru-cell :input-size input-size :hidden-size hidden-size))))
    (setf backward-cell
          (ecase cell-type
            (:rnn (make-instance 'rnn-cell :input-size input-size :hidden-size hidden-size))
            (:lstm (make-instance 'lstm-cell :input-size input-size :hidden-size hidden-size))
            (:gru (make-instance 'gru-cell :input-size input-size :hidden-size hidden-size))))
    (setf parameters (append (layer-parameters forward-cell)
                            (layer-parameters backward-cell)))))

(defmethod forward ((layer bidirectional-rnn) inputs)
  "Process sequence in both directions
   inputs: (batch, seq_len, input_size) tensor"
  (with-slots (forward-cell backward-cell merge-mode) layer
    (let* ((shape (neural-network::tensor-shape inputs))
           (batch-size (first shape))
           (seq-len (second shape))
           (input-size (third shape))
           (hidden-size (hidden-size forward-cell))
           (forward-hidden (init-hidden forward-cell batch-size))
           (backward-hidden (init-hidden backward-cell batch-size))
           (forward-outputs nil)
           (backward-outputs nil))
      
      ;; Forward pass
      (dotimes (t-idx seq-len)
        (let* ((input-data (neural-network::tensor-data inputs))
               (input-t-data (make-array (list batch-size input-size)
                                        :element-type 'double-float)))
          (dotimes (b batch-size)
            (dotimes (i input-size)
              (setf (aref input-t-data b i)
                    (aref input-data b t-idx i))))
          (let* ((input-t (make-tensor input-t-data))
                 (new-hidden (cell-forward forward-cell input-t forward-hidden)))
            (setf forward-hidden new-hidden)
            (push forward-hidden forward-outputs))))
      
      ;; Backward pass (reverse order)
      (loop for t-idx from (1- seq-len) downto 0 do
        (let* ((input-data (neural-network::tensor-data inputs))
               (input-t-data (make-array (list batch-size input-size)
                                        :element-type 'double-float)))
          (dotimes (b batch-size)
            (dotimes (i input-size)
              (setf (aref input-t-data b i)
                    (aref input-data b t-idx i))))
          (let* ((input-t (make-tensor input-t-data))
                 (new-hidden (cell-forward backward-cell input-t backward-hidden)))
            (setf backward-hidden new-hidden)
            (push backward-hidden backward-outputs))))
      
      ;; Merge outputs
      (let ((fwd-list (nreverse forward-outputs))
            (bwd-list backward-outputs))
        (ecase merge-mode
          (:concat
           ;; Concatenate forward and backward outputs
           (let ((result-data (make-array (list batch-size seq-len (* 2 hidden-size))
                                         :element-type 'double-float)))
             (dotimes (t-idx seq-len)
               (let ((fwd-out (nth t-idx fwd-list))
                     (bwd-out (nth t-idx bwd-list)))
                 (dotimes (b batch-size)
                   ;; Forward outputs
                   (dotimes (h hidden-size)
                     (setf (aref result-data b t-idx h)
                           (aref (neural-network::tensor-data 
                                  (if (listp fwd-out) (first fwd-out) fwd-out))
                                 b h)))
                   ;; Backward outputs
                   (dotimes (h hidden-size)
                     (setf (aref result-data b t-idx (+ h hidden-size))
                           (aref (neural-network::tensor-data 
                                  (if (listp bwd-out) (first bwd-out) bwd-out))
                                 b h))))))
             (make-tensor result-data :shape (list batch-size seq-len (* 2 hidden-size))))))))))

(defun bidirectional-lstm (input-size hidden-size &key (merge-mode :concat))
  "Create bidirectional LSTM"
  (make-instance 'bidirectional-rnn
                 :cell-type :lstm
                 :input-size input-size
                 :hidden-size hidden-size
                 :merge-mode merge-mode))

(defun bidirectional-gru (input-size hidden-size &key (merge-mode :concat))
  "Create bidirectional GRU"
  (make-instance 'bidirectional-rnn
                 :cell-type :gru
                 :input-size input-size
                 :hidden-size hidden-size
                 :merge-mode merge-mode))

;;;; ============================================================================
;;;; Stacked RNN (Higher-order composition)
;;;; ============================================================================

(defun stack-rnn (layers)
  "Stack multiple RNN layers - pure functional composition!"
  (lambda (sequence)
    (reduce (lambda (seq layer)
              (forward layer seq))
            layers
            :initial-value sequence)))

;;;; ============================================================================
;;;; Attention-augmented RNN
;;;; ============================================================================

(defclass attention-rnn (layer)
  ((rnn-cell :accessor rnn-cell)
   (attention-weights :accessor attention-weights))
  (:documentation "RNN with attention mechanism"))

(defmethod initialize-instance :after ((layer attention-rnn) &key input-size hidden-size)
  (with-slots (rnn-cell attention-weights parameters) layer
    (setf rnn-cell (make-instance 'lstm-cell
                                 :input-size input-size
                                 :hidden-size hidden-size))
    (setf attention-weights (randn (list hidden-size hidden-size)
                                  :requires-grad t
                                  :scale (/ 1.0 (sqrt hidden-size))))
    (setf parameters (append (layer-parameters rnn-cell)
                            (list attention-weights)))))

;;;; ============================================================================
;;;; Lisp Macros for Recurrent Processing
;;;; ============================================================================

(defmacro with-sequence-processing ((var sequence &key (initial-state nil)) &body body)
  "Macro for processing sequences with state"
  (let ((state-var (gensym "STATE"))
        (seq-var (gensym "SEQ"))
        (results (gensym "RESULTS")))
    `(let ((,state-var ,initial-state)
           (,seq-var ,sequence)
           (,results nil))
       (dolist (,var ,seq-var)
         (multiple-value-bind (output new-state)
             (progn ,@body)
           (push output ,results)
           (setf ,state-var new-state)))
       (values (nreverse ,results) ,state-var))))

(defmacro defrecurrent (name (input-var hidden-var) &body body)
  "Define a custom recurrent cell"
  `(defun ,name (,input-var ,hidden-var)
     ,@body))

;;;; ============================================================================
;;;; Higher-order sequence functions (very Lispy!)
;;;; ============================================================================

(defun sequence-map (fn sequence)
  "Map function over sequence, threading hidden state"
  (lambda (initial-hidden)
    (let ((hidden initial-hidden)
          (outputs nil))
      (dolist (input sequence)
        (multiple-value-bind (output new-hidden)
            (funcall fn input hidden)
          (push output outputs)
          (setf hidden new-hidden)))
      (values (nreverse outputs) hidden))))

(defun sequence-fold (fn initial-value sequence)
  "Fold over sequence with accumulator"
  (reduce fn sequence :initial-value initial-value))

;;;; ============================================================================
;;;; State management helpers
;;;; ============================================================================

(defun reset-hidden-state (layer)
  "Reset hidden state to nil"
  (typecase layer
    (recurrent-cell (setf (hidden-state layer) nil))
    (rnn-layer (reset-hidden-state (rnn-cell layer)))
    (bidirectional-rnn
     (reset-hidden-state (forward-cell layer))
     (reset-hidden-state (backward-cell layer)))))

(defun get-hidden-state (layer)
  "Get current hidden state"
  (typecase layer
    (recurrent-cell (hidden-state layer))
    (rnn-layer (get-hidden-state (rnn-cell layer)))
    (bidirectional-rnn
     (list (get-hidden-state (forward-cell layer))
           (get-hidden-state (backward-cell layer))))))

;;;; ============================================================================
;;;; Example: Named Entity Recognition with BiLSTM
;;;; ============================================================================

(defun create-ner-model (vocab-size embedding-dim hidden-size num-tags)
  "Create a BiLSTM-CRF model for Named Entity Recognition"
  (let ((embedding (randn (list vocab-size embedding-dim)
                         :requires-grad t
                         :name "embedding"))
        (bilstm (bidirectional-lstm embedding-dim hidden-size))
        (linear (linear (* 2 hidden-size) num-tags)))
    (list :embedding embedding
          :bilstm bilstm
          :linear linear)))
