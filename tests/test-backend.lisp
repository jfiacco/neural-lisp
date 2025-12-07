;;;; tests/test-backend.lisp - Backend Selection and Correctness Tests

(in-package #:neural-lisp-tests)

(def-suite backend-tests
  :description "Tests for backend selection and correctness"
  :in neural-lisp-tests)

(in-suite backend-tests)

;;;; ============================================================================
;;;; Backend Switching Tests
;;;; ============================================================================

(test backend-selection
  "Test that backend can be switched properly"
  (let ((original-backend *backend*))
    (unwind-protect
         (progn
           (use-backend :lisp)
           (is (eq *backend* :lisp))
           
           (use-backend :blas)
           (is (eq *backend* :blas))
           
           (use-backend :gpu)
           (is (eq *backend* :gpu)))
      ;; Restore original backend
      (setf *backend* original-backend))))

(test backend-with-macro
  "Test with-backend macro for temporary backend switching"
  (let ((original-backend *backend*))
    (use-backend :lisp)
    (is (eq *backend* :lisp))
    
    (with-backend :blas
      (is (eq *backend* :blas)))
    
    ;; Should revert after with-backend
    (is (eq *backend* :lisp))
    
    ;; Restore
    (setf *backend* original-backend)))

;;;; ============================================================================
;;;; Helper Functions
;;;; ============================================================================

(defun benchmark-operation (fn &key (iterations 10))
  "Benchmark an operation and return average time in seconds"
  (let ((times '()))
    (dotimes (i iterations)
      (let ((start (get-internal-real-time)))
        (funcall fn)
        (let ((end (get-internal-real-time)))
          (push (/ (- end start) internal-time-units-per-second) times))))
    ;; Return average, excluding first run (warmup)
    (let ((avg (if (> (length times) 1)
                   (/ (reduce #'+ (rest times)) (1- (length times)))
                   (first times))))
      ;; Ensure we never return exactly 0 to avoid division by zero
      (if (zerop avg)
          1.0e-9  ; Return a very small number instead
          avg))))

(defun create-test-matrices (size &key (element-type 'double-float))
  "Create two test matrices of given size for testing"
  (let ((a (make-array (list size size) :element-type element-type))
        (b (make-array (list size size) :element-type element-type)))
    (dotimes (i size)
      (dotimes (j size)
        (setf (aref a i j) (coerce (+ i j 1.0) element-type))
        (setf (aref b i j) (coerce (- i j) element-type))))
    (values a b)))

;;;; ============================================================================
;;;; GPU Availability Tests
;;;; ============================================================================

(test gpu-availability-check
  "Test GPU availability detection"
  (let ((gpu-available (neural-tensor-backend::gpu-available-p)))
    (format t "~%GPU Available: ~a~%" gpu-available)
    ;; GPU availability depends on whether :opencl feature is present
    (is (or (eq gpu-available t) (eq gpu-available nil))
        "gpu-available-p should return a boolean")))

(test gpu-fallback-behavior
  "Test that GPU operations fall back gracefully when GPU unavailable"
  (let ((original-backend *backend*))
    (unwind-protect
         (progn
           (use-backend :gpu)
           (multiple-value-bind (a b) (create-test-matrices 50)
             ;; Should work even if GPU not available (falls back)
             (let ((result (neural-tensor-backend::backend-matmul a b 50 50 50)))
               (is (not (null result)))
               (is (= (array-dimension result 0) 50))
               (is (= (array-dimension result 1) 50))
               (format t "~%GPU backend executed successfully (with or without real GPU)~%"))))
      (setf *backend* original-backend))))

;;;; ============================================================================
;;;; Correctness Tests - All Backends Should Give Same Results
;;;; ============================================================================

(test backend-correctness
  "Test that all backends produce the same results"
  (let ((original-backend *backend*))
    (unwind-protect
         (multiple-value-bind (a b) (create-test-matrices 20)
           ;; Compute with Lisp backend
           (use-backend :lisp)
           (let ((result-lisp (neural-tensor-backend::backend-matmul a b 20 20 20)))
             
             ;; Compute with BLAS backend
             (use-backend :blas)
             (let ((result-blas (neural-tensor-backend::backend-matmul a b 20 20 20)))
               
               ;; Compute with GPU backend
               (use-backend :gpu)
               (let ((result-gpu (neural-tensor-backend::backend-matmul a b 20 20 20)))
                 
                 ;; Compare results (allowing for floating point tolerance)
                 (dotimes (i 20)
                   (dotimes (j 20)
                     (is (< (abs (- (aref result-lisp i j) (aref result-blas i j))) 1e-6)
                         "Lisp and BLAS backends should produce same results")
                     (is (< (abs (- (aref result-lisp i j) (aref result-gpu i j))) 1e-6)
                         "Lisp and GPU backends should produce same results")))
                 
                 (format t "~%All backends produced consistent results~%")))))
      (setf *backend* original-backend))))
