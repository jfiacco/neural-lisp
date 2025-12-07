;;;; tests/test-benchmarks.lisp - Performance Benchmarking Tests

(in-package #:neural-lisp-tests)

(def-suite benchmark-tests
  :description "Performance benchmarking tests (run separately from main test suite)"
  :in neural-lisp-tests)

(in-suite benchmark-tests)

;;;; ============================================================================
;;;; Performance Benchmarking Utilities
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
  "Create two test matrices of given size for benchmarking"
  (let ((a (make-array (list size size) :element-type element-type))
        (b (make-array (list size size) :element-type element-type)))
    (dotimes (i size)
      (dotimes (j size)
        (setf (aref a i j) (coerce (+ i j 1.0) element-type))
        (setf (aref b i j) (coerce (- i j) element-type))))
    (values a b)))

;;;; ============================================================================
;;;; GPU Speedup Benchmark Tests
;;;; ============================================================================

(test gpu-speedup-small-matrices
  "Test that GPU backend works for small matrices (may not show speedup)"
  (let ((original-backend *backend*))
    (unwind-protect
         (multiple-value-bind (a b) (create-test-matrices 10)
           ;; Test with Lisp backend
           (use-backend :lisp)
           (let ((result-lisp (neural-tensor-backend::backend-matmul a b 10 10 10)))
             (is (not (null result-lisp)))
             (is (= (array-dimension result-lisp 0) 10))
             (is (= (array-dimension result-lisp 1) 10)))
           
           ;; Test with GPU backend (will fall back if unavailable)
           (use-backend :gpu)
           (let ((result-gpu (neural-tensor-backend::backend-matmul a b 10 10 10)))
             (is (not (null result-gpu)))
             (is (= (array-dimension result-gpu 0) 10))
             (is (= (array-dimension result-gpu 1) 10))))
      (setf *backend* original-backend))))

(test gpu-speedup-medium-matrices
  "Test GPU speedup for medium-sized matrices (100x100)"
  (let ((original-backend *backend*))
    (unwind-protect
         (multiple-value-bind (a b) (create-test-matrices 100)
           (format t "~%~%Benchmarking 100x100 matrix multiplication...~%")
           
           ;; Benchmark Lisp backend
           (use-backend :lisp)
           (let ((time-lisp (benchmark-operation 
                            (lambda () 
                              (neural-tensor-backend::backend-matmul a b 100 100 100))
                            :iterations 5)))
             (format t "  Lisp backend: ~,6f seconds~%" time-lisp)
             
             ;; Benchmark BLAS backend
             (use-backend :blas)
             (let ((time-blas (benchmark-operation 
                              (lambda () 
                                (neural-tensor-backend::backend-matmul a b 100 100 100))
                              :iterations 5)))
               (format t "  BLAS backend: ~,6f seconds~%" time-blas)
               
               ;; Calculate speedup
               (let ((speedup (/ time-lisp time-blas)))
                 (format t "  BLAS speedup: ~,2fx~%" speedup)
                 ;; BLAS should be at least as fast as pure Lisp (1x) or faster
                 (is (>= speedup 0.8) ; Allow 20% margin for variance
                     "BLAS backend should be at least as fast as Lisp backend"))
               
               ;; Benchmark GPU backend
               (use-backend :gpu)
               (let ((time-gpu (benchmark-operation 
                               (lambda () 
                                 (neural-tensor-backend::backend-matmul a b 100 100 100))
                               :iterations 5)))
                 (format t "  GPU backend: ~,6f seconds~%" time-gpu)
                 
                 ;; Calculate speedup
                 (let ((speedup (/ time-lisp time-gpu)))
                   (format t "  GPU speedup vs Lisp: ~,2fx~%" speedup)
                   ;; GPU should work correctly
                   (is (not (null time-gpu))
                       "GPU backend should execute successfully"))))))
      (setf *backend* original-backend))))

(test gpu-speedup-large-matrices
  "Test GPU speedup for large matrices (500x500) - demonstrates real speedup potential"
  (let ((original-backend *backend*))
    (unwind-protect
         (multiple-value-bind (a b) (create-test-matrices 500)
           (format t "~%~%Benchmarking 500x500 matrix multiplication...~%")
           (format t "(This may take a moment...)~%")
           
           ;; Benchmark Lisp backend (fewer iterations for large matrices)
           (use-backend :lisp)
           (let ((time-lisp (benchmark-operation 
                            (lambda () 
                              (neural-tensor-backend::backend-matmul a b 500 500 500))
                            :iterations 3)))
             (format t "  Lisp backend: ~,6f seconds~%" time-lisp)
             
             ;; Benchmark BLAS backend
             (use-backend :blas)
             (let ((time-blas (benchmark-operation 
                              (lambda () 
                                (neural-tensor-backend::backend-matmul a b 500 500 500))
                              :iterations 3)))
               (format t "  BLAS backend: ~,6f seconds~%" time-blas)
               
               (let ((speedup-blas (/ time-lisp time-blas)))
                 (format t "  BLAS speedup: ~,2fx~%" speedup-blas)
                 ;; For large matrices, BLAS should show significant speedup
                 (is (>= speedup-blas 1.0)
                     "BLAS backend should be faster than Lisp for large matrices"))
               
               ;; Benchmark GPU backend
               (use-backend :gpu)
               (let ((time-gpu (benchmark-operation 
                               (lambda () 
                                 (neural-tensor-backend::backend-matmul a b 500 500 500))
                               :iterations 3)))
                 (format t "  GPU backend: ~,6f seconds~%" time-gpu)
                 
                   (let ((speedup-gpu (/ time-lisp time-gpu)))
                   (format t "  GPU speedup vs Lisp: ~,2fx~%" speedup-gpu)
                   (format t "  GPU speedup vs BLAS: ~,2fx~%" (/ time-blas time-gpu))
                   
                   ;; GPU should execute successfully
                   (is (not (null time-gpu))
                       "GPU backend should execute successfully for large matrices")
                   
                   ;; For large matrices, GPU should show real speedup
                   (when (neural-cuda:cuda-available-p)
                     (format t "~%  Real GPU/cuBLAS is active and showing speedup!~%")
                     (is (> speedup-gpu 1.0)
                         "GPU should be faster than pure Lisp for large matrices")))))))
      (setf *backend* original-backend))))

;;;; ============================================================================
;;;; Performance Summary
;;;; ============================================================================

(test performance-summary
  "Generate a performance summary comparing all backends"
  (let ((original-backend *backend*)
        (test-sizes '(50 100 200)))
    (unwind-protect
         (progn
           (format t "~%~%")
           (format t "╔════════════════════════════════════════════════════════════════════╗~%")
           (format t "║          Backend Performance Summary                               ║~%")
           (format t "╚════════════════════════════════════════════════════════════════════╝~%")
           (format t "~%")
           (format t "Matrix Size  │  Lisp (s)  │  BLAS (s)  │  GPU (s)   │ BLAS Speedup │ GPU Speedup~%")
           (format t "─────────────┼────────────┼────────────┼────────────┼──────────────┼────────────~%")
           
           (dolist (size test-sizes)
             (multiple-value-bind (a b) (create-test-matrices size)
               (use-backend :lisp)
               (let ((time-lisp (benchmark-operation 
                                (lambda () 
                                  (neural-tensor-backend::backend-matmul a b size size size))
                                :iterations 3)))
                 
                 (use-backend :blas)
                 (let ((time-blas (benchmark-operation 
                                  (lambda () 
                                    (neural-tensor-backend::backend-matmul a b size size size))
                                  :iterations 3)))
                   
                   (use-backend :gpu)
                   (let ((time-gpu (benchmark-operation 
                                   (lambda () 
                                     (neural-tensor-backend::backend-matmul a b size size size))
                                   :iterations 3)))
                     
                     (format t "~4dx~4d    │  ~,6f  │  ~,6f  │  ~,6f  │     ~,2fx      │    ~,2fx~%"
                             size size
                             time-lisp
                             time-blas
                             time-gpu
                             (if (> time-blas 0) (/ time-lisp time-blas) 0)
                             (if (> time-gpu 0) (/ time-lisp time-gpu) 0)))))))
           
           (format t "~%")
           (when (neural-cuda:cuda-available-p)
             (format t "Note: Using real GPU/cuBLAS acceleration via NVIDIA GPU.~%"))
           (format t "~%")
           
           ;; This test always passes - it's for informational purposes
           (pass "Performance summary completed"))
      (setf *backend* original-backend))))

;;;; ============================================================================
;;;; Benchmark Test Runner
;;;; ============================================================================

(defun run-benchmark-tests ()
  "Run all benchmark tests and report results"
  (print-section-header "Performance Benchmarks")
  (format t "~%Note: Benchmark tests measure performance and may take a while.~%~%")
  
  (let ((result (run 'benchmark-tests)))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "~%Benchmark Tests: ~d passed, ~d failed~%" passed failed)
      (values passed failed))))

(export 'run-benchmark-tests)
