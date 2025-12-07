;;;; cuda-bindings.lisp - Minimal CUDA/cuBLAS CFFI Bindings

(defpackage :neural-cuda
  (:use :common-lisp :cffi)
  (:export #:cuda-available-p
           #:init-cuda
           #:shutdown-cuda
           #:cuda-malloc
           #:cuda-free
           #:cuda-memcpy-host-to-device
           #:cuda-memcpy-device-to-host
           #:cublas-sgemm
           #:cublas-dgemm
           #:with-cuda-context))

(in-package :neural-cuda)

;;;; ============================================================================
;;;; CUDA Runtime API Bindings
;;;; ============================================================================

(define-foreign-library libcuda
  (:unix (:or "/usr/lib/wsl/lib/libcuda.so.1" 
              "/usr/lib/wsl/lib/libcuda.so"
              "libcuda.so.1" 
              "libcuda.so"))
  (t (:default "libcuda")))

(define-foreign-library libcudart
  (:unix (:or "/usr/local/lib/ollama/cuda_v13/libcudart.so.13"
              "/usr/local/lib/ollama/cuda_v12/libcudart.so.12"
              "libcudart.so.13"
              "libcudart.so.12" 
              "libcudart.so.11"
              "libcudart.so"))
  (t (:default "libcudart")))

(define-foreign-library libcublas
  (:unix (:or "/usr/local/lib/ollama/cuda_v13/libcublas.so.13"
              "/usr/local/lib/ollama/cuda_v12/libcublas.so.12"
              "libcublas.so.13"
              "libcublas.so.12" 
              "libcublas.so.11" 
              "libcublas.so"))
  (t (:default "libcublas")))

;; Try to load libraries, but don't error if they fail
(defvar *cuda-loaded* nil)
(defvar *cublas-loaded* nil)

(defun try-load-cuda ()
  "Attempt to load CUDA libraries"
  (handler-case
      (progn
        ;; Set library paths
        (pushnew "/usr/lib/wsl/lib/" *foreign-library-directories* :test #'string=)
        (pushnew "/usr/local/lib/ollama/cuda_v13/" *foreign-library-directories* :test #'string=)
        (pushnew "/usr/local/lib/ollama/cuda_v12/" *foreign-library-directories* :test #'string=)
        (pushnew "/usr/local/cuda/lib64/" *foreign-library-directories* :test #'string=)
        
        (use-foreign-library libcuda)
        (format t "[CUDA] Loaded libcuda~%")
        
        (use-foreign-library libcudart)
        (format t "[CUDA] Loaded libcudart~%")
        
        (setf *cuda-loaded* t)
        t)
    (error (e)
      (format t "~%Warning: Could not load CUDA libraries: ~a~%" e)
      nil)))

(defun try-load-cublas ()
  "Attempt to load cuBLAS library"
  (when *cuda-loaded*
    (handler-case
        (progn
          (use-foreign-library libcublas)
          (format t "[CUDA] Loaded libcublas~%")
          (setf *cublas-loaded* t)
          t)
      (error (e)
        (format t "~%Warning: Could not load cuBLAS library: ~a~%" e)
        nil))))

;;;; ============================================================================
;;;; CUDA Error Handling
;;;; ============================================================================

(defconstant +cuda-success+ 0)
(defconstant +cublas-status-success+ 0)

(defcfun ("cudaGetDeviceCount" %cuda-get-device-count) :int
  (count :pointer))

(defcfun ("cudaGetDeviceProperties" %cuda-get-device-properties) :int
  (prop :pointer)
  (device :int))

(defcfun ("cudaMalloc" %cuda-malloc) :int
  (devPtr :pointer)
  (size :size))

(defcfun ("cudaFree" %cuda-free) :int
  (devPtr :pointer))

(defcfun ("cudaMemcpy" %cuda-memcpy) :int
  (dst :pointer)
  (src :pointer)
  (count :size)
  (kind :int))

(defconstant +cuda-memcpy-host-to-device+ 1)
(defconstant +cuda-memcpy-device-to-host+ 2)
(defconstant +cuda-memcpy-device-to-device+ 3)

;;;; ============================================================================
;;;; cuBLAS API Bindings
;;;; ============================================================================

(defcfun ("cublasCreate_v2" %cublas-create) :int
  (handle :pointer))

(defcfun ("cublasDestroy_v2" %cublas-destroy) :int
  (handle :pointer))

(defcfun ("cublasSgemm_v2" %cublas-sgemm) :int
  (handle :pointer)
  (transa :int)
  (transb :int)
  (m :int)
  (n :int)
  (k :int)
  (alpha :pointer)
  (A :pointer)
  (lda :int)
  (B :pointer)
  (ldb :int)
  (beta :pointer)
  (C :pointer)
  (ldc :int))

(defcfun ("cublasDgemm_v2" %cublas-dgemm) :int
  (handle :pointer)
  (transa :int)
  (transb :int)
  (m :int)
  (n :int)
  (k :int)
  (alpha :pointer)
  (A :pointer)
  (lda :int)
  (B :pointer)
  (ldb :int)
  (beta :pointer)
  (C :pointer)
  (ldc :int))

(defconstant +cublas-op-n+ 0) ; No transpose
(defconstant +cublas-op-t+ 1) ; Transpose

;;;; ============================================================================
;;;; High-Level Interface
;;;; ============================================================================

(defvar *cublas-handle* nil
  "Global cuBLAS handle")

(defun cuda-available-p ()
  "Check if CUDA is available and working"
  (when *cuda-loaded*
    (handler-case
        (with-foreign-object (count :int)
          (let ((status (%cuda-get-device-count count)))
            (and (= status +cuda-success+)
                 (> (mem-ref count :int) 0))))
      (error () nil))))

(defun init-cuda ()
  "Initialize CUDA and cuBLAS"
  (unless *cuda-loaded*
    (try-load-cuda))
  
  (when *cuda-loaded*
    (unless *cublas-loaded*
      (try-load-cublas))
    
    (when (and *cublas-loaded* (not *cublas-handle*))
      (let ((handle (foreign-alloc :pointer)))
        (when (= (%cublas-create handle) +cublas-status-success+)
          (setf *cublas-handle* (mem-ref handle :pointer))
          (format t "~%[CUDA] Initialized successfully with cuBLAS~%")
          t)))))

(defun shutdown-cuda ()
  "Shutdown CUDA and clean up"
  (when *cublas-handle*
    (%cublas-destroy *cublas-handle*)
    (setf *cublas-handle* nil))
  (format t "[CUDA] Shutdown complete~%"))

(defun cuda-malloc (size)
  "Allocate device memory"
  (with-foreign-object (devptr :pointer)
    (let ((status (%cuda-malloc devptr size)))
      (if (= status +cuda-success+)
          (mem-ref devptr :pointer)
          (error "CUDA malloc failed with status ~a" status)))))

(defun cuda-free (devptr)
  "Free device memory"
  (let ((status (%cuda-free devptr)))
    (unless (= status +cuda-success+)
      (error "CUDA free failed with status ~a" status))))

(defun cuda-memcpy-host-to-device (device-ptr host-ptr size)
  "Copy data from host to device"
  (let ((status (%cuda-memcpy device-ptr host-ptr size +cuda-memcpy-host-to-device+)))
    (unless (= status +cuda-success+)
      (error "CUDA memcpy H2D failed with status ~a" status))))

(defun cuda-memcpy-device-to-host (host-ptr device-ptr size)
  "Copy data from device to host"
  (let ((status (%cuda-memcpy host-ptr device-ptr size +cuda-memcpy-device-to-host+)))
    (unless (= status +cuda-success+)
      (error "CUDA memcpy D2H failed with status ~a" status))))

(defun cublas-dgemm (m n k alpha A lda B ldb beta C ldc)
  "Perform double-precision matrix multiplication: C = alpha*A*B + beta*C"
  (unless *cublas-handle*
    (error "cuBLAS not initialized. Call init-cuda first."))
  
  (with-foreign-objects ((alpha-ptr :double)
                         (beta-ptr :double))
    (setf (mem-ref alpha-ptr :double) (coerce alpha 'double-float))
    (setf (mem-ref beta-ptr :double) (coerce beta 'double-float))
    
    (let ((status (%cublas-dgemm *cublas-handle*
                                 +cublas-op-n+  ; No transpose for A
                                 +cublas-op-n+  ; No transpose for B
                                 m n k
                                 alpha-ptr
                                 A lda
                                 B ldb
                                 beta-ptr
                                 C ldc)))
      (unless (= status +cublas-status-success+)
        (error "cuBLAS DGEMM failed with status ~a" status)))))

(defmacro with-cuda-context (&body body)
  "Execute body with CUDA context initialized"
  `(unwind-protect
        (progn
          (init-cuda)
          ,@body)
     (shutdown-cuda)))

;; Attempt to load CUDA on package load
(eval-when (:load-toplevel :execute)
  (try-load-cuda)
  (try-load-cublas))
