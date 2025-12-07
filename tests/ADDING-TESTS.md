# Adding New Tests to Neural-Lisp

Quick guide for adding new test files to the test framework.

## 1. Create the Test File

Create `tests/test-<component>.lisp`:

```lisp
;;;; tests/test-<component>.lisp - Description

(in-package #:neural-lisp-tests)

(def-suite <component>-tests
  :description "Tests for <component>"
  :in neural-lisp-tests)

(in-suite <component>-tests)

;;; Your tests
(test test-basic-functionality
  "Test description"
  (is (= 1 1)))

(test test-edge-cases
  "Test edge cases"
  (signals error (some-invalid-operation)))
```

## 2. Add to Test Runner

Edit `tests/run-tests.lisp`:

### For Core Tests (Run by Default)

Add to `run-neural-lisp-tests` function:

```lisp
;; Run <component> tests
(format t "Running <Component> Tests...~%")
(let ((result (run '<component>-tests)))
  (when verbose (explain! result))
  (multiple-value-bind (passed failed)
      (count-test-results result)
    (format t "  <Component> Tests: ~d passed, ~d failed~%" passed failed)
    (record-suite-result "<Component>" passed failed)))
```

Add convenience function:

```lisp
(defun test-<component> () (run-suite '<component>-tests))
```

Add to exports:

```lisp
(export '(run-neural-lisp-tests
          ...
          test-<component>
          ...))
```

### For Advanced/Optional Tests

If tests require loading external files or should be run separately:

```lisp
(defun test-<component> ()
  "Run <component> tests"
  (format t "~%Loading <component> tests...~%")
  (load "tests/test-<component>.lisp")
  (funcall (find-symbol "RUN-<COMPONENT>-TESTS" "NEURAL-TENSOR-<COMPONENT>-TESTS")))
```

Add runner function in your test file:

```lisp
(defun run-<component>-tests ()
  "Run all <component> tests and report results"
  (let ((result (run '<component>-tests)))
    (multiple-value-bind (passed failed)
        (count-test-results result)
      (format t "~%<Component> Tests: ~d passed, ~d failed~%" passed failed)
      (values passed failed))))

(export 'run-<component>-tests)
```

## 3. Import Required Symbols (if needed)

Edit `tests/package.lisp` to import symbols from your component:

```lisp
(:import-from #:your-package
              #:function-name
              #:class-name
              ...)
```

## 4. Test Your Tests

```bash
./run-tests.sh
```

Or in REPL:

```lisp
(asdf:load-system :neural-lisp/tests)
(neural-lisp-tests:test-<component>)
```

## Test Types

### Regular Assertions
```lisp
(test simple-test
  (is (= 2 (+ 1 1)))
  (is (equal '(1 2 3) result)))
```

### Error Handling
```lisp
(test error-test
  (signals error (divide-by-zero))
  (finishes (safe-operation)))
```

### Floating Point Comparisons
```lisp
(test float-test
  (is (< (abs (- expected actual)) 1e-6)))
```

### Fixtures
```lisp
(def-fixture test-data ()
  (let ((data (create-test-data)))
    (&body)))

(test with-fixture
  (with-fixture test-data ()
    (is (not (null data)))))
```

## Benchmark Tests

For performance benchmarks, add to `tests/test-benchmarks.lisp` instead of regular test files. These are only run when explicitly requested via `(neural-lisp-tests:test-benchmarks)`.

## Tips

- **Test names**: Use descriptive names like `test-matmul-correctness` 
- **Assertions**: Each test should have clear assertions with messages
- **Cleanup**: Use `unwind-protect` for resource cleanup
- **Independence**: Tests should not depend on each other's state
- **Documentation**: Include docstrings explaining what each test verifies

## Example: Complete Test File

```lisp
;;;; tests/test-matrix.lisp - Matrix Operations Tests

(in-package #:neural-lisp-tests)

(def-suite matrix-tests
  :description "Tests for matrix operations"
  :in neural-lisp-tests)

(in-suite matrix-tests)

(test matrix-creation
  "Test matrix creation"
  (let ((m (make-matrix 3 3)))
    (is (not (null m)))
    (is (= 3 (matrix-rows m)))
    (is (= 3 (matrix-cols m)))))

(test matrix-multiplication
  "Test matrix multiplication"
  (let ((a (make-identity-matrix 2))
        (b (make-matrix 2 2 :initial-element 5.0)))
    (let ((result (matrix-multiply a b)))
      (is (< (abs (- 5.0 (matrix-ref result 0 0))) 1e-6))
      (is (< (abs (- 5.0 (matrix-ref result 1 1))) 1e-6)))))

(test matrix-invalid-dimensions
  "Test error handling for invalid dimensions"
  (let ((a (make-matrix 2 3))
        (b (make-matrix 4 2)))
    (signals dimension-mismatch-error
      (matrix-multiply a b))))
```

Then add to `run-tests.lisp` and export `test-matrix`.
