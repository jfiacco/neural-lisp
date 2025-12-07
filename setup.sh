#!/bin/bash
# setup.sh - Install dependencies and verify neural-lisp system

echo "========================================="
echo "Neural-Lisp Setup"
echo "========================================="
echo ""

# Check if SBCL is installed
if ! command -v sbcl &> /dev/null; then
    echo "ERROR: SBCL is not installed."
    echo "Please install SBCL first:"
    echo "  - Ubuntu/Debian: sudo apt-get install sbcl"
    echo "  - macOS: brew install sbcl"
    echo "  - Arch: sudo pacman -S sbcl"
    exit 1
fi

echo "✓ SBCL found: $(sbcl --version | head -n 1)"
echo ""

# Check if Quicklisp is installed
echo "Checking for Quicklisp..."
sbcl --noinform --non-interactive --eval "(progn (if (find-package :quicklisp) (format t \"✓ Quicklisp is installed~%\") (format t \"✗ Quicklisp is NOT installed~%Please install Quicklisp from https://www.quicklisp.org/~%\")) (quit))"

if [ $? -ne 0 ]; then
    echo ""
    echo "To install Quicklisp, run:"
    echo "  curl -O https://beta.quicklisp.org/quicklisp.lisp"
    echo "  sbcl --load quicklisp.lisp --eval '(quicklisp-quickstart:install)' --quit"
    exit 1
fi

echo ""
echo "Installing FiveAM testing framework..."
sbcl --noinform --non-interactive --eval "(progn (ql:quickload :fiveam :silent t) (format t \"✓ FiveAM installed~%\") (quit))"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install FiveAM"
    exit 1
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To load the system:"
echo "  sbcl --eval '(asdf:load-system :neural-lisp)'"
echo ""
echo "To run tests:"
echo "  sbcl --eval '(asdf:test-system :neural-lisp)'"
echo ""
echo "To run examples:"
echo "  cd examples && sbcl --script complete-demo.lisp"
echo "  cd examples && sbcl --script lisp-vs-python.lisp"
echo ""
