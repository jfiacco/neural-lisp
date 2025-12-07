#!/bin/bash
# Script to run all neural-lisp tests

set -e

sbcl --noinform --disable-debugger \
     --eval "(require :asdf)" \
     --eval "(push (truename \".\") asdf:*central-registry*)" \
     --eval "(asdf:load-system :neural-lisp/tests)" \
     --eval "(let ((exit-code (neural-lisp-tests:run-neural-lisp-tests))) (sb-ext:exit :code exit-code))"
