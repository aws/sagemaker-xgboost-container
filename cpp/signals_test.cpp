/**
 * C++ file to generate certain breaking cases to test the terminate handler and signal handler.
 */

#include <exception>
#include <stdexcept>
#include <unistd.h>
#include "signals.cpp"

namespace smt {

  extern "C" void cause_crash() {
    while (true) {
      int* big = new int[1 << 30];
      big++;
    }
  }

  extern "C" void cause_segfault() {
    int* n = nullptr;
    while (true) {
      (*n)++;
    }
  }

  extern "C" void throw_exception_with_oom_message() {
    throw std::runtime_error("cudaMalloc failed: out of memory");
  }

  extern "C" void throw_exception_with_random_message() {
    throw std::runtime_error("Unhandled exception");
  }

  extern "C" void throw_exception_with_empty_message() {
    throw std::exception();
  }

  extern "C" void successful_exit() {
    exit(0);
  }

}