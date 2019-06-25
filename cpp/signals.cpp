/*
 * This file contains code to deal with https://en.cppreference.com/w/cpp/error/terminate
 * and SIGSEV. The purpose is two fold, guarantee that enough information is stored to
 * facilitate debugging, and to provide sensible/useful messages to customers.
 */

#include <csignal>
#include <cstring>
#include <exception>
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>
#include <unistd.h>

/*
 * Macros to facilitate writing to failure/error files consumed by EASE
 */
#define LOG(file, msg)                                \
  {                                                   \
    std::ofstream fs{file, std::ofstream::app};	      \
    fs << msg;					      \
  }
#define LOG_ERROR(msg) LOG(ERROR_FILEPATH, msg)
#define LOG_FAILURE(msg) LOG(FAILURE_FILEPATH, msg)
#define LOG_STDERR(msg) std::cerr << msg

namespace {

  // Define the exit codes
  constexpr int CUST_ERROR_EXIT_CODE = 2;
  constexpr int ALGO_ERROR_EXIT_CODE = 1;

  // Define the customer facing error messages

  // Out of Memory user facing error messaging
  const char* OUT_OF_MEMORY_MESSAGE = "CustomerError: Out of Memory."
    "Please use a larger instance and/or "
    "reduce the values of other parameters (e.g. num_classes.) if applicable";

  // LIBSVM duplicated feature values user facing error messaging
  const char* DUPLICATED_FEATURE_MESSAGE = "CustomerError: Data Duplication Error. "
    "There are potential features with duplicated values in your dataset(e.g. 2 1:34 5:93 1:59). "
    "Please ensure your dataset does not contain duplicated values."
    "If you still see this problem after ensuring your data is correct, "
    "please reach out to AWS support for further investigation.";

  // LIBSVM negative feature index user facing error messaging
  const char* NEGATIVE_INDEX_MESSAGE = "CustomerError: Negative Index Error. "
    "There are negative feature indexes found in your dataset(e.g. 2 1:34 -5:93 8:10). "
    "Please ensure your dataset does not contain negative indexes. "
    "If you still see this problem after ensuring your data is correct, "
    "please reach out to AWS support for further investigation.";

  // Input data set contain 'inf' value
  const char* CONTAIN_INF_MESSAGE = "CustomerError: 'inf' found in the dataset."
    "Histogram method cannot handle 'inf' value. Please remove 'inf' from the dataset.";

  // Customer message for unhandled exceptions for which no customized message is decided
  const char* FALLBACK_EXCEPTION_MESSAGE = "AlgorithmError: Internal Server Error";

  // Customer message when a segmentation fault occurs in the CPU process
  const char* SIGSEV_MSG = "AlgorithmError: Segmentation Fault";


  // EASE failure file location
  std::string FAILURE_FILEPATH = "/opt/ml/output/failure";

  // EASE trusted algorithm error file location: https://tiny.amazon.com/f8519uzv
  std::string ERROR_FILEPATH = "/opt/ml/errors/errors.log";

  // Map for specific strings and what error messages to be shown to customer. We look for the 'key' in the what()
  //of the exception stacktrace for the general exception handler and show the 'value' to the user as the message.
  const std::map<const char*, const char*> CUSTOMER_MESSAGE_MAP =
  {
    {"out of memory", OUT_OF_MEMORY_MESSAGE},
    {"Check failed: tree[nid].is_leaf()", DUPLICATED_FEATURE_MESSAGE},
    {"src/data/././strtonum.h:141: Check failed: sign == true (0 vs. 1)", NEGATIVE_INDEX_MESSAGE},
    {"src/tree/updater_histmaker.cc:308: fv=inf, hist.last=inf", CONTAIN_INF_MESSAGE}
  };


  /*
   * Capture the backtrace and log it in the errors file
   */
  void log_backtrace() {
    constexpr int MAX_FRAMES = 100;
    void* buffer[MAX_FRAMES];
    const auto size = backtrace(buffer, MAX_FRAMES);
    char** strings = backtrace_symbols(buffer, size);

    // Log backtrace to Error file
    for (auto i = 0 ; i < size ; ++i) {
      LOG_ERROR(strings[i] << std::endl;);
    }
  }

  /*
   * In case of the generic Exception, this method looks for specific string in the what() of the
   * Exception stacktrace and shows specific message to the user based on the above defined map.
   */
  std::pair<const char*, int> get_failure_info(const std::exception& ex) {
    const char* exception_msg = ex.what();
    // If we know any predefined error strings, then in all likelihood we'll tag those as Customer Error.
    for (const auto& kv: CUSTOMER_MESSAGE_MAP) {
      if(strstr(exception_msg, kv.first) != nullptr) {
	return std::make_pair(kv.second, CUST_ERROR_EXIT_CODE);
      }
    }
    return std::make_pair(FALLBACK_EXCEPTION_MESSAGE, ALGO_ERROR_EXIT_CODE);
  }

  /*
   * Report msg/what and exit application with exit_code
   */
  void report_and_exit(const char* msg,
		       const char* what,
		       int exit_code = ALGO_ERROR_EXIT_CODE) {
    LOG_FAILURE(msg << std::endl);
    LOG_ERROR(msg << std::endl << what << std::endl);
    LOG_STDERR(msg << std::endl << what << std::endl);
    std::cout << msg << std::endl << what << std::endl;
    exit(exit_code);
  }
}

namespace sm {
  /*
   * Set a handler for terminate, which is invoked amognst other situations when
   * exceptions are not handled
   */
  extern "C" void set_terminate() {
    void (*handler)() = []() {
      LOG_ERROR("Teminating Application" << std::endl);
      LOG_STDERR("Teminating Application" << std::endl);

      // Log the backtrace before rethrowing
      log_backtrace();

      // Capture and rethrow exception, so that we can inspect it
      const auto& ex = std::current_exception();
      try {
	std::rethrow_exception(ex);
      } catch (const std::bad_alloc& ex) {
	report_and_exit(OUT_OF_MEMORY_MESSAGE, ex.what(), CUST_ERROR_EXIT_CODE);
      } catch (const std::exception& ex) {
	const auto& failure_info = get_failure_info(ex);
	report_and_exit(failure_info.first, ex.what(), failure_info.second);
      }

      // Everything that ends up here is going to be treated initially as an algorithm error
      std::exit(ALGO_ERROR_EXIT_CODE);
    };

    std::set_terminate(handler);
  }

  /*
   * Set a signal handler for SIGSEV, a.k.a. segmentation faults
   */
  extern "C" void set_signal_handlers() {
    void (*handler)(int) = [](int signal) {
      log_backtrace();
      report_and_exit(SIGSEV_MSG, SIGSEV_MSG);
    };

    std::signal(SIGSEGV, handler);
  }

  extern "C" void install_terminate_and_signal_handlers() {
    set_terminate();
    set_signal_handlers();
  }

  // This method is here mainly for testing purposes. The ability to overwrite the location
  // of failure/error files makes testing easier
  extern "C" void install_terminate_and_signal_handlers_override(char* failure_path, char* error_path) {
    FAILURE_FILEPATH = failure_path;
    ERROR_FILEPATH = error_path;
    set_terminate();
    set_signal_handlers();
  }

}