#pragma once
#include "field.hpp"
#include <CL/sycl.hpp>
#include <random>

void
prepare_random_vector(ff_p254_t* vec, uint64_t n)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(1ul, 1ul << 63);

  for (uint64_t i = 0; i < n; i++) {
    *(vec + i) = ff_p254_t(dis(gen));
  }
}

// Execution time of given SYCL event, which was obtained
// as result of job submission to SYCL queue
//
// Ensure SYCL queue has profiling enabled, otherwise using
// following function should cause panic !
sycl::cl_ulong
time_event(const sycl::event& e)
{
  sycl::cl_ulong start =
    e.get_profiling_info<sycl::info::event_profiling::command_start>();
  sycl::cl_ulong end =
    e.get_profiling_info<sycl::info::event_profiling::command_end>();

  return (end - start);
}
