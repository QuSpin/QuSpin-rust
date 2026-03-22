// Copyright 2024 Phillip Weinberg

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <concepts>
#include <iostream>
#include <thread>
#include <vector>

namespace quspin { namespace detail {

template<typename Type>
concept WorkTasksInterface = requires(
    Type work_tasks, const std::size_t task_id, const std::size_t thread_id) {
  { work_tasks.size() } -> std::convertible_to<std::size_t>;
  { work_tasks.do_work(task_id, thread_id) } -> std::same_as<void>;
};

enum class Schedule { Interleaved, SequentialBlocks };

template<WorkTasksInterface WorkTasks>
class WorkQueue {
    WorkTasks work_tasks;

  public:

    WorkQueue(WorkTasks work_tasks) : work_tasks(work_tasks) {}

    std::tuple<std::size_t, std::size_t, std::size_t> get_schedule(
        const Schedule schedule, const std::size_t nt,
        const std::size_t num_threads) {
      switch (schedule) {
        case Schedule::Interleaved: {
          return {nt, work_tasks.size(), num_threads};
        }
        case Schedule::SequentialBlocks: {
          const std::size_t num_jobs = work_tasks.size();
          const std::size_t chunk_size = (num_jobs + num_threads) / num_threads;
          const std::size_t start = nt * chunk_size;
          const std::size_t end = std::min(num_jobs, (nt + 1) * chunk_size);
          return {start, end, 1};
        }
      }

      throw std::runtime_error("Invalid schedule");
    }

    void run(const Schedule schedule, const std::size_t num_threads) {
      if (num_threads <= 0) {
        for (std::size_t i = 0; i < work_tasks.size(); i++) {
          this->work_tasks.do_work(i, 0);
        }
        return;
      }

      std::vector<std::thread> workers;
      for (std::size_t nt = 0; nt < num_threads; nt++) {
        auto res = this->get_schedule(schedule, nt, num_threads);
        workers.emplace_back(std::thread([res, this, nt]() {
          const std::size_t start = std::get<0>(res);
          const std::size_t end = std::get<1>(res);
          const std::size_t step = std::get<2>(res);
          for (std::size_t i = start; i < end; i += step) {
            this->work_tasks.do_work(i, nt);
          }
        }));
      }

      for (auto &f : workers) {
        f.join();
      }
    }
};

class atomic_flag {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    std::memory_order order = std::memory_order::relaxed;

  public:

    inline void clear() { std::atomic_flag_clear_explicit(&flag, order); }

    inline bool test_and_set() {
      return std::atomic_flag_test_and_set_explicit(&flag, order);
    }

    inline bool test() {
      return std::atomic_flag_test_and_set_explicit(&flag, order);
    }

    inline void notify_one() { std::atomic_flag_clear_explicit(&flag, order); }

    inline void wait() {
      while (std::atomic_flag_test_and_set_explicit(&flag, order)) {
        std::this_thread::yield();
      }
    }
};

template<std::size_t batch_size = 100, typename Iterator, typename Function>
void async_for_each(Iterator &&begin, Iterator &&end, Function &&f) {
  using task_t = std::remove_reference_t<decltype(*begin)>;
  using batch_t = std::array<task_t, batch_size>;

  std::vector<std::thread> workers;
  std::vector<atomic_flag> thread_done(std::thread::hardware_concurrency());
  std::vector<atomic_flag> thread_empty(std::thread::hardware_concurrency());
  std::vector<atomic_flag> thread_end_loop(std::thread::hardware_concurrency());
  std::vector<std::atomic_size_t> thread_batch_n(
      std::thread::hardware_concurrency());
  std::vector<batch_t> thread_queues(std::thread::hardware_concurrency());

  auto worker = [&f, &thread_queues](atomic_flag &done, atomic_flag &empty,
                                     atomic_flag &end_loop,
                                     std::atomic_size_t &batch_n,
                                     const std::size_t id) {
    auto &queue = thread_queues.at(id);

    while (!end_loop.test_and_set()) {
      // wait for main thread to fill up queue
      empty.wait();

      const std::size_t n_copy = batch_n.load();
      for (std::size_t i = 0; i < n_copy; i++) {
        f(queue.at(i));
      }

      // tell main thread that we are done
      empty.test_and_set();
    }
    done.test_and_set();
  };

  for (std::size_t id = 0; id < std::thread::hardware_concurrency(); id++) {
    thread_done.at(id).clear();
    thread_empty.at(id).test_and_set();
    thread_end_loop.at(id).clear();
    thread_batch_n.at(id).store(0);

    workers.emplace_back(
        worker, std::ref(thread_done.at(id)), std::ref(thread_empty.at(id)),
        std::ref(thread_end_loop.at(id)), std::ref(thread_batch_n.at(id)), id);
  }

  while (begin != end) {
    for (std::size_t id = 0; id < std::thread::hardware_concurrency(); id++) {
      if (thread_empty.at(id).test()) {
        std::size_t batch_n = 0;

        for (; batch_n < thread_queues.at(id).size() && begin != end;
             batch_n++, begin++) {
          thread_queues.at(id).at(batch_n) = *begin;
        }

        thread_batch_n.at(id).store(batch_n);
        thread_empty.at(id).clear();
        thread_empty.at(id).notify_one();
      }
    }
  }

  for (std::size_t id = 0; id < std::thread::hardware_concurrency(); id++) {
    thread_batch_n.at(id).store(0);
    thread_end_loop.at(id).test_and_set();
    thread_empty.at(id).clear();
    thread_empty.at(id).notify_one();
  }

  // wait for all threads to finish
  for (std::size_t id = 0; id < workers.size(); id++) {
    while (!thread_done.at(id).test()) {
      std::this_thread::yield();
    }
    workers.at(id).join();
  }
}
}}  // namespace quspin::detail
