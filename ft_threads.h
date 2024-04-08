#ifndef FT_THREADS_H
#define FT_THREADS_H

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace ft {

class ThreadPool {

public:
  using pointer = std::shared_ptr<ThreadPool>;

  ThreadPool(uint32_t threadCount)
      : _threads(std::vector<std::thread>(threadCount)),
        _busyThreads(threadCount), _stop(false) {

    for (uint32_t i = 0; i < threadCount; ++i)
      _threads[i] = std::thread(Worker(this));
  }

  ~ThreadPool() {

    {
      std::lock_guard<std::mutex> lock(_mutex);
      _stop = true;
      _conditionVariable.notify_all();
    }
    joinAll();
  }

  uint32_t queueSize() const {
    std::scoped_lock<std::mutex> lock(_mutex);
    return _queue.size();
  }

  template <typename F, typename... Args>
  auto addTask(F &&f, Args &&...args) -> std::future<decltype(f(args...))> {
    // get the function
    auto func = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    // create a shared_pointer to a packaged task and pass the function as the
    // entry
    auto pTask =
        std::make_shared<std::packaged_task<decltype(f(args...))()>>(func);

    // wrap the function into a void() ; we'll use lambda
    auto wrapper = [pTask]() { (*pTask)(); };

    // add the new wrapped task into the queue
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _queue.push(wrapper);
      _conditionVariable.notify_all();
    }

    return pTask->get_future();
  }

  void joinAll() {
    for (auto &t : _threads) {
      if (t.joinable())
        t.join();
    }
  }

private:
  struct Worker {

    Worker(ThreadPool *pool) : _pool(pool) {}
    void operator()() {
      std::unique_lock<std::mutex> lock(_pool->_mutex);
      while (!_pool->_stop || (_pool->_stop && !_pool->_queue.empty())) {

        _pool->_busyThreads--;
        _pool->_conditionVariable.wait(lock, [this] {
          return this->_pool->_stop || !this->_pool->_queue.empty();
        });
        _pool->_busyThreads++;
        if (!_pool->_queue.empty()) {
          auto func = _pool->_queue.front();
          _pool->_queue.pop();
          lock.unlock();
          func();
          lock.lock();
        }
      }
    };

    ThreadPool *_pool;
  };

  mutable std::mutex _mutex;
  std::condition_variable _conditionVariable;
  std::vector<std::thread> _threads;
  std::queue<std::function<void()>> _queue;
  int _busyThreads;
  bool _stop;
};

} // namespace ft

#endif // !FT_THREAD_H
