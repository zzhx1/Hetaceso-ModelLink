# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from modellink.error_utils import check_equal, ensure_valid

# A dictionary of all the memory buffers allocated.
_MEM_BUFFS = dict()


def allocate_mem_buff(name, numel, dtype, track_usage):
    """Allocate a memory buffer."""
    ensure_valid(name not in _MEM_BUFFS, error_message='memory buffer {} already allocated.'.format(name))
    _MEM_BUFFS[name] = MemoryBuffer(name, numel, dtype, track_usage)
    return _MEM_BUFFS[name]


def get_mem_buff(name):
    """Get the memory buffer."""
    return _MEM_BUFFS[name]


class MemoryBuffer:
    """
    Contiguous memory buffer.
    Allocate a contiguous memory of type `dtype` and size `numel`. It is
    used to reduce memory fragmentation.

    Usage: After the allocation, the `_start` index is set tot the first
           index of the memory. A memory chunk starting from `_start` index
           can be `allocated` for an input tensor, with the elements of the
           tensor being coppied. The buffer can be reused by resetting the
           `_start` index.

    """
    def __init__(self, name, numel, dtype, track_usage):
        if torch.distributed.get_rank() == 0:
            element_size = torch.tensor([], dtype=dtype).element_size()
            print('> building the {} memory buffer with {} num elements '
                  'and {} dtype ({:.1f} MB)...'.format(
                      name, numel, dtype, numel * element_size / 1024 / 1024),
                  flush=True)
        self.name = name
        self.numel = numel
        self.dtype = dtype
        self.data = torch.empty(self.numel,
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)

        # Index tracking the start of the free memory.
        self._start = 0

        # Values used for tracking usage.
        self.track_usage = track_usage
        if self.track_usage:
            self.in_use_value = 0.0
            self.total_value = 0.0


    def reset(self):
        """Reset the buffer start index to the beginning of the buffer."""
        self._start = 0


    def is_in_use(self):
        """Whether the current buffer hold on to any memory."""
        return self._start > 0


    def numel_in_use(self):
        """Return number of elements in use."""
        return self._start


    def add(self, tensor):
        """
        Allocate a chunk of memory from the buffer to tensor and copy
        the values.
        """
        error_info = 'Input tensor type {} different from buffer type {}'.format(
                     tensor.dtype, self.dtype)
        check_equal(tensor.dtype, self.dtype, error_info)
        # Number of elements of the input tensor.
        tensor_numel = torch.numel(tensor)
        new_start = self._start + tensor_numel
        ensure_valid(new_start <= self.numel, error_message='Not enough memory left in the buffer ({} > {})'.format(
                                              tensor_numel, self.numel - self._start))
        # New tensor is a view into the memory.
        new_tensor = self.data[self._start:new_start]
        self._start = new_start
        new_tensor = new_tensor.view(tensor.shape)
        new_tensor.copy_(tensor)
        # Return a pointer to the new tensor.
        return new_tensor


    def get_data(self):
        """Return the data currently in use."""
        if self.track_usage:
            self.in_use_value += float(self._start)
            self.total_value += float(self.numel)
        return self.data[:self._start]


    def print_average_usage(self):
        """
        Print memory usage average over time. We would like this value
        to be as high as possible.
        """
        ensure_valid(self.track_usage, error_message='You need to enable track usage.')
        if torch.distributed.get_rank() == 0:
            print(' > usage of {} memory buffer: {:.2f} %'.format(
                self.name, self.in_use_value * 100.0 / self.total_value),
                  flush=True)



class RingMemBuffer:
    """A ring of memory buffers."""

    def __init__(self, name, num_buffers, numel, dtype, track_usage):
        self.num_buffers = num_buffers
        self.buffers = [
            allocate_mem_buff(name + ' {}'.format(i), numel, dtype, track_usage)
            for i in range(num_buffers)]
        self._index = -1


    def get_next_buffer(self):
        self._index += 1
        self._index = self._index % self.num_buffers
        buff = self.buffers[self._index]
        ensure_valid(not buff.is_in_use(), error_message='buffer is already in use.')
        return buff
