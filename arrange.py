import matplotlib.pyplot as plt
import scipy.ndimage as si
import pyopencl as cl
import numpy as np
import os
import cv2

N=16

cl_range = np.zeros(N, dtype=np.int32)

with open('arrange_kernel.cl', 'r') as file:
    kernel = file.read()

# tworzenie kontekstu
ctx = cl.create_some_context()
#tworzenie kolejki
queue = cl.CommandQueue(ctx)
# budowanie programu
program = cl.Program(ctx, kernel).build()

# alokacja pamieci
memory_flags = cl.mem_flags.WRITE_ONLY | cl.mem_flags.ALLOC_HOST_PTR
memory = cl.Buffer(ctx, flags=memory_flags, size=cl_range.nbytes)

# wykonanie programu
kernel = program.arrange(queue, [N], None, memory)
cl.enqueue_copy(queue, cl_range, memory, wait_for=[kernel])

# wypisanie wyniku
print(cl_range)