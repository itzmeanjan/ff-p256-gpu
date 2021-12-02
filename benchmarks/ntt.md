## Benchmarking (I)NTT on 254-bit Prime Field

I had access to Nvidia Tesla V100 GPU, so I ran benchmark suite with CUDA backend.

> Note: Following numbers don't include time required to transfer input/ output in between host and device.

```bash
make cuda && ./run
```

```bash
running on Tesla V100-SXM2-16GB

Six-Step FFT

  dimension		          total
     4096		          9.183 ms
     8192		          1.584 ms
    16384		           2.21 ms
    32768		          4.748 ms
    65536		          9.629 ms
   131072		         18.169 ms
   262144		         34.227 ms
   524288		         67.973 ms
  1048576		        134.074 ms
  2097152		        273.136 ms
  4194304		        559.252 ms
  8388608		        1154.74 ms
 16777216		        2370.77 ms

Six-Step (I)FFT

  dimension		          total
     4096		          1.819 ms
     8192		          2.026 ms
    16384		          2.577 ms
    32768		          5.093 ms
    65536		          9.683 ms
   131072		         18.139 ms
   262144		          34.15 ms
   524288		          70.31 ms
  1048576		         137.57 ms
  2097152		        279.791 ms
  4194304		        573.536 ms
  8388608		        1182.05 ms
 16777216		        2424.42 ms
```
