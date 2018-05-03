---
layout: post
title: Distributed GDB
---

{% highlight terminal %}

mpirun -np 4 xterm -e gdb ./a.out
mpirun -np 4 xterm -e ggdb -ex run --args ./a.out arg1 arg2 ...
mpirun -np 4 xterm -e lldb -o run -- ./a.out arg1 arg2 ...

{% endhighlight %}

