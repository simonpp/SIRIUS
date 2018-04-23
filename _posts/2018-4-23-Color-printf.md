---
layout: post
title: Color printf
---

{% highlight C %}
if (failed) {
    printf("\x1b[31m" "Failed\n" "\x1b[0m" "\n");
} else {
    printf("\x1b[32m" "OK\n" "\x1b[0m" "\n");
}
{% endhighlight %}

