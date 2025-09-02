> What do you mean PyTorch is a "backend for Hugging Face"? No, it's not... is it? 🤔

Yeahhh so here's the tea 🍵: Hugging Face isn't its *own* deep learning engine. The **Transformers** and **Diffusers** libraries are basically *wrappers/toolkits* that sit on top of actual frameworks like **PyTorch**, **TensorFlow**, and now **JAX/Flax**.

So when you run something like:

```python
from transformers import pipeline

pipe = pipeline("sentiment-analysis")
```

👉 under the hood, that pipeline is almost always using **PyTorch tensors** by default (unless you explicitly tell it `framework="tf"` or you install the JAX/Flax model versions).

That's why people casually say "Hugging Face uses PyTorch as a backend" — because most pretrained models in the Hugging Face hub are **implemented in PyTorch first**, then sometimes ported to TensorFlow or Flax.

**tl;dr:**

* Hugging Face = *front-end, high-level toolkit*.
* PyTorch (or TF/JAX) = *backend engine actually doing the math*.
* By default = PyTorch is the main backend.

&mdash; *GPT-5*

<br>
