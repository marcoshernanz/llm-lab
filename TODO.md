- Tokenizer abstract class
- Model abstract class
- Optimizers
- Scale model up
- Dataset
- Profiler
- Draw graphs
- Gelu activation

- train_chunk is still Python-looped.
  For learning, this is fine. For speed, the next step would be lax.scan or a jitted multi-step function. I still would not rush it unless you feel the runtime pain.

- Full train loss at the end is expensive.
  018_decoder_refactor.py computes full-split train loss and full-split validation loss at the end. That is okay for now, but once the dataset grows, you may want:
  - full validation loss
  - subset train loss or no full train loss
    because full train evaluation is usually the less valuable one.

