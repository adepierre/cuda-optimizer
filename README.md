Automated tool to optimize arbitrary cuda code using LLM. The idea is to script the main loop (compile, validate, benchmark, profile) and use the LLM just as a tool to write the optimized cuda code based on the profiling results. This way the LLM can just focus on code writing rather than which commands to run and understanding their outputs.

Don't take this repo as a ready to use product. This was just a way for me to learn and test stuff with local models, sharing the code because why not.


## Results

I tested it using a "simple" MLP forward pass. Inputs and outputs dimensions can change at runtime (with some constraints) while layer sizes and batch size are compile time constants. All dimensions but the batch size are small, so all multiplications are between very rectangular matrices, far from the square case that's used most of the time all over the internet. Initial implementation is a very naive one on purpose to let the model explore freely. I used a quantized version of Qwen 3.8 27b, with thinking set to medium, so not quite the latest frontier model level with trillions of parameters but results are still pretty decent going from 10.87ms for the baseline implementation to 0.49ms at the end. Only one iteration failed due to a runtime crash during benchmark, the 115 other iterations generated valid code within the 5 retry attempts. The final version of the code can be found in ``cuda/optimized/impl.cu``.

<img width="1800" height="750" alt="timing_history_tokens" src="https://github.com/user-attachments/assets/9e0e4634-1bcc-44b7-8855-d56a9077f34c" />


## Custom cuda code

If you run a local model, make sure it doesn't run on the default GPU as this is the one that will be used to run the cuda code.

- edit ``cuda/common/interface_specific.hpp`` to define your problem specific dimensions, tensors etc... You can also edit the permutation structures to reflect that some tensors could be stored in a different layout (transposition will happen before/after the benchmark so it's not part of the timed section)
- edit ``cuda/reference/impl.cu`` with your custom code
- edit ``config.py``, mostly experiment name, tool paths/commands (cmake, ncu, git), binary (add .exe on windows for example), the LLM URL and the number of iterations
- run ``python optimize.py``

It will clean any previous run, create a new branch, initialize the optimized cuda code with your baseline implementation and start the optimization loop. Every time a version of the code performs better than the previous best one, a commit with the code will be created so you can easily retrace the history of the code.

