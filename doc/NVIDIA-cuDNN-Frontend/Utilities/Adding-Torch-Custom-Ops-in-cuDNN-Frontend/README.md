# Adding Torch Custom Ops in cuDNN Frontend

Best practices for registering cuDNN graph operations as PyTorch custom ops using `torch.Library` for minimal CPU overhead. Covers graph caching, UID management, handle management, sorted-pointer execution, autograd integration, and a per-call overhead budget breakdown.
