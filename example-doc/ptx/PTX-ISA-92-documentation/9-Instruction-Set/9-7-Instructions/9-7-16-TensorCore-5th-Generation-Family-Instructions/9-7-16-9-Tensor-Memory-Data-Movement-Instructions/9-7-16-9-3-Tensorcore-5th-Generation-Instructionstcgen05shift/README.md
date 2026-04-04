# 9.7.16.9.3. Tensorcore 5th Generation Instructions: tcgen05.shift

Documents `tcgen05.shift` for asynchronously shifting rows of a matrix down in Tensor Memory for weight-stationary convolution. Shifts by one row within a warp's Tensor Memory allocation. Completion tracked via mbarrier. Supports cta_group::1 and cta_group::2.
