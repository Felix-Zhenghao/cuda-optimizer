# 8.3. State spaces

The memory consistency model relations span all state spaces, but a memory operation's side-effects in one state space are only visible to operations accessing the same state space. This further constrains synchronization scope beyond the named scope qualifiers (.cta, .cluster, .gpu, .sys).
