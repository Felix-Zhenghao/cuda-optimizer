# 8.11.1. Reductions do not form Acquire Patterns

Documents a special case: atomic reduction instructions (red) do not form acquire patterns with acquire fences, unlike atom instructions. The MP (Message Passing) litmus test demonstrates this — using red instead of atom allows a thread to miss the synchronization guarantee, so atom must be used for correct acquire semantics.
