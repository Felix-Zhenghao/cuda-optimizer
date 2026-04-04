# 8.11 Special Cases

Documents that atomic reduction operations (red) do not form acquire patterns with acquire fences, unlike atom instructions. Demonstrated with a Message Passing litmus test where red.sys followed by fence.acquire does not synchronize with a release pattern.
