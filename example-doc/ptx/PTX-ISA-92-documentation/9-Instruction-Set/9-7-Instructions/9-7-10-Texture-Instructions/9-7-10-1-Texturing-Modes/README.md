# 9.7.10.1. Texturing Modes

PTX provides two texturing modes: unified mode (texture and sampler share a single .texref handle, up to 256 samplers) and independent mode (separate handles allowing mixing, but limited to 32 samplers). Mode is selected via .target options texmode_unified or texmode_independent.
