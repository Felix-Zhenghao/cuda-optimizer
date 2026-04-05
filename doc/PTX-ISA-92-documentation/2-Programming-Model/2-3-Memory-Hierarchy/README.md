# 2.3. Memory Hierarchy

PTX threads access multiple memory state spaces: private local, shared (per CTA/cluster), global, constant, param, texture, and surface. Texture and surface memory are cached but not coherent with global writes within the same kernel. Global, constant, and texture spaces persist across kernel launches.
