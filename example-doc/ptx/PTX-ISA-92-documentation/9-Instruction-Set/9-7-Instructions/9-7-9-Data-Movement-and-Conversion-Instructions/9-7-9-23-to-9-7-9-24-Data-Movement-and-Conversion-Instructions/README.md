# 9.7.9.23-24. Data Movement and Conversion Instructions: mapa, getctarank

Covers two cluster-level address mapping instructions. `mapa` maps a shared memory address to the corresponding address in a target CTA within the cluster, supporting both shared::cluster and generic addressing. `getctarank` returns the CTA rank that owns a given shared memory address. Both require sm_90+. Introduced in PTX ISA 7.8.
