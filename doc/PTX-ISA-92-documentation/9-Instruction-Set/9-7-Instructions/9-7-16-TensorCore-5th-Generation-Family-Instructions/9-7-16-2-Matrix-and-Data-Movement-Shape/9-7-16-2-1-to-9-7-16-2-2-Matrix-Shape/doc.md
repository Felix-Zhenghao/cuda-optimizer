###### 9.7.16.2.1.1. [Target ISA Note](#tcgen05-matrix-shape-target-isa-note)

* K = 96 is only supported for following architecture-specific targets:

  + `sm_103a`.

---

##### 9.7.16.2.2. [Specifying Matrix Shape](#tcgen05-specify-matrix-shape)

*M* and *N* can be specified in the [Instruction descriptor](#tcgen05-instruction-descriptor).

*K* can be specified explicitly if there are multiple values of *K* supported for a given MMA variant.
Otherwise, if *K* can be uniquely determined as per the [Table 39](#tcgen05-kind-shapes), then *K* cannot
be explicitly specified.
