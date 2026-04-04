# 9.7.9.13. Data Movement and Conversion Instructions: st.bulk

The `st.bulk` instruction initializes a region of shared memory to zero. The size operand specifies bytes to initialize (must be a multiple of 8, max 16MB). Only zero initialization is supported. Uses generic addressing if no state space is specified. Requires sm_100+. Introduced in PTX ISA 8.6.
