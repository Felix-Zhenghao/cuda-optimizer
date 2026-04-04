# 9.7.9.25.4.4. Data Movement and Conversion Instructions: multimem.cp.async.bulk

The `multimem.cp.async.bulk` instruction initiates an asynchronous bulk copy from shared::cta to a multimem global address, writing to all GPU memory locations the multimem address references. Size must be a 16-byte multiple with 16-byte alignment. Uses bulk_group completion. Supports optional cp_mask for byte-selective copying. Requires sm_90+; cp_mask requires sm_100+. Introduced in PTX ISA 9.1.
